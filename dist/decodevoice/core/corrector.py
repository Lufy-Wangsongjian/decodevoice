"""
LLM 纠错模块：转录完成后用云端 LLM 对分段文本做按段改写。

支持的 provider：
- hunyuan (腾讯混元 OpenAI 兼容接口, ChatCompletion)
- deepseek (DeepSeek, OpenAI 兼容接口)

设计原则：
1. 只改写 SpeakerSegment.text；start / end / speaker 字段必须保持不变
2. 整段拼接后一次性发给 LLM（避免 N 次调用），批处理时按 token 限制自动分批
3. 返回值仍是 SpeakerSegment 列表，方便 caller 直接替换 result.segments
4. 不依赖第三方 SDK：用 requests 直接调 HTTP 接口
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from .diarizer import SpeakerSegment

logger = logging.getLogger("decodevoice")

# ─── 路径常量 ───
CONTEXT_DIR = Path("output/context")
LLM_CONFIG_FILE = CONTEXT_DIR / "llm_config.json"

# ─── Provider 预设 ───
PROVIDER_PRESETS: dict[str, dict] = {
    "hunyuan": {
        "label": "腾讯混元 (hunyuan-pro)",
        "base_url": "https://api.hunyuan.tencent.com/v1",
        "default_model": "hunyuan-pro",
    },
    "deepseek": {
        "label": "DeepSeek (deepseek-chat)",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    "custom": {
        "label": "自定义（OpenAI 兼容）",
        "base_url": "",
        "default_model": "",
    },
}


@dataclass
class LLMConfig:
    """LLM 纠错所需的运行时配置。"""

    provider: str = "deepseek"        # hunyuan / deepseek / custom
    api_key: str = ""
    base_url: str = ""                # 自定义时必填
    model: str = ""                   # 留空时使用 provider 预设的 default_model
    timeout: int = 60                 # 单次请求超时（秒）
    temperature: float = 0.2
    max_batch_chars: int = 6000       # 单次请求的文本上限（字符数），超长自动分批
    enabled: bool = False             # 是否启用纠错（侧边栏开关）

    def resolve_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        preset = PROVIDER_PRESETS.get(self.provider, {})
        return (preset.get("base_url") or "").rstrip("/")

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        preset = PROVIDER_PRESETS.get(self.provider, {})
        return preset.get("default_model", "")

    def to_dict(self) -> dict:
        # api_key 不进日志，但落盘时仍保存（用户授权磁盘保存）
        return {
            "provider": self.provider,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "temperature": self.temperature,
            "max_batch_chars": self.max_batch_chars,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "LLMConfig":
        data = data or {}
        return cls(
            provider=data.get("provider", "deepseek"),
            api_key=data.get("api_key", "") or "",
            base_url=data.get("base_url", "") or "",
            model=data.get("model", "") or "",
            timeout=int(data.get("timeout", 60) or 60),
            temperature=float(data.get("temperature", 0.2) or 0.2),
            max_batch_chars=int(data.get("max_batch_chars", 6000) or 6000),
            enabled=bool(data.get("enabled", False)),
        )


def load_llm_config() -> LLMConfig:
    """读取 llm_config.json，文件不存在时返回默认配置（enabled=False）。"""
    if not LLM_CONFIG_FILE.exists():
        return LLMConfig()
    try:
        data = json.loads(LLM_CONFIG_FILE.read_text(encoding="utf-8"))
        return LLMConfig.from_dict(data)
    except Exception as e:
        logger.warning(f"读取 LLM 配置失败，使用默认: {e}")
        return LLMConfig()


def save_llm_config(cfg: LLMConfig) -> None:
    """把 LLM 配置落盘。"""
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    LLM_CONFIG_FILE.write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"LLM 配置已保存: {LLM_CONFIG_FILE}")


# ─── 改写 Prompt 模板 ───
_SYSTEM_PROMPT = """你是一名专业的中文语音转文字校对员，负责对 ASR 自动转录的结果做最小侵入式纠错。

【核心原则】
1. **专有名词优先**：根据下方提供的「参会人 / 关键术语 / 背景」纠正听写错误的人名、品牌、术语、缩写。
2. **保持原意**：不得删减、改写、扩写、总结任何原句内容；只替换错别字、补充漏字、修正同音字。
3. **时间戳不可触碰**：每段有固定编号 i，请保持原样输出。
4. **不确定时保守**：如果听写结果本身合理、只是和词表对不上，不要强行替换。
5. **口语连写**：保留 ASR 常见的口语连写、口头禅；不要把"嗯""啊"等语气词全部清除。
6. **不出现幻觉**：不要添加原文中没有的信息，不要改数字、金额、日期、邮箱、链接等。

【输出格式】
- 严格按 JSON 数组输出： [{"i": 0, "text": "..."}, {"i": 1, "text": "..."}, ...]
- 数组中必须包含输入的每一个 i，按 i 升序排列，不要省略任何段落。
- 严禁在 JSON 之外输出任何解释、注释、Markdown 代码块。"""


def _build_user_prompt(
    segments: list[SpeakerSegment],
    background: str,
    people: list[str],
    terms: list[str],
) -> str:
    parts: list[str] = []
    if background:
        parts.append(f"【会议背景】\n{background}")
    if people:
        parts.append("【参会人】\n" + "、".join(people))
    if terms:
        parts.append("【关键术语 / 品牌 / 人名】\n" + "、".join(terms))

    parts.append("【待校对段落】（i 为段落编号，start/end 为时间戳，请勿修改时间和 speaker 标签）")
    for i, seg in enumerate(segments):
        spk = seg.speaker if seg.speaker and seg.speaker != "UNKNOWN" else "?"
        parts.append(
            f'{{"i": {i}, "speaker": "{spk}", '
            f'"start": {seg.start:.2f}, "end": {seg.end:.2f}, '
            f'"text": {json.dumps(seg.text, ensure_ascii=False)}}}'
        )

    parts.append(
        "\n【输出】\n请返回与输入 i 一一对应的 JSON 数组，仅包含 i 和 text 字段。"
    )
    return "\n\n".join(parts)


# ─── JSON 解析兜底 ───
def _extract_json_array(text: str) -> list[dict]:
    """从 LLM 响应里抽出 JSON 数组。容忍：```json 包裹、前后说明文字。"""
    text = text.strip()

    # 去掉 ```json ... ``` 或 ``` ... ```
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    # 直接尝试
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 退化：找第一个 [ 到最后一个 ] 的子串
    lb = text.find("[")
    rb = text.rfind("]")
    if lb != -1 and rb != -1 and rb > lb:
        try:
            data = json.loads(text[lb:rb + 1])
            if isinstance(data, list):
                return data
        except Exception as e:
            logger.debug(f"提取 JSON 数组失败: {e}")

    return []


# ─── HTTP 调用 ───
def _call_chat_completion(
    cfg: LLMConfig,
    messages: list[dict],
) -> str:
    """调用 OpenAI 兼容的 chat/completions 接口。"""
    base_url = cfg.resolve_base_url()
    model = cfg.resolve_model()
    if not base_url:
        raise RuntimeError("未配置 LLM base_url")
    if not cfg.api_key:
        raise RuntimeError("未配置 LLM api_key")
    if not model:
        raise RuntimeError("未配置 LLM model")

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": cfg.temperature,
        "stream": False,
    }

    t0 = time.time()
    resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout)
    elapsed = time.time() - t0

    if resp.status_code != 200:
        raise RuntimeError(
            f"LLM 请求失败 (HTTP {resp.status_code}, 耗时 {elapsed:.1f}s): {resp.text[:300]}"
        )

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"LLM 响应格式异常: {data}") from e

    logger.info(
        f"LLM 调用成功 (provider={cfg.provider}, model={model}, 耗时 {elapsed:.1f}s, "
        f"返回 {len(content)} 字符)"
    )
    return content


# ─── 批处理：按 max_batch_chars 把 segments 切成多组 ───
def _batched(
    segments: list[SpeakerSegment], max_chars: int
) -> list[list[SpeakerSegment]]:
    """把 segments 按累计字符数切成多批，确保每批总字符数 ≤ max_chars。"""
    batches: list[list[SpeakerSegment]] = []
    cur: list[SpeakerSegment] = []
    cur_len = 0
    # 每段预留约 80 字符给 JSON 包装
    per_seg_overhead = 80

    for seg in segments:
        seg_len = len(seg.text) + per_seg_overhead
        if cur and cur_len + seg_len > max_chars:
            batches.append(cur)
            cur = []
            cur_len = 0
        cur.append(seg)
        cur_len += seg_len
    if cur:
        batches.append(cur)
    return batches


# ─── 主入口 ───
class LLMCorrector:
    """封装 LLM 二次校对的客户端。

    用法：
        corrector = LLMCorrector(config)
        corrected = corrector.correct(segments, background, people, terms)
        # corrected 与 segments 等长，time/speaker 一一对应，text 可能被改写
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    def correct(
        self,
        segments: list[SpeakerSegment],
        background: str = "",
        people: list[str] | None = None,
        terms: list[str] | None = None,
        progress_callback=None,
    ) -> list[SpeakerSegment]:
        """返回改写后的 segments（保持 start/end/speaker 不变，text 可能被改写）。

        失败时（如 API 错误）回退到原始 segments，并在日志中记录原因。
        """
        if not segments:
            return list(segments)

        if not self.config.enabled:
            logger.debug("LLM 纠错未启用，跳过")
            return list(segments)

        # 校验
        try:
            self.config.resolve_base_url()
            if not self.config.api_key:
                raise RuntimeError("API key 为空")
            if not self.config.resolve_model():
                raise RuntimeError("model 为空")
        except Exception as e:
            logger.warning(f"LLM 配置不完整，跳过纠错: {e}")
            return list(segments)

        # 准备上下文
        people_list = [p for p in (people or []) if p]
        terms_list = [t for t in (terms or []) if t]

        batches = _batched(segments, max_chars=self.config.max_batch_chars)
        logger.info(
            f"LLM 纠错开始: {len(segments)} 段 → {len(batches)} 批, "
            f"provider={self.config.provider}, model={self.config.resolve_model()}"
        )

        corrected_texts: list[str | None] = [None] * len(segments)
        total_batches = len(batches)
        offset = 0

        for batch_idx, batch in enumerate(batches, start=1):
            user_prompt = _build_user_prompt(
                batch, background, people_list, terms_list
            )
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            try:
                content = _call_chat_completion(self.config, messages)
                parsed = _extract_json_array(content)
            except Exception as e:
                logger.error(f"LLM 第 {batch_idx}/{total_batches} 批调用失败: {e}")
                parsed = []

            # 把改写结果按 i 落到对应位置
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                if "i" not in item or "text" not in item:
                    continue
                try:
                    idx = int(item["i"])
                except (TypeError, ValueError):
                    continue
                global_idx = offset + idx
                if 0 <= global_idx < len(segments) and isinstance(item["text"], str):
                    corrected_texts[global_idx] = item["text"]

            offset += len(batch)
            if progress_callback:
                progress_callback(
                    f"LLM 纠错 {batch_idx}/{total_batches}",
                    batch_idx / total_batches,
                )

        # 构造结果：能改写的就改写；解析失败的段保留原 text
        result: list[SpeakerSegment] = []
        changed_count = 0
        for i, seg in enumerate(segments):
            new_text = corrected_texts[i]
            if new_text is not None and new_text.strip() and new_text != seg.text:
                changed_count += 1
                result.append(
                    SpeakerSegment(
                        speaker=seg.speaker,
                        text=new_text,
                        start=seg.start,
                        end=seg.end,
                    )
                )
            else:
                result.append(seg)

        logger.info(
            f"LLM 纠错完成: 共 {len(segments)} 段，实际改写 {changed_count} 段"
        )
        return result
