"""
上下文配置模块：管理『项目级人名词表 / 常用术语』与『任务级临时补充』。

主要职责：
1. 加载 / 保存项目级默认词表（output/context/glossary.json）
2. 把默认词表 + 任务级补充拼成 Whisper 的 initial_prompt（限制在 ~200 token 之内）
3. 计算 prompt 哈希，用于断点命中校验（prompt 改了断点自动失效）
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("decodevoice")

# ─── 路径常量 ───
CONTEXT_DIR = Path("output/context")
GLOSSARY_FILE = CONTEXT_DIR / "glossary.json"

# Whisper 解码器对 initial_prompt 的硬上限（按 token 估算，约 224 token）
# 留 16 token 缓冲，按平均 1.5 字符/token 算 ≈ 320 字符
PROMPT_CHAR_LIMIT = 320


@dataclass
class Glossary:
    """项目级默认人名 / 术语词表。"""

    # 会议 / 录音的背景描述（自由文本）
    background: str = ""
    # 说话人列表：[{ "name": "张三", "alias": ["老张", "张工"] }, ...]
    people: list[dict] = field(default_factory=list)
    # 关键词 / 术语 / 品牌名 等
    terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "background": self.background,
            "people": self.people,
            "terms": self.terms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Glossary":
        return cls(
            background=(data or {}).get("background", "") or "",
            people=list((data or {}).get("people") or []),
            terms=list((data or {}).get("terms") or []),
        )


def load_glossary() -> Glossary:
    """从 output/context/glossary.json 读取项目级词表。文件不存在或损坏时返回空词表。"""
    if not GLOSSARY_FILE.exists():
        return Glossary()
    try:
        data = json.loads(GLOSSARY_FILE.read_text(encoding="utf-8"))
        return Glossary.from_dict(data)
    except Exception as e:
        logger.warning(f"读取词表失败，使用空词表: {e}")
        return Glossary()


def save_glossary(glossary: Glossary) -> None:
    """把项目级词表落盘到 output/context/glossary.json。"""
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    GLOSSARY_FILE.write_text(
        json.dumps(glossary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"项目级词表已保存: {GLOSSARY_FILE}")


def hash_prompt(prompt: str) -> str:
    """计算 prompt 内容的短哈希（用于断点命中校验）。"""
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()[:12]


def build_initial_prompt(
    glossary: Glossary,
    task_background: str = "",
    task_people: Iterable[str] | None = None,
    task_terms: Iterable[str] | None = None,
) -> str:
    """
    把项目级默认词表 + 任务级临时补充拼成 Whisper 的 initial_prompt。

    规则：
    - 输出形如： "<背景>。参会人: <人名1, 人名2>。关键术语: <术语1, 术语2>。"
    - 人名 / 术语会去重，优先保留项目级定义，任务级补充的放后面
    - 整体长度受 PROMPT_CHAR_LIMIT 限制，超长时按 background → people → terms 优先级截断
    - 若最终 prompt 为空，返回空串（调用方需处理「无上下文」的情况）

    参数:
        glossary: 项目级默认词表
        task_background: 任务级临时背景（覆盖项目级 background 之外的补充）
        task_people: 任务级临时参会人名（追加在项目级 people 之后）
        task_terms: 任务级临时关键词（追加在项目级 terms 之后）

    返回:
        拼好的 initial_prompt 字符串
    """
    # 合并背景：项目级在前，任务级在后（用句号分隔）
    bg_parts = [s for s in [glossary.background.strip(), task_background.strip()] if s]
    background = "。".join(bg_parts)

    # 合并参会人（去重，保留出现顺序）
    seen_names: set[str] = set()
    people_lines: list[str] = []

    def _add_person(name: str, aliases: list[str] | None = None) -> None:
        name = (name or "").strip()
        if not name or name in seen_names:
            return
        seen_names.add(name)
        aliases_clean = [a.strip() for a in (aliases or []) if a and a.strip()]
        if aliases_clean:
            people_lines.append(f"{name}({'/'.join(aliases_clean)})")
        else:
            people_lines.append(name)

    for p in glossary.people:
        if isinstance(p, dict):
            _add_person(p.get("name", ""), p.get("alias") or [])
        elif isinstance(p, str):
            _add_person(p)

    for name in task_people or []:
        _add_person(name)

    # 合并术语（去重，保留顺序）
    seen_terms: set[str] = set()
    term_lines: list[str] = []
    for t in list(glossary.terms) + list(task_terms or []):
        t = (t or "").strip()
        if not t or t in seen_terms:
            continue
        seen_terms.add(t)
        term_lines.append(t)

    # 拼接
    parts: list[str] = []
    if background:
        parts.append(background)
    if people_lines:
        parts.append("参会人: " + "、".join(people_lines))
    if term_lines:
        parts.append("关键术语: " + "、".join(term_lines))

    prompt = "。".join(parts)
    if not prompt:
        return ""

    # 限制长度
    if len(prompt) > PROMPT_CHAR_LIMIT:
        logger.warning(
            f"initial_prompt 超出 {PROMPT_CHAR_LIMIT} 字符限制（当前 {len(prompt)}），将截断"
        )
        prompt = prompt[:PROMPT_CHAR_LIMIT].rstrip("，,。;；:：") + "。"

    return prompt
