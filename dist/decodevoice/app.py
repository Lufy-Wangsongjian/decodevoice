"""
DecodeVoice — 语音转文字 + 说话人分离
========================================

使用方式:
    streamlit run app.py

功能:
    1. 上传音频/视频文件
    2. 自动提取音频（视频文件）
    3. Whisper 语音转录
    4. 说话人识别与分段
    5. 导出带说话人标注的文字结果
"""

import hashlib
import json
import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv()
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import streamlit as st

from core.extractor import extract_audio
from core.diarizer import AudioTranscriber, TranscriptionResult
from core.context import (
    CONTEXT_DIR,
    Glossary,
    build_initial_prompt,
    hash_prompt,
    load_glossary,
    save_glossary,
)
from core.corrector import (
    LLMConfig,
    LLMCorrector,
    PROVIDER_PRESETS,
    load_llm_config,
    save_llm_config,
)


# 初始化 context 目录
CONTEXT_DIR.mkdir(parents=True, exist_ok=True)


# ─── 历史记录管理 ───
HISTORY_FILE = Path("output/history.json")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return []
    return []


def save_history(entries: list[dict]):
    HISTORY_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2))


def save_history_entry(entry: dict):
    history = load_history()
    # 用 original_name 去重（同一文件多次保存只更新同一条，避免临时/持久路径产生重复）
    existing = next((h for h in history if h.get("original_name") == entry.get("original_name")), None)
    if existing:
        # 合并：保留已有字段（如 result_files），用新值覆盖
        existing.update({k: v for k, v in entry.items() if v not in (None, "", 0) or k not in existing})
    else:
        history.insert(0, entry)
    # 只保留最近 50 条
    save_history(history[:50])



# ─── 辅助函数（必须在页面逻辑之前定义）───
def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments) -> str:
    """将分段结果转为 SRT 字幕格式"""
    lines = []
    for i, seg in enumerate(segments, start=1):
        speaker_tag = f"[{seg.speaker}] " if seg.speaker and seg.speaker != "UNKNOWN" else ""
        lines.append(str(i))
        lines.append(f"{_srt_time(seg.start)} --> {_srt_time(seg.end)}")
        lines.append(f"{speaker_tag}{seg.text}")
        lines.append("")
    return "\n".join(lines)

# ─── 日志捕获：将 logger 输出同时写入内存缓冲区 ───
log_buffer = StringIO()

class LogCapture:
    """捕获日志到内存缓冲区，供 Streamlit 实时展示"""
    def __init__(self):
        self.buffer = log_buffer

    def write(self, msg: str):
        if msg.strip():
            self.buffer.write(msg)
            self.buffer.write("\n")

    def flush(self):
        pass

# 将 decodevoice logger 的输出接一份到缓冲区
log_handler = logging.StreamHandler(LogCapture())
log_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S"
))
logging.getLogger("decodevoice").addHandler(log_handler)

# ─── 页面配置 ───
st.set_page_config(
    page_title="DecodeVoice - 语音转文字",
    page_icon="",
    layout="wide",
)

# ─── 样式 ───
st.markdown("""
<style>
    .speaker-segment {
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 8px;
        border-left: 4px solid;
    }
    .speaker-0 { background: #e3f2fd; border-color: #1976d2; }
    .speaker-1 { background: #e8f5e9; border-color: #388e3c; }
    .speaker-2 { background: #fff3e0; border-color: #f57c00; }
    .speaker-3 { background: #fce4ec; border-color: #c62828; }
    .speaker-4 { background: #f3e5f5; border-color: #7b1fa2; }
    .timestamp { color: #666; font-size: 0.85em; }
    .speaker-label { font-weight: 700; font-size: 0.9em; margin-bottom: 4px; }
    .segment-text { font-size: 1.05em; line-height: 1.6; color: #333; }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ─── 标题 ───
st.title("DecodeVoice")
st.caption("提取音频/视频中的语音内容，按说话人自动分段")

# ─── 侧边栏：设置 ───
with st.sidebar:
    st.header("模型设置")

    model_size = st.selectbox(
        "Whisper 模型",
        options=AudioTranscriber.AVAILABLE_MODELS,
        index=AudioTranscriber.AVAILABLE_MODELS.index("large-v3"),
        help="模型越大精度越高，但速度越慢、显存占用越大。large-v3 效果最佳。",
    )

    language = st.text_input(
        "语言（留空自动检测）",
        value="zh",
        placeholder="zh / en / ja / ko ...",
        help="指定语言可提高准确率。中文用 zh，英文用 en。",
    )

    hf_token = st.text_input(
        "HuggingFace Token",
        value=os.environ.get("HF_TOKEN", ""),
        type="password",
        help="说话人分离功能需要。去 huggingface.co/settings/tokens 创建。",
    )

    enable_diarization = st.checkbox(
        "启用说话人分离",
        value=bool(hf_token),
        help="需要 HuggingFace Token 并接受 pyannote 模型使用协议。",
    )

    st.divider()

    min_speakers = st.number_input("最少说话人数", min_value=1, max_value=20, value=1)
    max_speakers = st.number_input("最多说话人数", min_value=1, max_value=20, value=4)

    # ─── 上下文纠错：LLM 配置 ───
    st.divider()
    st.markdown("**上下文纠错（LLM 二次校对）**")
    st.caption(
        "可选：转录完成后用云端 LLM 按段改写专有名词 / 术语 / 同音字，"
        "时间戳与说话人标签保持不变。"
    )

    _llm_cfg = load_llm_config()
    _llm_enabled = st.checkbox(
        "启用 LLM 纠错",
        value=_llm_cfg.enabled,
        help="关闭时只走 Whisper initial_prompt 注入，不调用任何 LLM。",
    )

    _provider_keys = list(PROVIDER_PRESETS.keys())
    _provider_labels = [PROVIDER_PRESETS[k]["label"] for k in _provider_keys]
    _current_provider_idx = (
        _provider_keys.index(_llm_cfg.provider)
        if _llm_cfg.provider in _provider_keys
        else _provider_keys.index("deepseek")
    )
    _provider_label = st.selectbox(
        "Provider",
        options=_provider_labels,
        index=_current_provider_idx,
    )
    _provider = _provider_keys[_provider_labels.index(_provider_label)]

    _api_key = st.text_input(
        "API Key",
        value=_llm_cfg.api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
        type="password",
        help="将以明文保存到 output/context/llm_config.json。留空时自动使用 DEEPSEEK_API_KEY 环境变量。",
    )
    _model_default = _llm_cfg.model or PROVIDER_PRESETS[_provider]["default_model"]
    _model = st.text_input(
        "Model",
        value=_model_default,
        help="留空时使用 provider 预设的默认模型。",
    )
    _base_url = st.text_input(
        "Base URL（仅自定义 provider 需要）",
        value=_llm_cfg.base_url if _provider == "custom" else PROVIDER_PRESETS[_provider]["base_url"],
        help="OpenAI 兼容接口的基础 URL，例如 https://api.deepseek.com/v1",
        disabled=(_provider != "custom"),
    )
    _llm_temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0,
        value=float(_llm_cfg.temperature), step=0.05,
    )
    _llm_max_batch = st.number_input(
        "单批最大字符数", min_value=1000, max_value=32000,
        value=int(_llm_cfg.max_batch_chars), step=500,
        help="单次 LLM 请求承载的最大文本量；超出将自动分批。",
    )

    if st.button("保存 LLM 配置", use_container_width=True):
        save_llm_config(LLMConfig(
            provider=_provider,
            api_key=_api_key.strip(),
            base_url=_base_url.strip() if _provider == "custom" else "",
            model=_model.strip(),
            timeout=int(_llm_cfg.timeout),
            temperature=float(_llm_temperature),
            max_batch_chars=int(_llm_max_batch),
            enabled=bool(_llm_enabled),
        ))
        st.success("LLM 配置已保存")
        st.rerun()

    # 把当前 UI 状态打包回 session，方便主流程读取最新值（即使没点保存）
    st.session_state["llm_runtime"] = {
        "enabled": bool(_llm_enabled),
        "provider": _provider,
        "api_key": _api_key.strip(),
        "base_url": _base_url.strip() if _provider == "custom" else "",
        "model": _model.strip() or PROVIDER_PRESETS[_provider]["default_model"],
        "temperature": float(_llm_temperature),
        "max_batch_chars": int(_llm_max_batch),
    }

    # ─── 上下文纠错：项目级人名 / 术语词表 ───
    st.divider()
    st.markdown("**项目级默认词表**")
    st.caption(
        "维护一份长期复用的人名 / 术语 / 品牌词表。"
        "上传音频时会被拼到 Whisper initial_prompt 中。"
    )

    _glossary = load_glossary()
    _new_bg = st.text_area(
        "会议背景（可选）",
        value=_glossary.background,
        height=80,
        placeholder="例：产品周会，讨论 Q3 路线图和研发资源",
        help="说明这次会议的主题，参与方等背景信息。",
    )
    _people_text = st.text_area(
        "参会人（每行一个，可写「姓名 别名1 别名2」）",
        value="\n".join(
            p.get("name", "") + (" " + " ".join(p.get("alias") or []) if p.get("alias") else "")
            for p in _glossary.people
        ),
        height=120,
        placeholder="例：\n张三 老张 张工\n李四",
        help="每行一个；多个别名用空格分隔。",
    )
    _terms_text = st.text_area(
        "关键术语 / 品牌 / 缩写（每行一个）",
        value="\n".join(_glossary.terms),
        height=100,
        placeholder="例：\nProject Atlas\nDecodeVoice\nPyAnnote",
    )

    if st.button("保存项目级词表", use_container_width=True):
        people_list = []
        for line in _people_text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            name = parts[0]
            alias = parts[1:] if len(parts) > 1 else []
            people_list.append({"name": name, "alias": alias})
        terms_list = [t.strip() for t in _terms_text.splitlines() if t.strip()]
        save_glossary(Glossary(
            background=_new_bg.strip(),
            people=people_list,
            terms=terms_list,
        ))
        st.success("项目级词表已保存")
        st.rerun()

    st.divider()
    st.markdown("**断点状态**")
    ckpt_dir = Path("output/checkpoints")
    if ckpt_dir.exists():
        ckpt_files = sorted(ckpt_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpt_files:
            # 按文件内容哈希分组
            from collections import defaultdict
            groups = defaultdict(list)
            for f in ckpt_files:
                prefix = f.stem.split("_")[1]  # ckpt_{hash}_{model}_{lang}_{stage}.pkl
                groups[prefix].append(f)
            st.caption(f"共 {len(ckpt_files)} 个断点文件（{len(groups)} 组）")
            for prefix, files in list(groups.items())[:3]:
                stages = [f.stem.split("_")[-1] for f in files]
                age_min = int((time.time() - min(f.stat().st_mtime for f in files)) / 60)
                st.caption(f"· {prefix[:8]}... → {', '.join(stages)} ({age_min}分钟前)")
        else:
            st.caption("暂无断点（转录完成后自动保存）")
    else:
        st.caption("暂无断点")

    st.divider()
    st.markdown("**硬件信息**")
    import torch
    device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.info(f"当前设备: {device}")

    st.divider()
    st.markdown("**预加载模型**")
    st.caption("提前把 large-v3 模型加载到内存，后续转录跳过加载步骤（节省 30-60s）。")

    st.caption(f"缓存状态: {'已加载' if AudioTranscriber.is_model_cached('large-v3', 'cuda' if torch.cuda.is_available() else 'cpu', 'float16' if torch.cuda.is_available() else 'int8') else '未加载'}")

    if st.button("预加载 large-v3 模型", use_container_width=True):
        with st.spinner("正在下载/加载模型（首次约 1-3 GB，请耐心等待）..."):
            try:
                AudioTranscriber.preload_model(
                    model_size="large-v3",
                    language=language.strip() if language.strip() else None,
                    progress_callback=None,
                )
                st.success("模型 large-v3 已加载到缓存！")
                st.rerun()
            except Exception as e:
                st.error(f"模型加载失败: {e}")

    st.divider()
    st.markdown("**断点续传**")
    force_restart = st.checkbox(
        "忽略断点，重新处理",
        value=False,
        help="勾选后将跳过已有断点，从头开始完整转录。",
    )

    # ─── 历史记录 ───
    st.divider()
    st.markdown("**历史记录**")
    history = load_history()
    if history:
        # 注意：不再因音频文件不存在而删除历史记录。
        # 历史记录是持久的，仅根据文件可用性决定按钮是否可点。
        for i, h in enumerate(history[:10]):
            name = h.get("original_name", "未知")
            status = h.get("status", "unknown")
            ts = h.get("timestamp", "")

            status_icon = {"completed": "✅", "error": "❌", "processing": "⏳"}.get(status, "📄")
            detail = f"{format_time(h.get('duration', 0))}" if h.get("duration") else ""
            age = ""
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    minutes = int((datetime.now() - dt).total_seconds() / 60)
                    age = f"{minutes}分钟前" if minutes < 1440 else f"{minutes // 1440}天前"
                except Exception:
                    pass

            audio_exists = bool(h.get("audio_path")) and Path(h["audio_path"]).exists()
            result_files = {k: v for k, v in (h.get("result_files") or {}).items()
                            if Path(v).exists()}

            st.caption(f"{status_icon} {name} · {detail} · {age}")
            col1, col2, col3 = st.columns(3)
            if col1.button("继续转录", key=f"resume_{i}",
                           disabled=not audio_exists, use_container_width=True):
                st.session_state.resume_audio_path = h["audio_path"]
                st.session_state.resume_original_name = h["original_name"]
                st.session_state.view_result = None
                st.rerun()
            if col2.button("查看结果", key=f"view_{i}",
                           disabled=not result_files, use_container_width=True):
                st.session_state.view_result = result_files
                st.session_state.view_result_name = name
                st.session_state.resume_audio_path = None
                st.session_state.resume_original_name = None
                st.rerun()
            if col3.button("删除", key=f"del_{i}", use_container_width=True):
                # 仅从历史移除，不删除已保存的结果文件
                history.pop(i)
                save_history(history)
                st.rerun()
            if not audio_exists and not result_files:
                st.caption("　⚠️ 音频与结果文件均已删除，仅保留记录")
    else:
        st.caption("暂无历史记录")

# ─── 主区域 ───

# 初始化 session state
if "resume_audio_path" not in st.session_state:
    st.session_state.resume_audio_path = None
if "resume_original_name" not in st.session_state:
    st.session_state.resume_original_name = None
if "view_result" not in st.session_state:
    st.session_state.view_result = None
if "view_result_name" not in st.session_state:
    st.session_state.view_result_name = None

# ─── 查看历史结果（不进入转录流程）───
if st.session_state.view_result:
    files = st.session_state.view_result
    st.subheader(f"历史结果：{st.session_state.view_result_name}")
    if st.button("← 返回", use_container_width=False):
        st.session_state.view_result = None
        st.session_state.view_result_name = None
        st.rerun()
    for label, fpath in files.items():
        p = Path(fpath)
        if not p.exists():
            continue
        content = p.read_text(encoding="utf-8")
        with st.expander(f"{label}（{p.name}）", expanded=(label == "说话人标注")):
            preview = content[:50000]
            if len(content) > 50000:
                preview += "\n...（内容过长已截断，请下载查看完整文件）"
            st.code(preview, language=None)
        st.download_button(
            f"下载 {label}",
            data=content,
            file_name=p.name,
            mime="text/plain",
            key=f"dl_view_{label}",
            use_container_width=True,
        )
    st.stop()

resume_path = st.session_state.resume_audio_path
resume_name = st.session_state.resume_original_name

# 从历史恢复
if resume_path and resume_name and Path(resume_path).exists():
    st.info(f"已加载历史任务: **{resume_name}**")
    input_path = Path(resume_path)
    uploaded_file_name = resume_name
    uploaded_file = None  # 占位，避免未定义
    direct_start = True
else:
    uploaded_file = st.file_uploader(
        "上传音频或视频文件",
        type=["mp3", "wav", "m4a", "aac", "ogg", "flac",
              "mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"],
        help="支持常见音频格式（mp3/wav/m4a 等）和视频格式（mp4/avi/mov 等）",
    )
    direct_start = False

if direct_start or (uploaded_file is not None):
    if direct_start:
        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        st.info(f"已加载: **{uploaded_file_name}** ({file_size_mb:.1f} MB)")
    else:
        # 保存上传文件
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        suffix = Path(uploaded_file.name).suffix
        input_path = upload_dir / f"input_{int(time.time())}{suffix}"
        input_path.write_bytes(uploaded_file.read())
        uploaded_file_name = uploaded_file.name
        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        st.info(f"已上传: **{uploaded_file_name}** ({file_size_mb:.1f} MB)")

    # ─── 任务级临时补充（仅本次任务生效，不写盘）───
    with st.expander("任务级临时补充（可选）", expanded=False):
        st.caption(
            "本任务的临时上下文：会被拼到项目级词表之后，作为本次转录的 initial_prompt。"
            "转录完成后可下载原始版与 _corrected 版。"
        )
        task_background = st.text_input(
            "本次会议 / 录音的背景（可选）",
            value="",
            placeholder="例：和客户张三对齐 Q3 实施计划",
        )
        task_people_text = st.text_input(
            "本次参会人（逗号分隔）",
            value="",
            placeholder="例：王五, 赵六",
        )
        task_terms_text = st.text_input(
            "本次关键词（逗号分隔）",
            value="",
            placeholder="例：CRM, OKR, Project Atlas",
        )
        st.session_state["task_context"] = {
            "background": task_background,
            "people": [s.strip() for s in task_people_text.split(",") if s.strip()],
            "terms": [s.strip() for s in task_terms_text.split(",") if s.strip()],
        }

    # ─── 开始处理 ───
    if st.button("开始转录", type="primary", use_container_width=True):
        # 清除历史恢复标记，开始新的转录任务
        st.session_state.resume_audio_path = None
        st.session_state.resume_original_name = None
        # 保存历史记录
        save_history_entry({
            "original_name": uploaded_file_name,
            "audio_path": str(input_path),
            "status": "processing",
            "stage": "started",
            "timestamp": datetime.now().isoformat(),
            "duration": 0,
        })
        try:
            # 重置日志缓冲区
            log_buffer.truncate(0)
            log_buffer.seek(0)

            # 进度区域
            progress_bar = st.progress(0, text="准备中...")
            status_container = st.empty()
            log_expander = st.expander("查看详细日志", expanded=True)
            log_area = log_expander.empty()
            stage_placeholder = st.empty()

            # Step 1: 提取/持久化音频
            suffix = Path(uploaded_file_name).suffix
            is_video = suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]

            if is_video:
                status_container.info("检测到视频文件，正在提取音频...")
                audio_path = extract_audio(str(input_path), output_dir="output")
                status_container.success(f"音频提取完成: {Path(audio_path).name}")
                progress_bar.progress(0.03, text="音频提取完成")
            else:
                # 将音频文件持久化到 output/ 目录，避免 uploads/ 清理后失效
                persist_dir = OUTPUT_DIR / "audio"
                persist_dir.mkdir(exist_ok=True)
                persist_path = persist_dir / f"{Path(uploaded_file_name).stem}{suffix}"
                if not persist_path.exists() or persist_path.stat().st_size != input_path.stat().st_size:
                    import shutil
                    shutil.copy2(str(input_path), str(persist_path))
                audio_path = str(persist_path)

            # 更新历史记录中的 audio_path
            save_history_entry({
                "original_name": uploaded_file_name,
                "audio_path": audio_path,
                "status": "processing",
                "stage": "audio_ready",
                "timestamp": datetime.now().isoformat(),
                "duration": 0,
            })

            # Step 2: 初始化转录器
            transcriber = AudioTranscriber(
                model_size=model_size,
                hf_token=hf_token or None,
            )
            status_container.info(f"模型: {model_size} | 设备: {transcriber.device} | 精度: {transcriber.compute_type}")

            # 进度回调（同时更新进度条、阶段文字和日志区域）
            def on_progress(stage: str | None, progress: float, log_msg: str | None):
                if log_msg:
                    # 刷新日志显示
                    logs = log_buffer.getvalue()
                    log_area.code(logs, language="log")
                if stage and progress >= 0:
                    progress_bar.progress(min(progress, 1.0), text=stage)
                    stage_placeholder.text(stage)

            # Step 3: 转录
            lang_param = language.strip() if language.strip() else None

            # 构造 initial_prompt（项目级词表 + 任务级补充）
            _glossary_now = load_glossary()
            _task_ctx = st.session_state.get("task_context", {}) or {}
            initial_prompt_str = build_initial_prompt(
                glossary=_glossary_now,
                task_background=_task_ctx.get("background", ""),
                task_people=_task_ctx.get("people", []),
                task_terms=_task_ctx.get("terms", []),
            )
            if initial_prompt_str:
                st.caption(
                    f"已构造上下文 prompt（{len(initial_prompt_str)} 字符，"
                    f"hash={hash_prompt(initial_prompt_str)}）"
                )
            else:
                st.caption("未配置上下文（项目级词表为空且任务级无补充）")

            result = transcriber.transcribe(
                audio_path=audio_path,
                language=lang_param,
                enable_diarization=enable_diarization,
                min_speakers=min_speakers if enable_diarization else None,
                max_speakers=max_speakers if enable_diarization else None,
                progress_callback=on_progress,
                force_restart=force_restart,
                initial_prompt=initial_prompt_str,
            )

            # 最终日志
            logs = log_buffer.getvalue()
            log_area.code(logs, language="log")

            progress_bar.progress(1.0, text="转录完成！")
            stage_placeholder.empty()

            status_container.success(
                f"转录完成！检测语言: {result.language} | "
                f"时长: {result.duration_seconds:.0f} 秒 | "
                f"段落数: {len(result.segments)}"
            )

            # 更新历史为完成
            save_history_entry({
                "original_name": uploaded_file_name,
                "audio_path": audio_path,
                "status": "completed",
                "stage": "done",
                "timestamp": datetime.now().isoformat(),
                "duration": result.duration_seconds,
            })

            # ─── 结果展示 ───
            st.divider()
            st.subheader("转录结果")

            # 说话人统计
            if enable_diarization:
                speakers = set(s.speaker for s in result.segments if s.speaker != "UNKNOWN")
                if speakers:
                    speaker_list = sorted(speakers)
                    cols = st.columns(len(speaker_list))
                    for i, spk in enumerate(speaker_list):
                        spk_segments = [s for s in result.segments if s.speaker == spk]
                        total_dur = sum(s.end - s.start for s in spk_segments)
                        cols[i].metric(
                            label=spk,
                            value=f"{total_dur:.0f}s",
                            delta=f"{len(spk_segments)} 段",
                        )

            st.divider()

            # 逐段显示（原始版，默认折叠以便用户聚焦校正版）
            color_map = {}
            color_classes = ["speaker-0", "speaker-1", "speaker-2", "speaker-3", "speaker-4"]

            with st.expander("原始转录（Whisper 输出）", expanded=False):
                for i, seg in enumerate(result.segments):
                    speaker = seg.speaker if seg.speaker != "UNKNOWN" else ""

                    if speaker and speaker not in color_map:
                        color_map[speaker] = color_classes[len(color_map) % len(color_classes)]

                    css_class = color_map.get(speaker, "")

                    start_str = format_time(seg.start)
                    end_str = format_time(seg.end)

                    st.markdown(f"""
                    <div class="speaker-segment {css_class}">
                        <div class="speaker-label">{speaker}</div>
                        <div class="timestamp">{start_str} → {end_str}</div>
                        <div class="segment-text">{seg.text}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ─── Step 4: LLM 二次校对（可选）───
            _runtime = st.session_state.get("llm_runtime", {}) or {}
            _run_llm = bool(_runtime.get("enabled")) and bool(_runtime.get("api_key"))

            corrected_segments: list | None = None
            if _run_llm:
                st.divider()
                st.subheader("LLM 二次校对")
                _llm_status = st.status("正在调用 LLM 校对...", expanded=True)
                try:
                    _llm_cfg = LLMConfig(
                        provider=_runtime.get("provider", "deepseek"),
                        api_key=_runtime.get("api_key", "") or os.environ.get("DEEPSEEK_API_KEY", ""),
                        base_url=_runtime.get("base_url", ""),
                        model=_runtime.get("model", ""),
                        temperature=float(_runtime.get("temperature", 0.2)),
                        max_batch_chars=int(_runtime.get("max_batch_chars", 6000)),
                        enabled=True,
                    )
                    _corrector = LLMCorrector(_llm_cfg)
                    _glossary_now2 = load_glossary()
                    _people_flat = [p.get("name", "") for p in _glossary_now2.people if p.get("name")]
                    _task_ctx2 = st.session_state.get("task_context", {}) or {}
                    _all_people = list(_people_flat) + list(_task_ctx2.get("people", []))
                    _all_terms = list(_glossary_now2.terms) + list(_task_ctx2.get("terms", []))
                    _bg_combined = "。".join(
                        s for s in [_glossary_now2.background, _task_ctx2.get("background", "")]
                        if s and s.strip()
                    )

                    def _llm_progress(stage: str, pct: float) -> None:
                        _llm_status.update(label=stage, state="running")

                    corrected_segments = _corrector.correct(
                        segments=result.segments,
                        background=_bg_combined,
                        people=_all_people,
                        terms=_all_terms,
                        progress_callback=_llm_progress,
                    )
                    _llm_status.update(label="LLM 校对完成", state="complete")
                except Exception as e:
                    _llm_status.update(label=f"LLM 校对失败：{e}", state="error")
                    corrected_segments = None
            elif _runtime.get("enabled") and not _runtime.get("api_key"):
                st.warning("侧边栏启用了 LLM 纠错但未填写 API key，已跳过校对。")

            # ─── 自动保存到 output/ 目录 ───
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            base_name = Path(uploaded_file_name).stem

            output_files = {}
            for label, fname, content in [
                ("纯文本", f"{base_name}.txt", result.full_text_plain),
                ("说话人标注", f"{base_name}_speakers.txt", result.full_text),
                ("SRT 字幕", f"{base_name}.srt", to_srt(result.segments)),
            ]:
                fpath = output_dir / fname
                fpath.write_text(content, encoding="utf-8")
                output_files[label] = str(fpath)

            st.success(f"已自动保存 {len(output_files)} 个原始文件到 output/ 目录：")
            for label, fpath in output_files.items():
                st.code(fpath, language=None)

            # ─── 校正版：可编辑表格 + 落盘 ───
            corrected_files: dict = {}
            corrected_full_text = ""
            corrected_full_text_plain = ""
            if corrected_segments is not None:
                st.divider()
                st.subheader("校正版（可直接编辑）")
                st.caption(
                    "下方表格已用 LLM 二次校对，可在此直接修改。点击「保存校正版」"
                    "后会把改动写盘成 `<base>_corrected.*`，原始文件不动。"
                )

                edit_df = st.data_editor(
                    data=[
                        {
                            "#": i,
                            "speaker": seg.speaker,
                            "start": round(seg.start, 2),
                            "end": round(seg.end, 2),
                            "text": seg.text,
                        }
                        for i, seg in enumerate(corrected_segments)
                    ],
                    key="corrected_editor",
                    use_container_width=True,
                    height=min(560, 80 + 35 * len(corrected_segments)),
                    column_config={
                        "#": st.column_config.NumberColumn("#", disabled=True, width="small"),
                        "speaker": st.column_config.TextColumn("说话人", disabled=True, width="small"),
                        "start": st.column_config.NumberColumn("开始(s)", disabled=True, width="small"),
                        "end": st.column_config.NumberColumn("结束(s)", disabled=True, width="small"),
                        "text": st.column_config.TextColumn("文本（可编辑）", width="large"),
                    },
                    hide_index=True,
                )

                # 把编辑后的 text 同步回 corrected_segments
                for i, row in enumerate(edit_df):
                    if 0 <= i < len(corrected_segments):
                        corrected_segments[i] = SpeakerSegment(
                            speaker=row.get("speaker", corrected_segments[i].speaker),
                            text=row.get("text", corrected_segments[i].text),
                            start=float(row.get("start", corrected_segments[i].start)),
                            end=float(row.get("end", corrected_segments[i].end)),
                        )

                if st.button("保存校正版", type="primary", use_container_width=True):
                    # 重新组装 full_text / full_text_plain
                    c_full_lines: list[str] = []
                    c_plain_lines: list[str] = []
                    for seg in corrected_segments:
                        timestamp = f"[{format_time(seg.start)} → {format_time(seg.end)}]"
                        if seg.speaker and seg.speaker != "UNKNOWN":
                            c_full_lines.append(f"{timestamp} {seg.speaker}: {seg.text}")
                        else:
                            c_full_lines.append(f"{timestamp} {seg.text}")
                        c_plain_lines.append(seg.text)
                    corrected_full_text = "\n\n".join(c_full_lines)
                    corrected_full_text_plain = "".join(c_plain_lines)

                    for label, fname, content in [
                        ("纯文本（校正）", f"{base_name}_corrected.txt", corrected_full_text_plain),
                        ("说话人标注（校正）", f"{base_name}_corrected_speakers.txt", corrected_full_text),
                        ("SRT 字幕（校正）", f"{base_name}_corrected.srt", to_srt(corrected_segments)),
                    ]:
                        fpath = output_dir / fname
                        fpath.write_text(content, encoding="utf-8")
                        corrected_files[label] = str(fpath)

                    st.success(f"已保存 {len(corrected_files)} 个校正版文件：")
                    for label, fpath in corrected_files.items():
                        st.code(fpath, language=None)

            # 将结果文件路径写入历史记录，便于日后直接查看结果
            all_files = {**output_files}
            # 若校正版已保存，把它们也写进 history 的 result_files（用 base_name 区分标签）
            for label, fpath in corrected_files.items():
                all_files[label] = fpath
            save_history_entry({
                "original_name": uploaded_file_name,
                "audio_path": audio_path,
                "status": "completed",
                "stage": "done",
                "timestamp": datetime.now().isoformat(),
                "duration": result.duration_seconds,
                "result_files": all_files,
            })

            # ─── 导出区域 ───
            st.divider()
            st.subheader("手动下载")
            st.caption("文件已自动保存到 output/ 目录，也可在此手动下载：")

            col1, col2, col3 = st.columns(3)

            # 纯文本（无标注，原始）
            col1.download_button(
                label="纯文本（原始）.txt",
                data=result.full_text_plain,
                file_name=f"transcript_plain_{base_name}.txt",
                mime="text/plain",
                use_container_width=True,
            )

            # 带说话人标注（原始）
            col2.download_button(
                label="说话人标注（原始）.txt",
                data=result.full_text,
                file_name=f"transcript_speakers_{base_name}.txt",
                mime="text/plain",
                use_container_width=True,
            )

            # SRT 字幕格式（原始）
            srt_content = to_srt(result.segments)
            col3.download_button(
                label="SRT 字幕（原始）.srt",
                data=srt_content,
                file_name=f"transcript_{base_name}.srt",
                mime="text/plain",
                use_container_width=True,
            )

            # 校正版下载（如果已保存）
            if corrected_files:
                st.divider()
                st.subheader("校正版下载")
                st.caption("与原始版一一对应，可直接下载：")
                # 若用户改完表格还没点保存，用表格内的最新编辑结果现算
                _export_segs = corrected_segments
                if "corrected_editor" in st.session_state:
                    try:
                        _edit_data = st.session_state["corrected_editor"].get("edited_rows", {}) \
                            if hasattr(st.session_state["corrected_editor"], "get") else {}
                    except Exception:
                        _edit_data = {}
                else:
                    _edit_data = {}

                c1, c2, c3 = st.columns(3)
                c1.download_button(
                    label="纯文本（校正）.txt",
                    data="".join(s.text for s in _export_segs),
                    file_name=f"transcript_plain_{base_name}_corrected.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
                _export_full_lines: list[str] = []
                for seg in _export_segs:
                    _ts = f"[{format_time(seg.start)} → {format_time(seg.end)}]"
                    if seg.speaker and seg.speaker != "UNKNOWN":
                        _export_full_lines.append(f"{_ts} {seg.speaker}: {seg.text}")
                    else:
                        _export_full_lines.append(f"{_ts} {seg.text}")
                c2.download_button(
                    label="说话人标注（校正）.txt",
                    data="\n\n".join(_export_full_lines),
                    file_name=f"transcript_speakers_{base_name}_corrected.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
                c3.download_button(
                    label="SRT 字幕（校正）.srt",
                    data=to_srt(_export_segs),
                    file_name=f"transcript_{base_name}_corrected.srt",
                    mime="text/plain",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"处理失败: {e}")
            save_history_entry({
                "original_name": uploaded_file_name,
                "audio_path": audio_path if 'audio_path' in dir() else str(input_path),
                "status": "error",
                "stage": str(e)[:100],
                "timestamp": datetime.now().isoformat(),
                "duration": 0,
            })
            import traceback
            st.code(traceback.format_exc())

        finally:
            # 清理上传文件
            if input_path.exists():
                input_path.unlink()


