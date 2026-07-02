"""
说话人分离模块：基于 whisperx 进行语音转录 + 说话人分离。
"""

import gc
import hashlib
import json
import logging
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import whisperx
    from whisperx.diarize import DiarizationPipeline

# 设置日志（同时输出到控制台和 Streamlit 回调）
logger = logging.getLogger("decodevoice")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)

CHECKPOINT_DIR = Path("output/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SpeakerSegment:
    """单个说话人段落"""
    speaker: str          # 说话人标识，如 SPEAKER_00
    text: str             # 转录文本
    start: float          # 开始时间（秒）
    end: float            # 结束时间（秒）


@dataclass
class TranscriptionResult:
    """完整转录结果"""
    segments: list[SpeakerSegment] = field(default_factory=list)
    language: str = ""
    duration_seconds: float = 0.0
    full_text: str = ""           # 全文（带说话人标注）
    full_text_plain: str = ""     # 纯文本（无标注）


# 模块级模型缓存（避免重复加载，预加载后一直复用）
_model_cache: dict = {}


class AudioTranscriber:
    """
    语音转录 + 说话人分离。

    使用 whisperx 流水线:
     1. Faster-Whisper 转录
     2. VAD + 对齐（获取词级时间戳）
     3. pyannote.audio 说话人分离

    支持预加载模型到缓存，后续转录跳过加载步骤。
    """

    # 常用模型
    AVAILABLE_MODELS = [
        "tiny", "tiny.en",
        "base", "base.en",
        "small", "small.en",
        "medium", "medium.en",
        "large-v2", "large-v3",
    ]

    @staticmethod
    def is_model_cached(model_size: str, device: str, compute_type: str) -> bool:
        """检查模型是否已预加载"""
        key = f"{model_size}_{device}_{compute_type}"
        return key in _model_cache

    @staticmethod
    def preload_model(
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = None,
        progress_callback=None,
    ) -> None:
        """
        预加载 Whisper 模型到缓存，后续转录可直接使用。
        首次需下载模型文件（约 1-3 GB）。
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        key = f"{model_size}_{device}_{compute_type}"

        if key in _model_cache:
            logger.info(f"模型 {model_size} 已在缓存中，无需重新加载")
            return

        logger.info(f"预加载模型 {model_size} (device={device}, compute_type={compute_type})...")
        logger.info(f"首次运行需下载约 1-3 GB 的模型文件，请耐心等待...")
        if progress_callback:
            progress_callback(f"正在下载/加载 {model_size} 模型...", 0.15)

        t0 = time.time()
        model = whisperx.load_model(
            model_size,
            device,
            compute_type=compute_type,
            language=language,
        )
        _model_cache[key] = model
        logger.info(f"模型 {model_size} 预加载完成，耗时 {time.time() - t0:.1f}s")
        logger.info(f"后续转录将跳过模型加载步骤，直接进行语音识别")

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
        hf_token: str | None = None,
    ):
        """
        参数:
            model_size: Whisper 模型大小，可选 tiny/base/small/medium/large-v2/large-v3
            device: 运行设备，"cpu" / "cuda" / "auto"
            compute_type: 精度，"float16" / "int8" / "float32" / "auto"
            hf_token: HuggingFace token，说话人分离需要（去 huggingface.co/settings/tokens 获取）
        """
        self.model_size = model_size
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")

        # 设备检测
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if compute_type == "auto":
            if self.device == "cuda":
                self.compute_type = "float16"
            else:
                self.compute_type = "int8"
        else:
            self.compute_type = compute_type

    def _checkpoint_path(self, audio_path: str, stage: str, language: str | None = None) -> Path:
        """生成断点文件路径（基于音频文件内容哈希 + 关键参数），同一文件无论叫什么名字都能匹配"""
        # 取文件前 1MB 和末尾 1MB 做哈希（兼顾大文件性能和唯一性）
        fsize = Path(audio_path).stat().st_size
        with open(audio_path, "rb") as f:
            head = f.read(min(fsize, 1_000_000))
            if fsize > 2_000_000:
                f.seek(-min(fsize, 1_000_000), 2)
            tail = f.read(min(fsize, 1_000_000))
        content_hash = hashlib.md5(head + tail + str(fsize).encode()).hexdigest()[:12]
        return CHECKPOINT_DIR / f"ckpt_{content_hash}_{self.model_size}_{language or 'auto'}_{stage}.pkl"

    def _save_checkpoint(self, path: Path, data: dict):
        """保存断点"""
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.debug(f"断点已保存: {path.name}")

    def _load_checkpoint(self, path: Path) -> dict | None:
        """加载断点，文件不存在或已过期返回 None"""
        if not path.exists():
            return None

        # 断点有效期 24 小时
        age = time.time() - path.stat().st_mtime
        if age > 86400:
            logger.debug(f"断点已过期: {path.name} (已过 {age/3600:.1f}h)")
            path.unlink(missing_ok=True)
            return None

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            logger.debug(f"断点已加载: {path.name}")
            return data
        except Exception as e:
            logger.warning(f"断点加载失败: {e}")
            path.unlink(missing_ok=True)
            return None

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        enable_diarization: bool = True,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        batch_size: int = 16,
        progress_callback=None,
        force_restart: bool = False,
        initial_prompt: str = "",
    ) -> TranscriptionResult:
        """
        执行完整流水线：转录音频并分离说话人。

        参数:
            audio_path: 音频文件路径
            language: 语言代码（如 "zh", "en"），None 则自动检测
            enable_diarization: 是否启用说话人分离
            min_speakers: 最少说话人数
            max_speakers: 最多说话人数
            batch_size: 批处理大小
            progress_callback: 进度回调 (stage: str, progress: float, log_msg: str | None)
            force_restart: 忽略已有断点，强制重跑
            initial_prompt: Whisper 解码器先验提示（注入人名/术语/背景，用于专名纠错）。
                            为空时表示不注入，行为与旧版本一致。
        """
        audio_path = str(audio_path)
        t_start = time.time()

        # 计算当前 prompt 的哈希，用于断点命中校验
        from .context import hash_prompt
        prompt_hash = hash_prompt(initial_prompt or "")

        def _log(msg: str, level: str = "info"):
            """同时输出到 logger 和进度回调"""
            getattr(logger, level)(msg)
            if progress_callback:
                progress_callback(None, -1, msg)

        def _progress(stage: str, pct: float):
            """更新阶段和进度"""
            logger.info(f"[{pct:.0%}] {stage}")
            if progress_callback:
                progress_callback(stage, pct, None)

        # 初始信息
        audio_file = Path(audio_path)
        audio_size_mb = audio_file.stat().st_size / (1024 * 1024) if audio_file.exists() else 0
        _log(f"===== 开始转录 =====")
        _log(f"文件: {audio_file.name} ({audio_size_mb:.1f} MB)")
        _log(f"模型: {self.model_size} | 设备: {self.device} | 精度: {self.compute_type}")
        _log(f"语言: {language or '自动检测'} | 说话人分离: {'启用' if enable_diarization else '关闭'} | 断点续传: {'关闭(强制重跑)' if force_restart else '启用'}")
        if initial_prompt:
            _log(f"上下文注入: 已启用 (initial_prompt, {len(initial_prompt)} 字符, hash={prompt_hash})")
        else:
            _log(f"上下文注入: 未启用（initial_prompt 为空）")

        # ── Step 1: 加载模型（优先从缓存取）──
        _progress("加载 Whisper 模型...", 0.0)
        cache_key = f"{self.model_size}_{self.device}_{self.compute_type}"

        if cache_key in _model_cache:
            model = _model_cache[cache_key]
            _log(f"使用已缓存的模型 {self.model_size}（跳过加载）", "info")
        else:
            _log(f"正在下载/加载模型 {self.model_size}，首次运行需下载约 1-3 GB，请耐心等待...")
            t1 = time.time()
            model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type,
                language=language,
            )
            _model_cache[cache_key] = model
            _log(f"模型加载完成，耗时 {time.time() - t1:.1f}s")

        # ── Step 2: 转录 ──
        _progress("正在转录语音...", 0.05)

        ckpt_transcribe = self._checkpoint_path(audio_path, "transcribe", language)
        cached_transcribe = None if force_restart else self._load_checkpoint(ckpt_transcribe)

        # 校验断点里的 prompt_hash 是否和当前一致；不一致视为过期
        if cached_transcribe and cached_transcribe.get("prompt_hash") != prompt_hash:
            _log(
                f"转录断点的上下文哈希不匹配（断点={cached_transcribe.get('prompt_hash')}，"
                f"当前={prompt_hash}），将重新转录"
            )
            cached_transcribe = None
            try:
                ckpt_transcribe.unlink(missing_ok=True)
            except Exception:
                pass

        if cached_transcribe:
            transcribe_result = cached_transcribe["transcribe_result"]
            detected_lang = cached_transcribe["detected_lang"]
            audio = np.array(cached_transcribe["audio"], dtype=np.float32)
            audio_dur = len(audio) / 16000.0
            _log(f"从断点恢复转录结果（检测语言: {detected_lang}，段落数: {len(transcribe_result.get('segments', []))}）")
        else:
            _log(f"正在加载音频文件并执行转录（batch_size={batch_size}）...")

            t2 = time.time()
            audio = whisperx.load_audio(audio_path)
            audio_dur = len(audio) / 16000.0
            _log(f"音频加载完成，时长 {audio_dur:.0f}s ({audio_dur/60:.1f} 分钟)")

            transcribe_kwargs = {"batch_size": batch_size}
            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt
            transcribe_result = model.transcribe(audio, **transcribe_kwargs)
            detected_lang = transcribe_result.get("language", language or "unknown")
            seg_count = len(transcribe_result.get("segments", []))
            _log(f"转录完成，检测语言: {detected_lang}，原始段落数: {seg_count}，耗时 {time.time() - t2:.1f}s")

            # 保存断点
            self._save_checkpoint(ckpt_transcribe, {
                "transcribe_result": transcribe_result,
                "detected_lang": detected_lang,
                "audio": audio.tolist(),  # numpy -> list for pickle
                "prompt_hash": prompt_hash,
            })

        _progress("转录完成", 0.30)

        # 不释放缓存的模型，下次复用

        # ── Step 3: 对齐（获取词级时间戳）──
        _progress("正在对齐时间轴...", 0.35)

        ckpt_align = self._checkpoint_path(audio_path, "align", language)
        cached_align = None if force_restart else self._load_checkpoint(ckpt_align)

        # prompt 改变 → 强制走一遍 align 流程（因为 transcribe 段可能重跑了）
        if cached_align and cached_align.get("prompt_hash") != prompt_hash:
            _log("对齐断点的 prompt_hash 不匹配，将重新对齐")
            cached_align = None
            try:
                ckpt_align.unlink(missing_ok=True)
            except Exception:
                pass

        if cached_align and cached_align.get("detected_lang") == detected_lang:
            aligned_result = cached_align["aligned_result"]
            _log(f"从断点恢复对齐结果（段落数: {len(aligned_result.get('segments', []))}）")
        else:
            _log(f"加载对齐模型 (语言: {detected_lang})...")

            t3 = time.time()
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_lang, device=self.device
            )
            aligned_result = whisperx.align(
                transcribe_result["segments"],
                model_a,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
            _log(f"时间轴对齐完成，耗时 {time.time() - t3:.1f}s")

            del model_a
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # 保存断点
            self._save_checkpoint(ckpt_align, {
                "aligned_result": aligned_result,
                "detected_lang": detected_lang,
                "prompt_hash": prompt_hash,
            })

        # ── Step 4: 说话人分离 ──
        ckpt_diarize = self._checkpoint_path(audio_path, "diarize", language)

        if enable_diarization:
            _progress("正在分离说话人...", 0.55)

            cached_diarize = None if force_restart else self._load_checkpoint(ckpt_diarize)
            if cached_diarize and cached_diarize.get("prompt_hash") != prompt_hash:
                _log("说话人分离断点的 prompt_hash 不匹配，将重新执行")
                cached_diarize = None
                try:
                    ckpt_diarize.unlink(missing_ok=True)
                except Exception:
                    pass
            if cached_diarize and cached_diarize.get("detected_lang") == detected_lang:
                aligned_result = cached_diarize["aligned_result"]
                _log(f"从断点恢复说话人分离结果（已标注说话人标签）")
            elif not self.hf_token:
                _log("未提供 HF token，跳过说话人分离", "warning")
                _progress("跳过说话人分离（无 token）", 0.85)
            else:
                try:
                    _log("加载 pyannote 说话人分离模型...")
                    t4 = time.time()
                    diarize_model = DiarizationPipeline(
                        token=self.hf_token,
                        device=self.device,
                    )

                    spk_range = f"min={min_speakers}, max={max_speakers}" if min_speakers or max_speakers else "自动"
                    _log(f"执行说话人分离 ({spk_range})，处理 {audio_dur:.0f}s 音频...")

                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                    # whisperx 3.8+ 返回 DataFrame，从 speaker 列获取唯一说话人
                    unique_speakers = diarize_segments["speaker"].nunique() if "speaker" in diarize_segments.columns else 0
                    _log(f"检测到 {unique_speakers} 个说话人，耗时 {time.time() - t4:.1f}s")

                    # 将说话人标签赋予词级对齐结果
                    aligned_result = whisperx.assign_word_speakers(
                        diarize_segments, aligned_result
                    )

                    # 将 SPEAKER_00/01/02 映射为 A/B/C
                    aligned_result = _remap_speakers(aligned_result)
                    _log("说话人标签已分配到文本段落（已映射为 A/B/C 等标识）")

                    # 保存说话人分离断点
                    self._save_checkpoint(ckpt_diarize, {
                        "aligned_result": aligned_result,
                        "detected_lang": detected_lang,
                        "prompt_hash": prompt_hash,
                    })
                except Exception as e:
                    _log(f"说话人分离失败: {e}", "error")
                    _progress("说话人分离失败，使用无标注结果", 0.85)
        else:
            _progress("跳过说话人分离", 0.85)

        _progress("生成结果中...", 0.90)

        # 安全网：确保说话人标签已映射为 A/B/C（处理旧断点未映射的情况）
        aligned_result = _remap_speakers(aligned_result)

        # ── Step 5: 构建输出 ──
        ckpt_result = self._checkpoint_path(audio_path, "result", language)
        cached_result = None if force_restart else self._load_checkpoint(ckpt_result)
        if cached_result and cached_result.get("prompt_hash") != prompt_hash:
            _log("最终结果断点的 prompt_hash 不匹配，将重新生成")
            cached_result = None
            try:
                ckpt_result.unlink(missing_ok=True)
            except Exception:
                pass

        duration = len(audio) / 16000.0  # 16kHz 采样率

        if cached_result:
            _log("从断点恢复最终结果")
            result = TranscriptionResult(
                language=cached_result["language"],
                duration_seconds=cached_result["duration_seconds"],
            )
            result.segments = [
                SpeakerSegment(**s) for s in cached_result["segments"]
            ]
            result.full_text = cached_result["full_text"]
            result.full_text_plain = cached_result["full_text_plain"]
            speaker_set = set(s.speaker for s in result.segments if s.speaker != "UNKNOWN")
        else:

            result = TranscriptionResult(
                language=detected_lang,
                duration_seconds=duration,
            )

            full_text_lines = []
            full_text_plain_lines = []
            speaker_set: set[str] = set()

            for seg in aligned_result.get("segments", []):
                speaker = seg.get("speaker", "UNKNOWN")
                text = seg.get("text", "").strip()
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)

                if not text:
                    continue

                if speaker != "UNKNOWN":
                    speaker_set.add(speaker)

                seg_obj = SpeakerSegment(
                    speaker=speaker,
                    text=text,
                    start=start,
                    end=end,
                )
                result.segments.append(seg_obj)

                timestamp = f"[{_format_time(start)} → {_format_time(end)}]"

                if speaker and speaker != "UNKNOWN":
                    full_text_lines.append(f"{timestamp} {speaker}: {text}")
                else:
                    full_text_lines.append(f"{timestamp} {text}")

                full_text_plain_lines.append(text)

            result.full_text = "\n\n".join(full_text_lines)
            result.full_text_plain = "".join(full_text_plain_lines)

            # 保存最终结果断点
            self._save_checkpoint(ckpt_result, {
                "language": detected_lang,
                "duration_seconds": duration,
                "segments": [
                    {"speaker": s.speaker, "text": s.text, "start": s.start, "end": s.end}
                    for s in result.segments
                ],
                "full_text": result.full_text,
                "full_text_plain": result.full_text_plain,
                "prompt_hash": prompt_hash,
            })

        total_time = time.time() - t_start
        _log(f"===== 转录完成 =====")
        _log(f"总耗时: {total_time:.1f}s ({total_time/60:.1f} 分钟)")
        _log(f"音频时长: {duration:.0f}s，处理速度: {duration/total_time:.1f}x 实时")
        _log(f"检测语言: {detected_lang}，说话人数: {len(speaker_set) or '无标注'}，输出段落: {len(result.segments)}")
        _log(f"总文字数: {len(result.full_text_plain)}")

        _progress("完成！", 1.0)

        return result


def _remap_speakers(aligned_result: dict) -> dict:
    """
    将 SPEAKER_00, SPEAKER_01, ... 按首次出现顺序映射为 A, B, C, ...
    未知/无标注的保持 UNKNOWN。
    """
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    speaker_map: dict[str, str] = {}
    next_idx = 0

    for seg in aligned_result.get("segments", []):
        spk = seg.get("speaker", "UNKNOWN")
        if spk and spk != "UNKNOWN" and spk not in speaker_map:
            speaker_map[spk] = labels[next_idx % len(labels)]
            next_idx += 1

    if speaker_map:
        # 同时更新 segment 和 word 级别的 speaker
        for seg in aligned_result.get("segments", []):
            spk = seg.get("speaker", "")
            if spk in speaker_map:
                seg["speaker"] = speaker_map[spk]

        if "word_segments" in aligned_result:
            for ws in aligned_result["word_segments"]:
                spk = ws.get("speaker", "")
                if spk in speaker_map:
                    ws["speaker"] = speaker_map[spk]

    return aligned_result


def _format_time(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
