"""
音频提取模块：从视频文件中提取音频轨道。
"""

import subprocess
import tempfile
from pathlib import Path


def extract_audio(video_path: str, output_dir: str | None = None) -> str:
    """
    从视频文件中提取音频，输出为 WAV 格式（16kHz, 单声道）。

    参数:
        video_path: 输入视频/音频文件路径
        output_dir: 输出目录，默认临时目录

    返回:
        提取后的音频文件路径（.wav）
    """
    input_path = Path(video_path)

    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {video_path}")

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{input_path.stem}_audio.wav")
    else:
        suffix = "_audio.wav"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        output_path = tmp.name
        tmp.close()

    # 提取音频并转换为 16kHz 单声道 WAV
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vn",                      # 丢弃视频流
        "-acodec", "pcm_s16le",    # PCM 16bit
        "-ar", "16000",             # 采样率 16kHz
        "-ac", "1",                 # 单声道
        "-y",                       # 覆盖已有文件
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 提取音频失败:\n{result.stderr}")

    if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
        raise RuntimeError("提取的音频文件为空")

    return output_path


def get_audio_duration(audio_path: str) -> float:
    """获取音频时长（秒）。"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe 获取时长失败:\n{result.stderr}")
    return float(result.stdout.strip())
