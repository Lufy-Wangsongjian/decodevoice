# DecodeVoice — 语音转文字 + 说话人分段

从音频/视频文件中提取语音内容，自动识别不同说话人并按说话人分段输出文字。

## 功能

- 支持常见音频格式（mp3, wav, m4a, aac, ogg, flac）
- 支持常见视频格式（mp4, avi, mov, mkv, webm, flv, wmv）
- 自动从视频中提取音频轨道
- Whisper 高精度语音识别（支持中文、英文等多语言）
- 说话人分离（Speaker Diarization），区分不同说话人
- 导出格式：纯文本、带说话人标注文本、SRT 字幕
- **上下文纠错（v1.1+）**：通过人名/术语词表 + 可选 LLM 二次校对，显著降低专有名词识别错误

## 前置依赖

### 1. Python 3.10+

```bash
python --version
```

### 2. FFmpeg（音频提取必需）

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 3. HuggingFace Token（说话人分离需要）

1. 注册 [huggingface.co](https://huggingface.co)
2. 创建 token: https://huggingface.co/settings/tokens
3. 接受 pyannote 模型协议:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

## 安装

```bash
cd decodevoice

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

> 注意: `whisperx` 第一次运行时会自动下载模型（约 1.5GB ~ 3GB），请保持网络通畅。

## 运行

```bash
streamlit run app.py
```

浏览器访问 http://localhost:8501

## 使用步骤

1. 在侧边栏选择模型大小（推荐 `large-v3`）
2. 输入目标语言（中文填 `zh`）
3. 填入 HuggingFace Token（说话人分离需要）
4. （可选）维护项目级人名 / 术语词表
5. （可选）配置 LLM 二次校对（混元 / DeepSeek / 自定义 OpenAI 兼容接口）
6. 上传音频或视频文件
7. （可选）在"任务级临时补充"里填本次任务专属的参会人 / 关键词
8. 点击"开始转录"
9. 查看结果，可选择导出格式下载

## 上下文纠错（Context Correction）

为了减少 ASR 在人名、品牌、术语上的同音字误识别（比如把"彦秋"听成"燕求"），DecodeVoice 提供三层纠错能力，可单独或叠加使用。

### 第一层：Whisper `initial_prompt` 注入（默认启用）

把项目级人名词表 + 任务级临时补充拼成一段 ≤ 320 字符的 prompt，作为 Whisper 解码器的先验提示。对人名 / 品牌 / 术语的纠错效果最稳，零额外依赖。

配置位置：侧边栏 → **项目级默认词表**

```json
// output/context/glossary.json
{
  "background": "产品周会，讨论 Q3 路线图",
  "people": [
    {"name": "张三", "alias": ["老张", "张工"]},
    {"name": "李四", "alias": []}
  ],
  "terms": ["DecodeVoice", "Project Atlas", "PyAnnote"]
}
```

### 第二层：LLM 二次校对（可选，需配置 API key）

转录完成后，把全文 + 上下文发给云端 LLM（默认 DeepSeek，亦支持腾讯混元或任意 OpenAI 兼容接口），按段改写文本，**只改 `text` 字段，时间戳和说话人标签保持不变**。

配置位置：侧边栏 → **上下文纠错（LLM 二次校对）**

| 字段 | 说明 |
|------|------|
| Provider | `hunyuan`（腾讯混元）/ `deepseek` / `custom`（自定义 OpenAI 兼容） |
| API Key | LLM 服务的 API key，落盘到 `output/context/llm_config.json` |
| Model | 模型名；留空时使用 provider 预设的默认模型 |
| Base URL | 仅 `custom` 模式需要，例如 `https://your-llm-gateway/v1` |
| Temperature | 默认 0.2，越低越保守 |
| 单批最大字符数 | 单次 LLM 请求承载的文本量；超出自动分批，默认 6000 |

拿到 API key 后：

1. 勾选「启用 LLM 纠错」
2. 填 Provider / API Key / Model
3. 点击「保存 LLM 配置」

> 注意：API key 会以明文保存到 `output/context/llm_config.json`，建议使用专用 key 或具备额度上限的子账号。

### 第三层：人工编辑（兜底）

转录 + LLM 校对后，结果页提供 `st.data_editor` 可编辑表格。修改后点 **「保存校正版」** 即可把改动落盘成 `<base_name>_corrected.txt` / `<base_name>_corrected_speakers.txt` / `<base_name>_corrected.srt`。原始文件保留不动。

### 输出文件

- **原始版**：`<base_name>.txt` / `<base_name>_speakers.txt` / `<base_name>.srt`
- **校正版**：`<base_name>_corrected.txt` / `<base_name>_corrected_speakers.txt` / `<base_name>_corrected.srt`

两份都自动写入 `output/` 目录，并写进 `output/history.json` 的 `result_files` 字段，侧边栏"历史记录"可一键查看。

### 断点与上下文哈希

`output/checkpoints/` 下的断点会带 `prompt_hash` 字段。一旦你修改了项目级词表或任务级补充（导致 prompt 哈希变化），旧的转录断点会**自动失效**，强制重新转录以确保新 prompt 生效。

## 模型选择建议

| 模型 | 大小 | 速度 | 精度 | 适用场景 |
|------|------|------|------|----------|
| tiny | ~75MB | 极快 | 一般 | 快速预览 |
| base | ~145MB | 快 | 中等 | 英文为主 |
| small | ~488MB | 较快 | 较好 | 日常使用 |
| medium | ~1.5GB | 中等 | 良好 | 平衡选择 |
| large-v3 | ~3GB | 慢 | 最佳 | 专业转录 |

## 项目结构

```
decodevoice/
├── app.py                 # Streamlit 主应用
├── core/
│   ├── extractor.py       # 音频提取（FFmpeg）
│   ├── diarizer.py        # 转录 + 说话人分离（whisperx）
│   ├── context.py         # 项目级人名/术语词表 + initial_prompt 拼装
│   └── corrector.py       # 云端 LLM（混元/DeepSeek）二次校对客户端
├── requirements.txt
├── README.md
├── uploads/               # 上传文件临时目录
└── output/                # 输出目录
    ├── context/
    │   ├── glossary.json  # 项目级人名/术语词表
    │   └── llm_config.json# LLM 纠错配置
    ├── checkpoints/       # 转录断点（带 prompt_hash）
    ├── audio/             # 持久化的音频文件
    └── history.json       # 历史记录
```
