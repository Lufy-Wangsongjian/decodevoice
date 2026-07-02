---
name: decodevoice-context-correction
overview: 为 DecodeVoice 新增『上下文纠错』特性：上传音频时支持填写项目级默认词表 + 任务级临时补充，流水线在 ASR 阶段用 initial_prompt 注入人名/术语先验；转录完成后用云端 LLM（混元/DeepSeek）对全文做按段改写；结果页提供可编辑表格作为人工兜底；最终同时输出原始版与 _corrected 版并写回历史。
---

