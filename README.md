# AI 技术文献助手

一个轻量的中英文文献问答工具：导入资料、检索相关段落，再由大语言模型基于原文回答，并展示来源和页码。

使用 Python、Streamlit 和 SQLite 构建，支持本地语义检索与可替换的模型 API。

## 功能

- 导入文本型 PDF、Markdown 和 TXT，自动去重并保留页码。
- 支持 BM25、TF-IDF、本地语义向量及兼容 API 检索。
- 自动模式先使用 BM25，结果置信度不足时结合语义检索，以 RRF 融合排名；语义模型不可用时退回 BM25。
- 支持 DeepSeek 等模型生成回答，附带可核对的原文引用。
- 文档与向量缓存在本地保存，提供检索评测工具。

## 快速开始

需要 Python 3.10+。在项目目录执行以下 PowerShell 命令：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe -m streamlit run app.py
```

打开页面，在“资料库”中导入文件或内置示例，然后到“提问与证据”输入问题。无需激活虚拟环境；不配置 API Key 也可使用 BM25 检索。

## 回答模型配置

编辑 `.env`，以下为项目自带的 DeepSeek 配置：

```dotenv
TECHLIT_LLM_API_KEY=你的DeepSeek-Key
TECHLIT_LLM_BASE_URL=https://api.deepseek.com
TECHLIT_LLM_MODEL=deepseek-flash
TECHLIT_LLM_API_MODE=responses
```

修改后重启应用。其他兼容服务可替换地址、模型和 Key；API 模式支持 `responses` 与 `chat_completions`，按服务商接口选择。

## 本地语义检索

使用 `intfloat/multilingual-e5-small`，支持中英文及跨语言检索，输出 384 维向量，可在 CPU 上运行，无需向量 API Key。

安装依赖并下载模型：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[local]"
.\.venv\Scripts\python.exe -m techlit_assistant.local_embedding
```

在 `.env` 中确认：

```dotenv
TECHLIT_EMBEDDING_PROVIDER=local
TECHLIT_LOCAL_MODEL_PATH=data/models/multilingual-e5-small
```

重启后选择“自动（推荐）”或“语义向量（本地 / API）”。模型权重约 470 MB，首次下载需要联网，后续向量推理离线运行；首次检索加载和建立索引会稍慢。

也可将 `TECHLIT_EMBEDDING_PROVIDER` 设为 `api`，并填写 `TECHLIT_EMBEDDING_API_KEY`、`TECHLIT_EMBEDDING_BASE_URL` 和 `TECHLIT_EMBEDDING_MODEL`，使用独立的向量服务。

## 测试与评测

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m techlit_assistant.evaluation --seed-dir data/samples --questions eval/questions.jsonl --mode auto --k 5
```

评测输出 Recall@k、MRR、无答案无命中率和检索延迟，默认使用独立的 `data/evaluation.db`，不修改应用资料库。评测方法见 [docs/evaluation.md](docs/evaluation.md)。

## 使用限制

- 暂不支持扫描件 OCR；检索结果和模型回答仍需核对原文。
- 本地向量分数不等于回答可信度，也不保证无答案问题返回空结果。
- 使用云端回答时，问题与检索片段会发送给所配置的模型服务；使用向量 API 时，问题与文本块也会发送给对应服务。
