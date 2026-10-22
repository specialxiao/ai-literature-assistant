# AI 技术文献助手

一个面向 AI 工程求职的轻量 RAG 项目。它可以导入 PDF、Markdown 和 TXT，检索带页码的证据，并让所配置的模型基于证据回答问题。

项目优先保证四件事：**能运行、能追溯、能评测、能替换模型供应商**。

## 1. 能做什么

- 上传并解析文本型 PDF、Markdown、TXT。
- 按页切分文本，保留论文标题、文件名和页码。
- 使用 SHA-256 去重，相同文件不会重复入库。
- 使用离线 BM25 或 TF-IDF 完成中英文检索，无需模型或 API Key。
- 使用本地 `multilingual-e5-small` 完成中英文及跨语言语义检索，无需向量 API Key 或显卡。
- 接入支持 `/embeddings` 的兼容 API，完成语义向量检索。
- 自动模式先判断 BM25 置信度，必要时调用 embedding，并使用 RRF 融合排名。
- 通过 Responses 或 Chat Completions API 生成带 `[S1]` 来源标记的回答。
- 将不同端点的向量缓存隔离，避免同名模型跨供应商误用缓存。
- 使用固定问题集计算 Recall@k、MRR、无答案拒检率和检索延迟。

## 2. 五分钟启动

需要 Python 3.10+。在项目目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe -m streamlit run app.py
```

以上命令不需要执行 `Activate.ps1`，因此不会受到 PowerShell 脚本执行策略影响。

打开页面后：

1. 在“资料库”中点击“导入内置示例”。
2. 保持“自动（推荐）”；未安装或下载本地模型时会退回 BM25。启用本地语义检索见 **4.2**。
3. 在“提问与证据”中输入：`RAG 的主要检索评测指标是什么？`
4. 查看命中的证据、文件名和页码。

此流程完全离线，不要求 API Key。

## 3. 选择运行方式

| 方案 | Base URL | API Key | LLM API 模式 | Embedding dimensions |
|---|---|---|---|---|
| 完全离线 | 不需要 | 不需要 | 不生成回答 | 不需要 |
| DeepSeek 官方 + 本地 E5（推荐） | `https://api.deepseek.com` | 仅回答需要 | `responses` | 本地固定 384 维，无需配置 |
| OpenAI 官方 | 留空 | 需要 | `responses` | 可按模型文档设置 |
| 兼容云服务 | 使用服务商提供的地址 | 通常需要 | 通常为 `chat_completions` | 未明确支持时留空 |
| 本地兼容服务 | 本地 `/v1` 地址 | 可留空 | 通常为 `chat_completions` | 未明确支持时留空 |

“OpenAI-compatible”不代表服务一定同时支持 Responses、Chat Completions、embeddings 和 `dimensions` 参数。请以所选服务商文档为准；模型名也必须真实存在于对应端点。

## 4. 配置模型服务

回答模型和向量模型是两套独立配置。你可以用云端模型回答，同时使用另一个服务生成 embedding；也可以只配置其中一个。

### 4.1 DeepSeek 官方（当前项目已预设）

首次使用请先创建本地配置：`Copy-Item configs/deepseek.env.example .env`，再填写自己的 DeepSeek Key。如果已有 `.env`，不要覆盖；只修改需要的字段。仓库中的示例配置不包含真实密钥：

```dotenv
TECHLIT_LLM_API_KEY=在这里填写你的DeepSeek-Key
TECHLIT_LLM_BASE_URL=https://api.deepseek.com
TECHLIT_LLM_MODEL=deepseek-flash
TECHLIT_LLM_API_MODE=responses
```

然后停止并重新启动 Streamlit。不要把 API Key 发到聊天、截图或提交到 Git；`.env` 已在 `.gitignore` 中。

DeepSeek 负责回答，本地 E5 负责语义向量，两者独立；本地模型尚未就绪时，自动模式仍可使用 BM25。不要将 DeepSeek 回答端点当成 embedding 端点。参考：[DeepSeek Python 示例](https://api-docs.deepseek.com/api_samples/chat_python/)、[DeepSeek Responses API](https://api-docs.deepseek.com/guides/responses_api/)。

### 4.2 本地中英文语义模型（推荐）

已选择 [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)：支持中英文及跨语言检索，输出 384 维向量，在 CPU 上运行，不需要部署额外服务或填写向量 Key。

在项目目录依次执行一次：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,local]"
.\.venv\Scripts\python.exe -m techlit_assistant.local_embedding
```

第一次需要联网安装依赖并下载模型，权重约 470 MB，依赖和缓存还会额外占用磁盘；请预留数 GB 空间。下载工具只获取所需文件，固定模型版本，并校验各文件 SHA-256；使用 HTTPS 下载，不依赖代理可能不支持的 HEAD 元数据请求。模型保存在项目的 `data/models/multilingual-e5-small/`，已经加入 Git 忽略规则。下载成功后，向量推理只读取本地文件，不会在提问时自动联网下载。

示例配置已设置本地 E5；从旧版本升级时在 `.env` 补充以下配置即可，不要覆盖原来的 DeepSeek Key：

```dotenv
TECHLIT_EMBEDDING_PROVIDER=local
TECHLIT_LOCAL_MODEL_PATH=data/models/multilingual-e5-small
TECHLIT_LOCAL_MIN_SCORE=0
```

然后重启应用：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

使用步骤：

1. 在“资料库”导入文档。建议加入中英文资料后再测试。
2. 选择“自动（推荐）”，或“语义向量（本地 / API）”以强制测试语义检索。
3. 尝试用中文询问英文文档中的概念，或用英文询问中文资料。跨语言能力不等于每个问题都能命中，应检查原文证据。
4. 首次语义请求需要加载模型并为已有文本建立向量索引，耗时较长；之后复用进程内模型及 SQLite 文档向量缓存。新文档只补算新增块。

实现细节与边界：

- 自动添加 E5 要求的 `query: ` / `passage: ` 前缀，执行带注意力掩码的平均池化和向量归一化。
- 模型单次输入最多 512 tokens。长文本块按实际 tokenizer 切成重叠 64 tokens 的窗口，不跨越原有页码；各窗口向量取平均后再次归一化，仍缓存为一个块向量。因此不会直接丢弃尾部，但短小关键信息可能被长块稀释；此版本不做窗口级精排。
- E5 余弦分数不是相关概率，不能套用其他 embedding 的阈值。默认 `TECHLIT_LOCAL_MIN_SCORE=0` 侧重排序，**不保证无答案问题返回空结果**；拒答仍需回答模型判断证据是否充分。若要按分数拒检，应先用独立标注集调阈值。
- 自动模式高置信度时只运行 BM25，不会加载本地模型。缺少依赖、模型文件或推理失败时回退 BM25，并显示原因；强制语义模式则显示错误。不自动切换到收费向量 API。
- 支持中英文不代表所有语言/学科表现相同，效果需要在你的论文集上评测。

如果下载受网络限制，可在能访问 Hugging Face 的电脑运行上述下载命令，将整个模型目录复制过来，并将 `TECHLIT_LOCAL_MODEL_PATH` 指向该目录。不需要关闭 HTTPS 证书校验。

### 4.3 其他兼容云服务

把服务商文档中的 Base URL、Key 和模型名填入 `.env`：

```dotenv
TECHLIT_LLM_API_KEY=your-key
TECHLIT_LLM_BASE_URL=https://your-provider.example/v1
TECHLIT_LLM_MODEL=your-chat-model
TECHLIT_LLM_API_MODE=chat_completions

TECHLIT_EMBEDDING_PROVIDER=api
TECHLIT_EMBEDDING_API_KEY=your-key
TECHLIT_EMBEDDING_BASE_URL=https://your-provider.example/v1
TECHLIT_EMBEDDING_MODEL=your-embedding-model
TECHLIT_EMBEDDING_DIMENSIONS=
```

如果回答和 embedding 来自不同供应商，分别填写两组 Base URL 和 API Key。

### 4.4 本地兼容服务

先启动本地模型服务，再配置它暴露的 OpenAI-compatible 地址：

```dotenv
TECHLIT_LLM_API_KEY=
TECHLIT_LLM_BASE_URL=http://127.0.0.1:PORT/v1
TECHLIT_LLM_MODEL=your-local-chat-model
TECHLIT_LLM_API_MODE=chat_completions
```

本地服务可以不需要真实 Key；程序会为底层客户端使用一个内部占位值。`PORT`、路径和模型名必须替换为本地服务实际配置。

### 4.5 OpenAI 官方

使用 OpenAI 官方端点时 Base URL 留空：

```dotenv
TECHLIT_LLM_API_KEY=your-openai-key
TECHLIT_LLM_BASE_URL=
TECHLIT_LLM_MODEL=gpt-5.4-mini
TECHLIT_LLM_API_MODE=responses

TECHLIT_EMBEDDING_PROVIDER=api
TECHLIT_EMBEDDING_API_KEY=your-openai-key
TECHLIT_EMBEDDING_BASE_URL=
TECHLIT_EMBEDDING_MODEL=text-embedding-3-small
TECHLIT_EMBEDDING_DIMENSIONS=512
```

OpenAI 官方文档：[Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create)、[Chat Completions API](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create)、[Embeddings API](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create)。

### 4.6 旧配置兼容

上一版的变量仍然有效：

| 旧变量 | 新变量 |
|---|---|
| `OPENAI_API_KEY` | `TECHLIT_LLM_API_KEY` / `TECHLIT_EMBEDDING_API_KEY` |
| `OPENAI_BASE_URL` | 两组 `*_BASE_URL` |
| `OPENAI_CHAT_MODEL` | `TECHLIT_LLM_MODEL` |
| `OPENAI_EMBEDDING_MODEL` | `TECHLIT_EMBEDDING_MODEL` |
| `OPENAI_EMBEDDING_DIMENSIONS` | `TECHLIT_EMBEDDING_DIMENSIONS` |

新的 `TECHLIT_*` 字段优先于旧版 `OPENAI_*`；即使 `.env` 中的新字段为空，也不会误用同类旧配置，因此系统里的代理地址或旧 Key 不会污染当前项目。命令行显式设置的 `TECHLIT_*` 会覆盖 `.env`，便于运行评测实验。建议新项目只使用 `TECHLIT_*`。

## 5. 配置字段说明

| 字段 | 必需 | 说明 |
|---|---|---|
| `TECHLIT_LLM_API_KEY` | 云服务通常需要 | 回答模型凭证；本地服务可空 |
| `TECHLIT_LLM_BASE_URL` | 第三方/本地需要 | 通常包含 `/v1`，以服务商文档为准 |
| `TECHLIT_LLM_MODEL` | 生成回答时需要 | 回答模型名 |
| `TECHLIT_LLM_API_MODE` | 生成回答时需要 | `responses` 或 `chat_completions` |
| `TECHLIT_EMBEDDING_PROVIDER` | 推荐显式设置 | `local` 使用本地 E5；`api` 使用兼容服务；旧配置未设置时仍为 `api` |
| `TECHLIT_LOCAL_MODEL_PATH` | 本地模式 | 本地模型目录，默认 `data/models/multilingual-e5-small` |
| `TECHLIT_LOCAL_MIN_SCORE` | 可选 | 本地 E5 余弦阈值，默认 `0`，应通过真实问题集调参 |
| `TECHLIT_EMBEDDING_API_KEY` | 云服务通常需要 | 向量服务凭证；未填时可复用 LLM Key |
| `TECHLIT_EMBEDDING_BASE_URL` | 第三方/本地需要 | 未填时可复用 LLM Base URL |
| `TECHLIT_EMBEDDING_MODEL` | 向量 API 模式需要 | 云端 embedding 模型名；本地 E5 不读取此项 |
| `TECHLIT_EMBEDDING_DIMENSIONS` | 可选 | 供应商未明确支持时必须留空 |
| `TECHLIT_DB_PATH` | 可选 | SQLite 路径，默认 `data/app.db` |

## 6. 系统架构

```text
PDF / MD / TXT
      │
      ▼
逐页解析与清洗 ──→ 滑窗切分（不跨页）
      │
      ▼
SQLite：documents / chunks / embeddings
      │
      ├──→ BM25：默认词法检索、置信度诊断
      ├──→ TF-IDF：离线对照基线
      └──→ 本地 E5 / 兼容 Embeddings API：低置信度时运行
                         │
                         ▼
               RRF 融合后的 Top-k 证据
                         │
                         ▼
       Responses 或 Chat Completions 受约束生成
                         │
                         ▼
                回答 + [S1] 映射 + 原文
```

核心模块：

```text
app.py                         Streamlit 界面
src/techlit_assistant/
  config.py                    多供应商配置与旧配置兼容
  database.py                  SQLite 持久化、去重、向量缓存
  ingest.py                    PDF / 文本解析和逐页切分
  retrieval.py                 BM25、自动级联、RRF、TF-IDF 与 embedding
  local_embedding.py           CPU E5、显式下载、token 滑窗与归一化
  retriever_factory.py         界面与评测共用的本地 / API 选择
  answer.py                    双 API 模式、受约束回答、引用校验
  evaluation.py                Recall@k / MRR / 延迟评测
data/samples/                  合成演示资料
eval/questions.jsonl           固定冒烟评测集
tests/                         核心验收测试
docs/                          路线图与人工评测方案
```

### 自动模式如何选择

1. 始终先运行不产生 API 成本的 BM25，并取较大的候选集合。
2. 根据查询词的 IDF 覆盖率、第一与第二名的相对差距、OOV 比例及精确缩写命中判断置信度。
3. 高置信度时直接返回 BM25；默认条件为覆盖率至少 `0.70`、OOV 不高于 `0.40`，并且头部差距至少 `0.12` 或存在精确命中。
4. 低置信度且已配置 embedding 时，运行语义检索，并通过 RRF（`k=60`）融合两个排名；两种分数不会被直接相加。
5. Embedding 未配置或运行失败时自动退回 BM25，并在界面显示实际策略与原因。
6. BM25 词法覆盖极低且绝大多数查询词不在资料中时，其词法结果为空；自动模式仍可依靠跨语言语义检索找到证据。语义检索有结果不代表问题一定可回答。

这些阈值是可解释的初始值，应通过自己的真实论文评测集继续调整。

## 7. 关键工程取舍

- **Streamlit 而不是前后端分离**：两到四周内优先完成稳定演示。
- **SQLite 而不是向量数据库**：作品集规模足够，数据结构透明，部署简单。
- **BM25 作为默认词法路径**：没有模型、网络或 Key 时仍然可以演示完整检索链路。
- **自动级联而不是每次调用向量 API**：BM25 高置信度时直接返回；低置信度时才产生 embedding 调用。
- **RRF 而不是直接比较分数**：BM25 与余弦相似度量纲不同，按排名融合更稳妥。
- **回答与 embedding 解耦**：避免被单一服务商锁定，也方便比较模型组合。
- **端点隔离向量缓存**：缓存键同时包含 Base URL 指纹、模型和维度配置。
- **本地 E5 缓存独立**：同时区分模型版本、目录/文件指纹和前缀/滑窗/池化预处理版本，不与云端向量混用。
- **不跨页切分**：可能轻微牺牲召回，但能提供可信页码。
- **引用白名单校验**：模型只能使用本轮真实检索结果中的 `[S<n>]`。

## 8. 运行评测

评测默认使用独立的 `data/evaluation.db`，不使用应用的 `data/app.db`；程序也会拒绝显式指定应用库。`--reset` 只清空所选评测库。不要把评测结果写入自己的资料库。

分别运行离线基线、本地 E5 和自动模式：

```powershell
.\.venv\Scripts\python.exe -m techlit_assistant.evaluation --reset --seed-dir data/samples --questions eval/questions.jsonl --mode tfidf --k 5
.\.venv\Scripts\python.exe -m techlit_assistant.evaluation --reset --seed-dir data/samples --questions eval/questions.jsonl --mode bm25 --k 5
.\.venv\Scripts\python.exe -m techlit_assistant.evaluation --reset --seed-dir data/samples --questions eval/questions.jsonl --mode local --k 5
.\.venv\Scripts\python.exe -m techlit_assistant.evaluation --reset --seed-dir data/samples --questions eval/questions.jsonl --mode auto --k 5
```

兼容 embedding API：

```powershell
.\.venv\Scripts\python.exe -m techlit_assistant.evaluation --reset --seed-dir data/samples --questions eval/questions.jsonl --mode api --k 5
```

旧命令中的 `--mode openai` 仍可使用，它现在是 `--mode api` 的兼容别名。

评测输出包括：

- `recall_at_k`：前 k 个结果是否覆盖标注证据页。
- `mrr`：第一个正确证据的排名倒数。
- `unanswerable_no_hit_rate`：无答案问题是否没有越过检索阈值。
- 平均与 P95 检索延迟。
- `embedding_call_rate`：触发语义检索的问题比例（保留旧字段名）；本地调用也计入，不等于 API 调用或付费比例。
- `embedding_api_query_rate`：触发向量 API 的问题比例（含失败尝试），不是 HTTP 请求次数；本地 E5 此值为 `0`。

首次运行包括模型冷加载和文档索引时间，重复运行会复用磁盘向量缓存；同一进程内才会复用已加载模型。对比延迟时请分别记录冷启动、索引构建及进程内热查询耗时，模型下载不属于检索延迟。新增测试题应包含中文问英文、英文问中文、长文本尾部证据和无答案题。

内置资料只用于冒烟测试。正式作品集应选择 3–5 篇公开论文，人工标注 20–30 个问题，其中至少 5 个问题无法从资料回答。人工回答评测方法见 [docs/evaluation.md](docs/evaluation.md)。

## 9. 测试与代码检查

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

测试覆盖文档去重、切分、中英文 BM25、混合分词、自动路由、RRF、embedding 失败降级、无证据拒答、引用白名单、两种生成 API 模式、配置优先级和向量缓存隔离。

安装并下载本地模型后，可选运行真实模型测试（中问英、英问中、384 维归一化、缓存复用和长文本尾部）：

```powershell
$env:TECHLIT_RUN_LOCAL_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_local_model_integration.py
Remove-Item Env:TECHLIT_RUN_LOCAL_TESTS
```

这些测试使用临时数据库，不修改应用资料库，也不调用 DeepSeek。

## 10. 常见问题

| 现象 | 优先检查 |
|---|---|
| `404` 或 endpoint not found | Base URL 是否包含正确版本路径；API 模式是否选错 |
| DeepSeek 显示 `Connection error` | 必须使用 `https://api.deepseek.com`，不要使用网页地址或未写协议的地址；修改后重启 |
| DeepSeek 显示认证失败 | `.env` 中的 `TECHLIT_LLM_API_KEY` 是否为 DeepSeek 官方平台签发的 Key |
| 模型不存在 | 模型名是否属于当前 Base URL 对应的服务 |
| `dimensions` 参数报错 | 将 `TECHLIT_EMBEDDING_DIMENSIONS` 留空后重试 |
| 可以回答但不能语义检索 | 服务可能支持 chat 但不支持 `/embeddings`；为 embedding 单独配置端点 |
| 本地服务无法连接 | 服务是否启动、端口是否正确、模型是否已经加载 |
| PowerShell 禁止运行 `Activate.ps1` | 不需要激活；直接运行 `.\.venv\Scripts\python.exe -m ...` |
| PDF 没有文本 | 可能是扫描件；MVP 暂不支持 OCR |
| 本地向量依赖未安装 | 安装 `.[local]`，再运行模型下载命令，见 4.2 |
| 模型尚未下载 / 下载失败 | 检查 Hugging Face 网络访问，或复制完整模型目录；自动模式仍可使用 BM25 |
| 安装 PyTorch 提示找不到某个很长的路径 | Windows 路径长度限制；建议将项目放在 `C:\Projects\ai-literature-assistant` 等短目录再创建环境，或使用下方修复命令 |
| 本地语义检索也返回无关片段 | E5 分数不是相关概率；检查证据，补充无答案评测题后再调本地阈值 |

当前目录较深时，PyTorch 安装的路径长度修复（不改系统策略）：

```powershell
$sitePackages = '\\?\' + (Join-Path (Get-Location) '.venv\Lib\site-packages')
.\.venv\Scripts\python.exe -m pip install --upgrade --no-deps --target $sitePackages "torch>=2.2,<3"
.\.venv\Scripts\python.exe -m pip install -e ".[dev,local]"
```

## 11. 隐私与限制

- 原始文件和 SQLite 数据库存储在本地。
- 使用 API 回答时，问题和检索片段会发送到 `TECHLIT_LLM_BASE_URL` 对应的服务。
- 使用 API 生成 embedding 时，文本块会发送到 `TECHLIT_EMBEDDING_BASE_URL` 对应的服务。
- 本地 E5 不向 API 上传问题或文档，推理只读取本地模型；首次显式下载需要连接 Hugging Face。若启用 DeepSeek 回答，问题和命中的证据仍会发给 DeepSeek，并非整个应用完全离线。
- 使用第三方服务前，应自行确认其数据保留、隐私和合规政策。
- MVP 不包含 OCR、联网论文搜索、知识图谱、多 Agent 和用户权限系统。

### 上传 GitHub 前的隐私检查

仓库只包含程序、测试、文档、空白凭证示例和三个合成演示文档。真实 `.env`、数据库及其日志、用户上传资料、模型权重、向量缓存、虚拟环境和运行临时文件不应提交。

提交新修改前执行：

```powershell
git status --short
.\.venv\Scripts\python.exe scripts/check_publish.py
git diff --cached --stat
```

检查脚本读取实际暂存内容，阻止常见密钥、当前 `.env` 中的凭证、用户主目录路径及不允许的数据文件；不会输出密钥值。这只是辅助检查，不能保证发现所有隐私，新增文档、截图和 Git 历史仍须人工审查。`.gitignore` 不会自动移除已提交过的文件；如果密钥曾经进入公开提交，应立即撤销并更换密钥，而不只是删文件。

## 12. 2–4 周作品集计划

| 周次 | 成果 | 完成标准 |
|---|---|---|
| 第 1 周 | 解析、切分、SQLite、BM25/TF-IDF | 任意文本型 PDF 可返回带页码证据 |
| 第 2 周 | 可替换模型接口、引用校验、UI | 离线与至少一种 API 配置均可稳定演示 |
| 第 3 周 | 真实评测集、参数实验、测试 | README 展示真实 Recall@5、MRR 和失败案例 |
| 第 4 周（可选） | 混合检索、过滤、部署、视频 | 只实现能改善已测指标的功能 |

详细任务见 [docs/roadmap.md](docs/roadmap.md)。

面试时可以重点说明：为什么保留离线基线、为什么将回答与 embedding 解耦、如何避免跨供应商缓存污染，以及如何用固定评测集决定下一项优化。
