from __future__ import annotations

from pathlib import Path

import streamlit as st
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
)

from techlit_assistant.answer import CompatibleAnswerer
from techlit_assistant.config import Settings
from techlit_assistant.database import Repository
from techlit_assistant.ingest import IngestionError, ingest_bytes
from techlit_assistant.retrieval import (
    AutoRetriever,
    BM25Retriever,
    EmbeddingRetriever,
    RetrievalTrace,
    TfidfRetriever,
)
from techlit_assistant.retriever_factory import build_semantic_retriever

PROJECT_ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="AI 技术文献助手", page_icon="📚", layout="wide")
try:
    SETTINGS = Settings.from_env(PROJECT_ROOT)
except ValueError as exc:
    st.error(f"配置错误：{exc}")
    st.info("请修改项目目录中的 .env，然后重新启动应用。")
    st.stop()


@st.cache_resource
def get_repository(database_path: str) -> Repository:
    return Repository(database_path)


def build_embedding_retriever(repository: Repository) -> EmbeddingRetriever | None:
    return build_semantic_retriever(repository, SETTINGS)


def build_retriever(repository: Repository, mode: str):
    if mode == "自动（推荐）":
        return AutoRetriever(
            BM25Retriever(repository),
            build_embedding_retriever(repository),
        )
    if mode == "BM25（离线）":
        return BM25Retriever(repository)
    if mode == "TF-IDF（离线）":
        return TfidfRetriever(repository)
    embedding_retriever = build_embedding_retriever(repository)
    if embedding_retriever is None:
        raise ValueError("请配置本地 E5 模型，或配置向量 API 的模型及凭证")
    return embedding_retriever


def render_source(hit, index: int) -> None:
    with st.expander(
        f"[S{index}] {hit.chunk.document_title} · p.{hit.chunk.page} · score={hit.score:.3f}"
    ):
        st.caption(hit.chunk.filename)
        st.write(hit.chunk.text)


def friendly_request_error(exc: Exception) -> str:
    if isinstance(exc, APIConnectionError):
        return "无法连接模型服务。请检查 Base URL 是否完整、网络或代理是否可用，然后重新启动应用。"
    if isinstance(exc, AuthenticationError):
        return "模型服务认证失败。请检查 API Key 是否属于当前 Base URL。"
    if isinstance(exc, NotFoundError):
        return "接口或模型不存在。请检查 Base URL、API 模式和模型名。"
    if isinstance(exc, RateLimitError):
        return "请求受到限流，或账户额度不足。请稍后重试并检查服务商余额。"
    if isinstance(exc, BadRequestError):
        return f"模型服务拒绝了请求：{str(exc)[:300]}"
    if isinstance(exc, APIStatusError):
        return f"模型服务返回 HTTP {exc.status_code}：{str(exc)[:300]}"
    return f"本次请求失败：{exc}"


def render_retrieval_trace(trace: RetrievalTrace) -> None:
    strategy_labels = {
        "bm25": "BM25（高置信度，未运行语义模型）",
        "rrf": "BM25 + 语义向量（RRF 融合）",
        "bm25_fallback": "BM25（语义向量不可用时自动降级）",
    }
    reason_labels = {
        "high_coverage": "词法覆盖充分",
        "high_oov": "查询中未登录词较多",
        "low_coverage": "词法覆盖不足",
        "score_tie": "头部结果区分度不足",
        "no_lexical_hit": "没有词法命中",
    }
    base_reason = trace.reason.split(":")[-1]
    st.caption(
        f"实际策略：{strategy_labels.get(trace.strategy, trace.strategy)}；"
        f"原因：{reason_labels.get(base_reason, base_reason)}；"
        f"覆盖率={trace.lexical.idf_coverage:.2f}，"
        f"头部差距={trace.lexical.relative_gap:.2f}，"
        f"OOV={trace.lexical.oov_ratio:.2f}"
    )
    if trace.error:
        st.warning(f"语义向量调用失败，已自动退回 BM25：{trace.error}")


repository = get_repository(str(SETTINGS.database_path))

st.title("📚 AI 技术文献助手")
st.caption("小而完整的 RAG：可追溯证据、离线基线、可复现评测")

with st.sidebar:
    st.header("检索设置")
    retrieval_mode = st.radio(
        "检索器",
        [
            "自动（推荐）",
            "BM25（离线）",
            "TF-IDF（离线）",
            "语义向量（本地 / API）",
        ],
    )
    top_k = st.slider("返回证据数", 1, 10, 5)
    if SETTINGS.llm.configured:
        st.success("回答模型已配置")
    elif SETTINGS.llm.model:
        st.warning("回答模型已选择，但缺少 API Key。请填写 .env 后重启。")
    else:
        st.info("回答模型未配置；证据检索仍可使用。请在 .env 中填写模型和 API Key。")
    if SETTINGS.llm.model:
        st.caption(
            f"模型：{SETTINGS.llm.model}\n\n"
            f"端点：{SETTINGS.llm.display_name}\n\n"
            f"模式：{SETTINGS.llm.api_mode}"
        )
    if SETTINGS.embedding_provider == "local":
        st.caption("向量：multilingual-e5-small\n\n运行：本地 CPU · 中英文 · 384 维")
        if not (SETTINGS.local_model_path / "model.safetensors").is_file():
            st.warning(
                "本地模型尚未下载，自动模式暂退回 BM25。请按 README 的“本地语义检索”完成安装和下载。"
            )
        st.caption("首次语义检索需要加载模型并建立向量索引，可能较慢；后续复用缓存。")
    elif SETTINGS.embedding.configured:
        st.caption(f"向量：{SETTINGS.embedding.model}\n\n端点：{SETTINGS.embedding.display_name}")
    else:
        st.caption("向量模型未配置；自动模式使用离线 BM25。")
    st.caption(f"本地数据库：{SETTINGS.database_path.name}")

library_tab, ask_tab, about_tab = st.tabs(["资料库", "提问与证据", "评测说明"])

with library_tab:
    left, right = st.columns([2, 1])
    with left:
        uploaded_files = st.file_uploader(
            "上传 PDF、Markdown 或 TXT",
            type=["pdf", "md", "txt"],
            accept_multiple_files=True,
        )
        if st.button("导入上传文件", type="primary", disabled=not uploaded_files):
            for uploaded in uploaded_files:
                try:
                    result = ingest_bytes(repository, uploaded.name, uploaded.getvalue())
                    if result.created:
                        st.success(f"{uploaded.name}：新增 {result.chunks_created} 个文本块")
                    else:
                        st.info(f"{uploaded.name}：内容已存在，已跳过")
                except IngestionError as exc:
                    st.error(f"{uploaded.name}：{exc}")

        if st.button("导入内置示例"):
            for path in sorted((PROJECT_ROOT / "data" / "samples").glob("*.md")):
                result = ingest_bytes(repository, path.name, path.read_bytes())
                if result.created:
                    st.success(f"已导入 {path.name}")
            st.rerun()

    with right:
        documents = repository.list_documents()
        st.metric("文档数", len(documents))
        st.metric("文本块数", len(repository.list_chunks()))
        if documents:
            for document in documents:
                st.write(f"• {document.title}（{document.page_count} 页）")
        else:
            st.warning("资料库为空。可先导入内置示例。")
        if documents and st.button("清空本地资料库"):
            repository.clear()
            st.rerun()

with ask_tab:
    with st.form("question_form"):
        query = st.text_area(
            "问题",
            placeholder="例如：RAG 的主要检索评测指标是什么？",
            height=90,
        )
        generate_answer = st.checkbox(
            "让模型基于证据生成回答",
            value=SETTINGS.llm.configured,
            disabled=not SETTINGS.llm.configured,
        )
        submitted = st.form_submit_button("检索并回答", type="primary")

    if submitted:
        if not query.strip():
            st.warning("请输入问题。")
        elif not repository.list_documents():
            st.warning("资料库为空，请先导入资料。")
        else:
            try:
                with st.spinner("正在检索证据…"):
                    retriever = build_retriever(repository, retrieval_mode)
                    hits = retriever.search(query, top_k=top_k)
                trace = getattr(retriever, "last_trace", None)
                if trace:
                    render_retrieval_trace(trace)
                if not hits:
                    st.warning("现有资料中没有找到达到阈值的证据。")
                else:
                    if generate_answer:
                        with st.spinner("正在基于证据生成回答…"):
                            answer = CompatibleAnswerer(
                                SETTINGS.llm.api_key,
                                SETTINGS.llm.model,
                                SETTINGS.llm.base_url,
                                SETTINGS.llm.api_mode,
                            ).answer(query, hits)
                        st.subheader("回答")
                        st.markdown(answer.text)
                    else:
                        st.info("当前为检索模式；下方是最相关的原文证据。")

                    st.subheader("证据")
                    for index, hit in enumerate(hits, start=1):
                        render_source(hit, index)
            # This is the interactive boundary: API, parsing, and storage errors should
            # become a recoverable user message instead of terminating the Streamlit run.
            except Exception as exc:  # noqa: BLE001
                st.error(friendly_request_error(exc))


with about_tab:
    st.markdown(
        """
        ### MVP 验收重点

        - Recall@5 与 MRR 衡量检索；引用支持率与正确拒答率衡量回答。
        - 自动模式先运行 BM25；置信度不足时运行本地或 API embedding，并用 RRF 融合结果。
        - embedding 未配置或调用失败时，自动模式会退回离线 BM25。
        - 内置样例只做冒烟测试，作品集结果应使用 3–5 篇真实公开论文和 20–30 个标注问题。
        - 先稳定端到端链路，再根据失败案例决定是否做混合检索、OCR 或查询改写。

        运行：`.\\.venv\\Scripts\\python.exe -m techlit_assistant.evaluation --database data/evaluation.db --reset --seed-dir data/samples --questions eval/questions.jsonl --mode bm25 --k 5`
        """
    )
