"""Optional CPU-only multilingual encoders. Inference never downloads model files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from threading import Lock
from urllib.request import urlopen

import numpy as np

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
PREPROCESS_VERSION = "joint-prefix-boundaries-mean-windows-v3"
MAX_TOKENS = 512
OVERLAP = 64
_LOAD_LOCK = Lock()
MODEL_FILES = {
    "config.json": "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959",
    "model.safetensors": "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477",
    "tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    "tokenizer_config.json": "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
    "special_tokens_map.json": "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7",
    "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
}
MINILM_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MINILM_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MINILM_FILES = {
    "config.json": "6300193cb75e01cf80c96decef7187dfb33094d97cc1490b7ead6ff134476e4e",
    "model.safetensors": "eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b",
    "tokenizer.json": "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8",
    "tokenizer_config.json": "5036ea374ffedd706e3bef33e2e0d6953cb868ef8a490e76e32ba0faa37a6b9b",
    "special_tokens_map.json": "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d",
    "sentencepiece.bpe.model": MODEL_FILES["sentencepiece.bpe.model"],
    "sentence_bert_config.json": "70f4448f31320443fe3557cacea5abf2dcc4915dda8c80646bec9f3bb0aa5a1f",
}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(directory: Path, model_name: str = "e5") -> None:
    """Use verified official HTTPS GETs, including behind HEAD-incompatible proxies."""
    directory.mkdir(parents=True, exist_ok=True)
    files = MINILM_FILES if model_name == "minilm" else MODEL_FILES
    model_id = MINILM_ID if model_name == "minilm" else MODEL_ID
    revision = MINILM_REVISION if model_name == "minilm" else MODEL_REVISION
    for filename, expected_hash in files.items():
        target = directory / filename
        if target.is_file() and _file_sha256(target) == expected_hash:
            print(f"Verified cached {filename}", flush=True)
            continue
        print(f"Downloading {filename} ...", flush=True)
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
        partial = target.with_suffix(target.suffix + ".part")
        with urlopen(url, timeout=60) as response, partial.open("wb") as handle:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(block)
        if _file_sha256(partial) != expected_hash:
            raise ValueError(f"{filename} 完整性校验失败，请重新运行下载命令")
        partial.replace(target)


def token_windows(tokens: list[int], capacity: int, overlap: int = OVERLAP) -> list[list[int]]:
    if not 0 <= overlap < capacity:
        raise ValueError("Window overlap must be smaller than capacity")
    windows = []
    for start in range(0, max(1, len(tokens)), capacity - overlap):
        windows.append(tokens[start : start + capacity])
        if start + capacity >= len(tokens):
            break
    return windows


@lru_cache(maxsize=2)
def _load_cpu_model(model_path: str):
    # The outer lock serializes lru_cache misses across concurrent Streamlit sessions.
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            '本地向量依赖未安装，请运行：python -m pip install -e ".[local]"'
        ) from exc
    if not (Path(model_path) / "model.safetensors").is_file():
        raise RuntimeError(
            "本地模型尚未下载。请在项目目录运行：python -m techlit_assistant.local_embedding"
        )
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    model = (
        AutoModel.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True
        )
        .to("cpu")
        .eval()
    )
    if model.config.hidden_size != 384:
        raise ValueError("本地模型必须输出 384 维向量，请检查模型目录与配置")
    return tokenizer, model


class LocalEmbedder:
    """One cached unit vector per original chunk, averaging its token windows.

    Windows avoid silently discarding long chunks' tails. Averaging is a small-MVP
    compromise: it can dilute a short relevant passage in a very long chunk.
    """

    def __init__(self, model_path: Path, model_name: str = "e5"):
        if model_name not in {"e5", "minilm"}:
            raise ValueError("Local model must be e5 or minilm")
        self.model_path = model_path.resolve()
        self.model_name = model_name
        self.model = MINILM_ID if model_name == "minilm" else MODEL_ID
        self.max_tokens = 128 if model_name == "minilm" else MAX_TOKENS
        self.overlap = 32 if model_name == "minilm" else OVERLAP

    @property
    def cache_model_key(self) -> str:
        # Include local artifacts so replacing model/tokenizer files invalidates vectors.
        artifacts = []
        for filename in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ):
            path = self.model_path / filename
            if path.exists():
                stat = path.stat()
                artifacts.append((filename, stat.st_size, stat.st_mtime_ns))
        fingerprint = sha256(json.dumps([str(self.model_path), artifacts]).encode()).hexdigest()[
            :16
        ]
        revision = MINILM_REVISION if self.model_name == "minilm" else MODEL_REVISION
        return (
            f"local:{self.model}@{revision}:{PREPROCESS_VERSION}:"
            f"w{self.max_tokens}:o{self.overlap}:{fingerprint}"
        )

    @property
    def cache_dimensions(self) -> int:
        return 384

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, "" if self.model_name == "minilm" else "passage: ")

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], "" if self.model_name == "minilm" else "query: ")[0]

    def _encode(self, texts: Sequence[str], prefix: str, batch_size: int = 8) -> list[list[float]]:
        if not texts:
            return []
        config_path = self.model_path / "config.json"
        expected = MINILM_FILES if self.model_name == "minilm" else MODEL_FILES
        if config_path.is_file() and _file_sha256(config_path) != expected["config.json"]:
            raise ValueError("本地模型目录与 TECHLIT_LOCAL_MODEL 不匹配，请检查配置或重新下载")
        with _LOAD_LOCK:
            tokenizer, model = _load_cpu_model(str(self.model_path))
        import torch

        prefix_ids = tokenizer.encode(prefix.rstrip(), add_special_tokens=False) if prefix else []
        capacity = self.max_tokens - len(prefix_ids) - tokenizer.num_special_tokens_to_add(False)
        # Both pinned encoders use <s> ... </s>. Generic PreTrainedTokenizerFast
        # does not implement build_inputs_with_special_tokens; its Rust post-
        # processor adds them only during encode(). Preserve the exact token IDs
        # in overflow windows while adding the official boundary tokens explicitly.
        cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
        if cls_id is None or sep_id is None or tokenizer.num_special_tokens_to_add(False) != 2:
            raise ValueError("不支持的本地模型特殊标记配置")
        vectors = []
        # Bound intermediate memory to one chunk, even when importing a large library.
        for text in texts:
            # Tokenize the complete prefixed string once: SentencePiece whitespace
            # boundaries can differ when prefix and content are encoded separately.
            complete = tokenizer.encode(
                prefix + text, add_special_tokens=False, truncation=False, verbose=False
            )
            tokens = complete[len(prefix_ids) :]
            windows = token_windows(tokens, capacity, self.overlap)
            window_vectors = []
            for start in range(0, len(windows), batch_size):
                features = [
                    {"input_ids": [cls_id, *prefix_ids, *window, sep_id]}
                    for window in windows[start : start + batch_size]
                ]
                batch = tokenizer.pad(features, padding=True, return_tensors="pt")
                with torch.inference_mode():
                    output = model(**batch).last_hidden_state
                    mask = batch["attention_mask"].unsqueeze(-1).to(output.dtype)
                    pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                window_vectors.extend(normalized.cpu().numpy())
            vector = np.mean(window_vectors, axis=0)
            norm = np.linalg.norm(vector)
            if vector.shape != (384,) or not np.isfinite(vector).all() or norm <= 0:
                raise ValueError("本地模型返回了无效向量")
            vectors.append((vector / norm).tolist())
        return vectors


# Backward compatibility for code importing the original E5-only adapter.
LocalE5Embedder = LocalEmbedder


def main() -> None:
    from .config import Settings

    parser = argparse.ArgumentParser(description="Download and verify a pinned local encoder.")
    parser.add_argument("--model", choices=["e5", "minilm"])
    args = parser.parse_args()
    settings = Settings.from_env(Path.cwd())
    model_name = args.model or settings.local_model_name
    path = settings.local_model_path
    if args.model and args.model != settings.local_model_name:
        path = (
            settings.project_root
            / "data/models"
            / (
                "paraphrase-multilingual-MiniLM-L12-v2"
                if model_name == "minilm"
                else "multilingual-e5-small"
            )
        )
    print(f"Preparing {model_name} in {path}", flush=True)
    try:
        download_model(path, model_name)
        with _LOAD_LOCK:
            _load_cpu_model(str(path))
    except Exception as exc:  # Actionable CLI boundary; preserve partial files for inspection.
        raise SystemExit(
            f"模型准备失败：{exc}\n请检查网络/依赖后重试，或参阅 README 的“本地语义检索”。"
        ) from exc
    print("本地模型已就绪。后续向量推理无需联网，也不需要 API Key。", flush=True)


if __name__ == "__main__":
    main()
