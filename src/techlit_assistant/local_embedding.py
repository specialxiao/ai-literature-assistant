"""Optional, CPU-only E5 encoder. Inference never downloads model files."""

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
PREPROCESS_VERSION = "e5-prefix-mean-windows-v1"
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


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(directory: Path) -> None:
    """Use verified official HTTPS GETs, including behind HEAD-incompatible proxies."""
    directory.mkdir(parents=True, exist_ok=True)
    for filename, expected_hash in MODEL_FILES.items():
        target = directory / filename
        if target.is_file() and _file_sha256(target) == expected_hash:
            print(f"Verified cached {filename}", flush=True)
            continue
        print(f"Downloading {filename} ...", flush=True)
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{filename}"
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
        raise ValueError("本地模型目录必须是 multilingual-e5-small（384 维）")
    return tokenizer, model


class LocalE5Embedder:
    """One cached unit vector per original chunk, averaging its token windows.

    Windows avoid silently discarding long chunks' tails. Averaging is a small-MVP
    compromise: it can dilute a short relevant passage in a very long chunk.
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path.resolve()
        self.model = MODEL_ID

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
        return f"local:{MODEL_ID}@{MODEL_REVISION}:{PREPROCESS_VERSION}:{fingerprint}"

    @property
    def cache_dimensions(self) -> int:
        return 384

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, "passage: ")

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], "query: ")[0]

    def _encode(self, texts: Sequence[str], prefix: str, batch_size: int = 8) -> list[list[float]]:
        if not texts:
            return []
        with _LOAD_LOCK:
            tokenizer, model = _load_cpu_model(str(self.model_path))
        import torch

        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        capacity = MAX_TOKENS - len(prefix_ids) - tokenizer.num_special_tokens_to_add(False)
        vectors = []
        # Bound intermediate memory to one chunk, even when importing a large library.
        for text in texts:
            tokens = tokenizer.encode(
                text, add_special_tokens=False, truncation=False, verbose=False
            )
            windows = token_windows(tokens, capacity)
            window_vectors = []
            for start in range(0, len(windows), batch_size):
                features = [
                    {"input_ids": tokenizer.build_inputs_with_special_tokens(prefix_ids + window)}
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


def main() -> None:
    from .config import Settings

    parser = argparse.ArgumentParser(description="Download the pinned bilingual E5 model once.")
    parser.parse_args()
    settings = Settings.from_env(Path.cwd())
    print(f"Downloading {MODEL_ID} to {settings.local_model_path}", flush=True)
    try:
        download_model(settings.local_model_path)
        with _LOAD_LOCK:
            _load_cpu_model(str(settings.local_model_path))
    except Exception as exc:  # Actionable CLI boundary; preserve partial files for inspection.
        raise SystemExit(
            f"模型准备失败：{exc}\n请检查网络/依赖后重试，或参阅 README 的“本地语义检索”。"
        ) from exc
    print("本地模型已就绪。后续向量推理无需联网，也不需要 API Key。", flush=True)


if __name__ == "__main__":
    main()
