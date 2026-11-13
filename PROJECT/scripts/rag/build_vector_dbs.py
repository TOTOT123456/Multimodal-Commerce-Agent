#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build one Faiss vector database per embedding model for the 100-tool corpus."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import RAG_DIR, TOOL_CORPUS_PATH, load_jsonl


DEFAULT_OUTPUT = RAG_DIR / "vector_dbs"

MODEL_CONFIGS = {
    "bge-base": "BAAI/bge-base-zh-v1.5",
    "bge-large": "BAAI/bge-large-zh-v1.5",
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}


def parse_model_paths(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--model-path must be KEY=PATH, got: {item}")
        key, value = item.split("=", 1)
        if not key.strip() or not value.strip():
            raise ValueError(f"--model-path must be KEY=PATH, got: {item}")
        out[key] = value
    return out


def load_sentence_transformer(model_name_or_path: str, device: str):
    from sentence_transformers import SentenceTransformer

    kwargs = {"trust_remote_code": True}
    if device != "auto":
        kwargs["device"] = device
    try:
        return SentenceTransformer(model_name_or_path, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        return SentenceTransformer(model_name_or_path, **kwargs)


def build_one(
    key: str,
    model_name_or_path: str,
    docs: list[dict],
    output_dir: Path,
    batch_size: int,
    device: str,
    normalize: bool,
    text_field: str,
) -> None:
    import faiss
    import numpy as np

    missing = [doc.get("tool_id", "<unknown>") for doc in docs if text_field not in doc]
    if missing:
        raise ValueError(f"text field {text_field!r} missing in docs, examples={missing[:5]}")

    texts = [str(doc[text_field]) for doc in docs]
    db_dir = output_dir / key
    db_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model = load_sentence_transformer(model_name_or_path, device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    encode_seconds = time.perf_counter() - t0

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(db_dir / "index.faiss"))
    np.save(db_dir / "embeddings.npy", embeddings)

    with (db_dir / "docs.jsonl").open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    config = {
        "model_key": key,
        "model_name_or_path": model_name_or_path,
        "doc_count": len(docs),
        "embedding_dim": int(embeddings.shape[1]),
        "normalize_embeddings": normalize,
        "metric": "cosine_via_faiss_inner_product" if normalize else "inner_product",
        "text_field": text_field,
        "encode_seconds": round(encode_seconds, 4),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (db_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"built {key}: {len(docs)} docs, dim={embeddings.shape[1]} -> {db_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, default=TOOL_CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", default=["bge-base", "bge-large", "qwen3-embedding-0.6b"], help="Model keys to build, or 'all'.")
    parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        help="Override or add a model path, e.g. bge-base=/path/to/model or bge-base-ft=/path/to/final.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--text-field", default="document")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--force", action="store_true", help="Delete existing db dirs first.")
    args = parser.parse_args()

    try:
        import faiss  # noqa: F401
        import numpy  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Use the llm_bge conda env or install the RAG requirements.\n"
            f"Original error: {exc}"
        )

    docs = load_jsonl(args.docs)
    if not docs:
        raise SystemExit(f"empty docs: {args.docs}")

    model_paths = parse_model_paths(args.model_path)
    models = list(MODEL_CONFIGS) if "all" in args.models else args.models
    unknown = [model for model in models if model not in MODEL_CONFIGS and model not in model_paths]
    if unknown:
        raise SystemExit(f"unknown model key(s): {unknown}; choices={list(MODEL_CONFIGS)} or pass --model-path KEY=PATH")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in models:
        db_dir = args.output_dir / key
        if db_dir.exists() and args.force:
            shutil.rmtree(db_dir)
        elif db_dir.exists() and (db_dir / "index.faiss").exists():
            print(f"skip existing db: {db_dir} (use --force to rebuild)")
            continue
        build_one(
            key=key,
            model_name_or_path=model_paths.get(key, MODEL_CONFIGS.get(key, key)),
            docs=docs,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            normalize=not args.no_normalize,
            text_field=args.text_field,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
