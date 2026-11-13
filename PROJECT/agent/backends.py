"""固定 canonical 链路的在线后端与可替换接口。"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PROJECT.scripts.finetune.infer_finetuned_intent import (
    SYSTEM_PROMPT as INTENT_SYSTEM_PROMPT,
    parse_json_response,
)
from PROJECT.scripts.finetune.tool_trace_common import (
    PROMPT_VERSION,
    SYSTEM_PROMPT as PLANNER_SYSTEM_PROMPT,
    prompt_hash,
    render_messages,
    strict_parse_tools,
)

from .memory import fallback_tokenize
from .types import IntentResult, PlannerResult, RetrievalResult


class IntentBackend(Protocol):
    def classify(self, query: str) -> IntentResult: ...


class ToolRetriever(Protocol):
    def retrieve(self, query: str) -> RetrievalResult: ...


class PlannerBackend(Protocol):
    def count_input_tokens(self, query: str, candidate_tools: list[str]) -> int: ...

    def plan(self, query: str, candidate_tools: list[str]) -> PlannerResult: ...


def query_cache_key(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


class CachedIntentBackend:
    """按完整 query 复用已实际运行的 Intent 结果。"""

    def __init__(
        self,
        backend: IntentBackend,
        cache: dict[str, IntentResult],
        *,
        require_hit: bool = False,
    ):
        self.backend = backend
        self.cache = cache
        self.require_hit = require_hit

    def classify(self, query: str) -> IntentResult:
        cached = self.cache.get(query_cache_key(query))
        if cached is None and self.require_hit:
            raise KeyError(f"missing Intent cache: {query_cache_key(query)}")
        return (
            IntentResult(
                intent_type=cached.intent_type,
                keywords=cached.keywords,
                raw_output=cached.raw_output,
                latency_ms=0.0,
            )
            if cached is not None
            else self.backend.classify(query)
        )


class FixedEcommerceIntentBackend:
    """评测已知电商请求时跳过重复的 Intent 推理；不用于混合流量。"""

    def classify(self, query: str) -> IntentResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        return IntentResult(
            intent_type="电商类",
            raw_output="fixed_ecommerce_evaluation_scope",
            latency_ms=0.0,
        )


class CachedToolRetriever:
    """按完整 query 复用已实际运行的 canonical 检索结果。"""

    def __init__(
        self,
        backend: ToolRetriever,
        cache: dict[str, RetrievalResult],
        *,
        require_hit: bool = False,
    ):
        self.backend = backend
        self.cache = cache
        self.require_hit = require_hit

    def retrieve(self, query: str) -> RetrievalResult:
        cached = self.cache.get(query_cache_key(query))
        if cached is None and self.require_hit:
            raise KeyError(f"missing retrieval cache: {query_cache_key(query)}")
        return (
            RetrievalResult(
                candidate_tools=list(cached.candidate_tools),
                source_tools=list(cached.source_tools),
                reranked_tools=list(cached.reranked_tools),
                revision=cached.revision,
                route_hash=cached.route_hash,
                latency_ms=0.0,
                diagnostics={**cached.diagnostics, "cache_hit": True},
            )
            if cached is not None
            else self.backend.retrieve(query)
        )


class CachedPlannerBackend:
    """按 query 与候选顺序复用已实际运行的 Planner 结果。"""

    def __init__(
        self,
        backend: PlannerBackend,
        cache: dict[str, PlannerResult],
        *,
        require_hit: bool = False,
    ):
        self.backend = backend
        self.cache = cache
        self.require_hit = require_hit
        self._maximum_cached_input_tokens = max(
            (int(value.input_tokens) for value in cache.values()),
            default=0,
        )

    @staticmethod
    def key(query: str, candidate_tools: list[str]) -> str:
        payload = json.dumps(
            {"query": query, "candidate_tools": candidate_tools},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def count_input_tokens(self, query: str, candidate_tools: list[str]) -> int:
        key = self.key(query, candidate_tools)
        cached = self.cache.get(key)
        if cached is None and self.require_hit:
            raise KeyError(f"missing Planner cache: {key}")
        return (
            int(cached.input_tokens)
            if cached is not None
            else self.backend.count_input_tokens(query, candidate_tools)
        )

    def cached_input_tokens(
        self,
        query: str,
        candidate_tools: list[str],
        *,
        allow_missing: bool = False,
    ) -> int:
        cached = self.cache.get(self.key(query, candidate_tools))
        if cached is not None:
            return int(cached.input_tokens)
        if allow_missing:
            # A missing raw-query/candidate-order pair is needed only for the
            # resolved-query delta guard. Use the largest measured prompt count:
            # this is conservative for acceptance and avoids otherwise unused
            # Planner generation solely to count the same tokenizer template.
            return self._maximum_cached_input_tokens
        if self.require_hit:
            raise KeyError(
                f"missing Planner cache: {self.key(query, candidate_tools)}"
            )
        return self.backend.count_input_tokens(query, candidate_tools)

    def plan(self, query: str, candidate_tools: list[str]) -> PlannerResult:
        key = self.key(query, candidate_tools)
        cached = self.cache.get(key)
        if cached is None and self.require_hit:
            raise KeyError(f"missing Planner cache: {key}")
        return (
            PlannerResult(
                tools=list(cached.tools),
                strict_ok=cached.strict_ok,
                strict_error=cached.strict_error,
                raw_output=cached.raw_output,
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
                latency_ms=0.0,
                prompt_version=cached.prompt_version,
                prompt_hash=cached.prompt_hash,
                model_revision=cached.model_revision,
            )
            if cached is not None
            else self.backend.plan(query, candidate_tools)
        )




def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing canonical {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"canonical {label} hash changed: expected {expected}, found {actual}"
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_project_reference(project_root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if value == "PROJECT":
        return str(project_root)
    if value.startswith("PROJECT/"):
        return str(project_root.parent / value)
    candidate = project_root / value
    return str(candidate) if candidate.exists() else value


@dataclass(slots=True)
class CanonicalRoute:
    project_root: Path
    path: Path
    payload: dict[str, Any]
    sha256: str

    @classmethod
    def load(
        cls,
        project_root: Path,
        path: Path,
        required_revision: str | None = None,
    ) -> "CanonicalRoute":
        project_root = project_root.resolve()
        path = path.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 5:
            raise ValueError(f"unsupported canonical route schema: {path}")
        if payload.get("status") != "canonical":
            raise ValueError(f"retrieval route is not canonical: {path}")
        revision = str(payload.get("canonical_revision", ""))
        if required_revision and revision != required_revision:
            raise ValueError(
                f"canonical revision changed: expected {required_revision!r}, found {revision!r}"
            )
        required = {
            "catalog",
            "intent",
            "embedding",
            "first_stage",
            "reranker",
            "second_stage",
            "planner",
        }
        if not required <= set(payload):
            raise ValueError(f"canonical route lacks fields: {sorted(required - set(payload))}")
        first = payload["first_stage"]
        reranker = payload["reranker"]
        second = payload["second_stage"]
        planner = payload["planner"]
        intent = payload["intent"]
        if int(first["candidate_n_per_source"]) != 40:
            raise ValueError("Agent requires the frozen first-stage N=40 route")
        if int(reranker["candidate_k"]) != 20:
            raise ValueError("Agent requires exact Top-20 reranking")
        if int(second["output_k"]) != int(planner["candidate_k"]):
            raise ValueError("retrieval output_k and Planner candidate_k differ")
        if int(planner["candidate_k"]) != 10:
            raise ValueError("Agent requires the frozen Planner Top-10 interface")
        if str(planner.get("prompt_version", "")) != PROMPT_VERSION:
            raise ValueError("canonical route and Planner prompt versions differ")
        if planner.get("candidate_order") != "retrieval_order":
            raise ValueError("Agent requires retrieval-order Planner candidates")
        if planner.get("decoding") != "greedy":
            raise ValueError("Agent requires deterministic greedy Planner decoding")
        expected_planner_prompt = hashlib.sha256(
            PLANNER_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()
        if str(planner.get("system_prompt_sha256", "")) != expected_planner_prompt:
            raise ValueError("canonical route and Planner system prompts differ")
        expected_intent_prompt = hashlib.sha256(INTENT_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
        if str(intent.get("system_prompt_sha256", "")) != expected_intent_prompt:
            raise ValueError("canonical route and Intent system prompts differ")
        if int(intent.get("max_new_tokens", 0)) != 48:
            raise ValueError("Agent requires the frozen Intent max_new_tokens=48")
        embedding = payload["embedding"]
        embedding_files = {
            "checkpoint",
            "model_sha256",
            "vector_db",
            "vector_embeddings_sha256",
            "vector_docs_sha256",
            "vector_config_sha256",
            "tool_corpus",
            "tool_corpus_sha256",
            "query_instruction",
        }
        if not embedding_files <= set(embedding):
            raise ValueError(
                f"canonical embedding route lacks: {sorted(embedding_files - set(embedding))}"
            )
        catalog = payload["catalog"]
        if set(catalog) != {"path", "sha256"}:
            raise ValueError("canonical catalog contract requires only path and sha256")
        verify_file_sha256(
            Path(resolve_project_reference(project_root, str(catalog["path"]))),
            str(catalog["sha256"]),
            "tool catalog",
        )
        return cls(project_root=project_root, path=path, payload=payload, sha256=sha256_file(path))

    @property
    def revision(self) -> str:
        return str(self.payload["canonical_revision"])


def torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported torch dtype: {name}")
    return mapping[name]


def synchronize(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize(device)


class CanonicalIntentBackend:
    """Qwen3-0.6B full-LoRA Intent；关键词仍不进入检索。"""

    def __init__(
        self,
        route: CanonicalRoute,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        base_model: str | None = None,
        max_new_tokens: int | None = None,
        verify_adapter_hash: bool = True,
    ):
        self.route = route
        config = route.payload["intent"]
        self.base_model = resolve_project_reference(
            route.project_root, base_model or str(config["base_model"])
        )
        self.adapter = Path(resolve_project_reference(route.project_root, str(config["adapter"])))
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = int(
            config["max_new_tokens"] if max_new_tokens is None else max_new_tokens
        )
        self.verify_adapter_hash = bool(verify_adapter_hash)
        self._tokenizer: Any = None
        self._model: Any = None
        self._lock = threading.RLock()

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.adapter.exists():
            raise FileNotFoundError(f"missing canonical Intent adapter: {self.adapter}")
        if self.verify_adapter_hash:
            verify_file_sha256(
                self.adapter / "adapter_model.safetensors",
                str(self.route.payload["intent"]["adapter_model_sha256"]),
                "Intent adapter",
            )
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch_dtype(self.dtype),
            trust_remote_code=True,
        ).to(self.device)
        self._model = PeftModel.from_pretrained(self._model, self.adapter)
        self._model.eval()
        self._model.generation_config.temperature = None
        self._model.generation_config.top_p = None
        self._model.generation_config.top_k = None

    def classify(self, query: str) -> IntentResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        import torch

        with self._lock:
            self._load()
            messages = [
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            try:
                prompt = self._tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking")
                prompt = self._tokenizer.apply_chat_template(messages, **kwargs)
            encoded = self._tokenizer(prompt, return_tensors="pt").to(self.device)
            synchronize(self.device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            synchronize(self.device)
            latency_ms = (time.perf_counter() - started) * 1000.0
            output_ids = generated[0, encoded["input_ids"].shape[1] :]
            raw = self._tokenizer.decode(output_ids, skip_special_tokens=True)
            intent_type, keywords = parse_json_response(raw)
            if intent_type not in {"电商类", "闲聊类"}:
                raise RuntimeError(f"canonical Intent returned an invalid type: {intent_type!r}")
            return IntentResult(
                intent_type=intent_type,
                keywords=keywords,
                raw_output=raw,
                latency_ms=latency_ms,
            )


class _BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.tokenized = [fallback_tokenize(value) for value in documents]
        self.k1 = k1
        self.b = b
        self.lengths = [len(value) for value in self.tokenized]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.frequencies = [Counter(value) for value in self.tokenized]
        document_frequency: Counter[str] = Counter()
        for value in self.tokenized:
            document_frequency.update(set(value))
        count = len(self.tokenized)
        self.inverse = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        query_tokens = fallback_tokenize(query)
        result = []
        for length, frequencies in zip(self.lengths, self.frequencies):
            score = 0.0
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / max(1.0, self.average_length)
                )
                score += self.inverse.get(term, 0.0) * frequency * (self.k1 + 1.0) / denominator
            result.append(score)
        return result


def ranked_tools(tool_ids: list[str], scores: list[float]) -> list[str]:
    return [
        tool_ids[index]
        for index in sorted(
            range(len(tool_ids)),
            key=lambda index: (-float(scores[index]), tool_ids[index]),
        )
    ]


def weighted_rrf(
    vector_tools: list[str],
    bm25_tools: list[str],
    candidate_n: int,
    vector_weight: float,
    rrf_k: int,
) -> list[str]:
    vector_position = {tool: rank for rank, tool in enumerate(vector_tools[:candidate_n], 1)}
    bm25_position = {tool: rank for rank, tool in enumerate(bm25_tools[:candidate_n], 1)}
    missing = 9999
    return sorted(
        set(vector_position) | set(bm25_position),
        key=lambda tool: (
            -(
                vector_weight / (rrf_k + vector_position.get(tool, missing))
                + (1.0 - vector_weight) / (rrf_k + bm25_position.get(tool, missing))
            ),
            tool,
        ),
    )


def second_rrf(
    source_tools: list[str],
    reranked_tools: list[str],
    source_weight: float,
    rrf_k: int,
) -> list[str]:
    source_position = {tool: rank for rank, tool in enumerate(source_tools, 1)}
    reranker_position = {tool: rank for rank, tool in enumerate(reranked_tools, 1)}
    missing = 9999
    return sorted(
        set(source_position) | set(reranker_position),
        key=lambda tool: (
            -(
                source_weight / (rrf_k + source_position.get(tool, missing))
                + (1.0 - source_weight) / (rrf_k + reranker_position.get(tool, missing))
            ),
            source_position.get(tool, missing),
            reranker_position.get(tool, missing),
            tool,
        ),
    )


class CanonicalToolRetriever:
    """easy2 + BM25 RRF → Top-20 v2-m3 → 二级 RRF Top-10。"""

    def __init__(
        self,
        route: CanonicalRoute,
        embedding_device: str = "cuda:0",
        reranker_device: str = "cuda:0",
        reranker_dtype: str = "float16",
        vector_db_dir: Path | None = None,
        tool_corpus_path: Path | None = None,
        reranker_batch_size: int = 64,
        reranker_checkpoint: str | Path | None = None,
        verify_reranker_hash: bool = True,
    ):
        self.route = route
        self.embedding_device = embedding_device
        self.reranker_device = reranker_device
        self.reranker_dtype = reranker_dtype
        self.reranker_batch_size = int(reranker_batch_size)
        self.reranker_checkpoint = (
            Path(reranker_checkpoint).resolve() if reranker_checkpoint is not None else None
        )
        self.verify_reranker_hash = bool(verify_reranker_hash)
        embedding = route.payload["embedding"]
        self.vector_db_dir = vector_db_dir or Path(
            resolve_project_reference(route.project_root, str(embedding["vector_db"]))
        )
        self.tool_corpus_path = tool_corpus_path or Path(
            resolve_project_reference(route.project_root, str(embedding["tool_corpus"]))
        )
        verify_file_sha256(
            self.tool_corpus_path,
            str(embedding["tool_corpus_sha256"]),
            "tool corpus",
        )
        verify_file_sha256(
            self.vector_db_dir / "docs.jsonl",
            str(embedding["vector_docs_sha256"]),
            "vector DB docs",
        )
        verify_file_sha256(
            self.vector_db_dir / "config.json",
            str(embedding["vector_config_sha256"]),
            "vector DB config",
        )
        verify_file_sha256(
            self.vector_db_dir / "embeddings.npy",
            str(embedding["vector_embeddings_sha256"]),
            "vector DB embeddings",
        )
        self.docs = load_jsonl(self.tool_corpus_path)
        if len(self.docs) != 100:
            raise ValueError(f"canonical tool corpus must contain 100 tools, found {len(self.docs)}")
        self.tool_ids = [str(row["tool_id"]) for row in self.docs]
        if len(self.tool_ids) != len(set(self.tool_ids)):
            raise ValueError("tool corpus contains duplicate tool ids")
        self.docs_by_id = {str(row["tool_id"]): str(row.get("document", "")) for row in self.docs}
        self._bm25 = _BM25Index([self.docs_by_id[tool] for tool in self.tool_ids])
        self._vector_docs = load_jsonl(self.vector_db_dir / "docs.jsonl")
        self._vector_config = json.loads(
            (self.vector_db_dir / "config.json").read_text(encoding="utf-8")
        )
        self._vector_tool_ids = [str(row["tool_id"]) for row in self._vector_docs]
        if set(self._vector_tool_ids) != set(self.tool_ids):
            raise ValueError("vector DB and tool corpus contain different tools")
        self._embeddings: Any = None
        self._embedding_model: Any = None
        self._reranker: Any = None
        self._lock = threading.RLock()

    def _load_embedding(self) -> None:
        if self._embedding_model is not None:
            return
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model_reference = resolve_project_reference(
            self.route.project_root, str(self._vector_config["model_name_or_path"])
        )
        expected_model = Path(
            resolve_project_reference(
                self.route.project_root,
                str(self.route.payload["embedding"]["checkpoint"]),
            )
        ).resolve()
        if Path(model_reference).resolve() != expected_model:
            raise ValueError("vector DB points to a non-canonical embedding checkpoint")
        verify_file_sha256(
            expected_model / "model.safetensors",
            str(self.route.payload["embedding"]["model_sha256"]),
            "Embedding model",
        )
        kwargs: dict[str, Any] = {
            "device": self.embedding_device,
            "trust_remote_code": True,
        }
        try:
            self._embedding_model = SentenceTransformer(model_reference, **kwargs)
        except TypeError:
            kwargs.pop("trust_remote_code")
            self._embedding_model = SentenceTransformer(model_reference, **kwargs)
        self._embeddings = np.asarray(
            np.load(self.vector_db_dir / "embeddings.npy"), dtype="float32"
        )
        if self._embeddings.shape[0] != len(self._vector_tool_ids):
            raise ValueError("vector DB embedding rows and docs are not aligned")

    def _load_reranker(self) -> None:
        if self._reranker is not None:
            return
        from sentence_transformers import CrossEncoder

        reranker_path = str(
            self.reranker_checkpoint
            or resolve_project_reference(
                self.route.project_root, str(self.route.payload["reranker"]["checkpoint"])
            )
        )
        if not Path(reranker_path).exists():
            raise FileNotFoundError(f"missing canonical reranker: {reranker_path}")
        if self.verify_reranker_hash:
            verify_file_sha256(
                Path(reranker_path) / "model.safetensors",
                str(self.route.payload["reranker"]["model_sha256"]),
                "Reranker model",
            )
        model_kwargs = {"torch_dtype": torch_dtype(self.reranker_dtype)}
        self._reranker = CrossEncoder(
            reranker_path,
            device=self.reranker_device,
            trust_remote_code=True,
            max_length=512,
            model_kwargs=model_kwargs,
        )

    def retrieve(self, query: str) -> RetrievalResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        import numpy as np

        with self._lock:
            started = time.perf_counter()
            self._load_embedding()
            bm25_started = time.perf_counter()
            bm25_scores = self._bm25.scores(query)
            bm25_tools = ranked_tools(self.tool_ids, bm25_scores)
            bm25_ms = (time.perf_counter() - bm25_started) * 1000.0

            vector_started = time.perf_counter()
            query_instruction = str(
                self.route.payload["embedding"].get("query_instruction", "")
            )
            vector_query = f"{query_instruction}{query}" if query_instruction else query
            query_embedding = self._embedding_model.encode(
                [vector_query],
                batch_size=1,
                normalize_embeddings=bool(self._vector_config.get("normalize_embeddings", True)),
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            vector_scores = self._embeddings @ np.asarray(query_embedding, dtype="float32")
            vector_tools = ranked_tools(self._vector_tool_ids, vector_scores.tolist())
            vector_ms = (time.perf_counter() - vector_started) * 1000.0

            first = self.route.payload["first_stage"]
            source = weighted_rrf(
                vector_tools,
                bm25_tools,
                int(first["candidate_n_per_source"]),
                float(first["vector_weight"]),
                int(first["rrf_k"]),
            )
            reranker_config = self.route.payload["reranker"]
            reranker_candidates = source[: int(reranker_config["candidate_k"])]
            self._load_reranker()
            reranker_started = time.perf_counter()
            scores = self._reranker.predict(
                [(query, self.docs_by_id[tool]) for tool in reranker_candidates],
                batch_size=self.reranker_batch_size,
                show_progress_bar=False,
            )
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            scores = [float(value) for value in scores]
            source_position = {tool: index for index, tool in enumerate(source)}
            reranked = [
                tool
                for tool, _, _ in sorted(
                    zip(reranker_candidates, scores, range(len(reranker_candidates))),
                    key=lambda item: (-item[1], source_position[item[0]], item[0]),
                )
            ]
            reranker_ms = (time.perf_counter() - reranker_started) * 1000.0

            second = self.route.payload["second_stage"]
            final = second_rrf(
                source,
                reranked,
                float(second["source_weight"]),
                int(second["rrf_k"]),
            )[: int(second["output_k"])]
            if len(final) != int(second["output_k"]) or len(final) != len(set(final)):
                raise RuntimeError("canonical retrieval did not produce ten unique tools")
            total_ms = (time.perf_counter() - started) * 1000.0
            return RetrievalResult(
                candidate_tools=final,
                source_tools=source,
                reranked_tools=reranked,
                revision=self.route.revision,
                route_hash=self.route.sha256,
                latency_ms=total_ms,
                diagnostics={
                    "bm25_ms": bm25_ms,
                    "vector_ms": vector_ms,
                    "reranker_ms": reranker_ms,
                    "first_stage_candidates": len(source),
                    "reranker_candidates": len(reranker_candidates),
                },
            )

    def rerank_candidates(
        self,
        query: str,
        candidate_tools: list[str],
    ) -> tuple[list[str], list[float], float]:
        """用同一 canonical CrossEncoder 对给定候选做精确重排。"""

        if not query.strip():
            raise ValueError("query must not be empty")
        if not candidate_tools or len(candidate_tools) != len(set(candidate_tools)):
            raise ValueError("candidate tools must be non-empty and unique")
        if not set(candidate_tools) <= set(self.tool_ids):
            raise ValueError("candidate tools contain an unknown tool id")
        with self._lock:
            self._load_reranker()
            started = time.perf_counter()
            raw_scores = self._reranker.predict(
                [(query, self.docs_by_id[tool]) for tool in candidate_tools],
                batch_size=self.reranker_batch_size,
                show_progress_bar=False,
            )
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            scores = [float(value) for value in raw_scores]
            source_position = {tool: index for index, tool in enumerate(candidate_tools)}
            ranked = [
                tool
                for tool, _ in sorted(
                    zip(candidate_tools, scores),
                    key=lambda item: (-item[1], source_position[item[0]], item[0]),
                )
            ]
            score_by_tool = dict(zip(candidate_tools, scores))
            ranked_scores = [score_by_tool[tool] for tool in ranked]
            latency_ms = (time.perf_counter() - started) * 1000.0
            return ranked, ranked_scores, latency_ms


class CanonicalPlannerBackend:
    """最新 DPO checkpoint，保持 tool-trace-v1 Prompt 与严格 JSON 解析。"""

    def __init__(
        self,
        route: CanonicalRoute,
        catalog_by_id: dict[str, dict[str, Any]],
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        base_model: str | None = None,
        verify_adapter_hash: bool = True,
    ):
        self.route = route
        self.catalog_by_id = catalog_by_id
        self.catalog_ids = set(catalog_by_id)
        if len(self.catalog_ids) != 100:
            raise ValueError(
                f"canonical Planner catalog must contain 100 tools, found {len(self.catalog_ids)}"
            )
        config = route.payload["planner"]
        self.base_model = resolve_project_reference(
            route.project_root, base_model or str(config["base_model"])
        )
        self.adapter = Path(resolve_project_reference(route.project_root, str(config["adapter"])))
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = int(config["max_new_tokens"])
        self.verify_adapter_hash = bool(verify_adapter_hash)
        adapter_file = self.adapter / "adapter_model.safetensors"
        self.model_revision = (
            str(config["adapter_model_sha256"])
            if self.verify_adapter_hash
            else sha256_file(adapter_file)
        )
        self._tokenizer: Any = None
        self._model: Any = None
        self._lock = threading.RLock()

    def _load_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not self.adapter.exists():
            raise FileNotFoundError(f"missing canonical Planner adapter: {self.adapter}")
        if self.verify_adapter_hash:
            verify_file_sha256(
                self.adapter / "adapter_model.safetensors",
                str(self.route.payload["planner"]["adapter_model_sha256"]),
                "Planner adapter",
            )
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        self._load_tokenizer()
        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch_dtype(self.dtype),
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).to(self.device)
        self._model = PeftModel.from_pretrained(self._model, self.adapter)
        self._model.eval()
        self._model.generation_config.temperature = None
        self._model.generation_config.top_p = None
        self._model.generation_config.top_k = None

    def _prompt(self, query: str, candidate_tools: list[str]) -> tuple[str, list[dict[str, str]]]:
        if len(candidate_tools) != int(self.route.payload["planner"]["candidate_k"]):
            raise ValueError("Planner must receive exactly the canonical Top-10 candidates")
        if len(candidate_tools) != len(set(candidate_tools)):
            raise ValueError("Planner candidates must be unique")
        messages = render_messages(query, candidate_tools, self.catalog_by_id, "rag")
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        try:
            prompt = self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking")
            prompt = self._tokenizer.apply_chat_template(messages, **kwargs)
        return prompt, messages

    def count_input_tokens(self, query: str, candidate_tools: list[str]) -> int:
        with self._lock:
            self._load_tokenizer()
            prompt, _ = self._prompt(query, candidate_tools)
            encoded = self._tokenizer(prompt, return_tensors="pt")
            return int(encoded["attention_mask"].sum().item())

    def plan(self, query: str, candidate_tools: list[str]) -> PlannerResult:
        import torch

        with self._lock:
            self._load_model()
            prompt, messages = self._prompt(query, candidate_tools)
            encoded = self._tokenizer(prompt, return_tensors="pt").to(self.device)
            synchronize(self.device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            synchronize(self.device)
            latency_ms = (time.perf_counter() - started) * 1000.0
            input_width = encoded["input_ids"].shape[1]
            output_ids = generated[0, input_width:]
            raw = self._tokenizer.decode(output_ids, skip_special_tokens=True)
            tools, strict_ok, strict_error = strict_parse_tools(
                raw,
                self.catalog_ids,
                set(candidate_tools),
            )
            output_tokens = int((output_ids != self._tokenizer.pad_token_id).sum().item())
            return PlannerResult(
                tools=tools,
                strict_ok=strict_ok,
                strict_error=strict_error,
                raw_output=raw,
                input_tokens=int(encoded["attention_mask"].sum().item()),
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                prompt_version=PROMPT_VERSION,
                prompt_hash=prompt_hash(messages, PROMPT_VERSION),
                model_revision=self.model_revision,
            )

    def score_single_tool_candidates(
        self,
        query: str,
        candidate_tools: list[str],
        normalization: str = "mean",
    ) -> tuple[PlannerResult, dict[str, Any]]:
        """用 Planner 条件似然在固定 Top-10 中选择一个工具。"""

        if normalization not in {"sum", "mean"}:
            raise ValueError("normalization must be sum or mean")
        import torch
        import torch.nn.functional as functional

        with self._lock:
            self._load_model()
            prompt, messages = self._prompt(query, candidate_tools)
            prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
            sequences: list[list[int]] = []
            labels: list[list[int]] = []
            answers: list[str] = []
            for tool in candidate_tools:
                answer = json.dumps(
                    {"tools": [tool]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                answer_ids = self._tokenizer.encode(answer, add_special_tokens=False)
                answers.append(answer)
                sequences.append(prompt_ids + answer_ids)
                labels.append([-100] * len(prompt_ids) + answer_ids)

            maximum = max(len(value) for value in sequences)
            padded_ids = []
            padded_masks = []
            padded_labels = []
            for sequence, label in zip(sequences, labels):
                padding = maximum - len(sequence)
                padded_ids.append([self._tokenizer.pad_token_id] * padding + sequence)
                padded_masks.append([0] * padding + [1] * len(sequence))
                padded_labels.append([-100] * padding + label)
            input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(
                padded_masks, dtype=torch.long, device=self.device
            )
            label_tensor = torch.tensor(
                padded_labels, dtype=torch.long, device=self.device
            )

            synchronize(self.device)
            started = time.perf_counter()
            with torch.inference_mode():
                logits = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits[:, :-1, :]
                shifted_labels = label_tensor[:, 1:]
                mask = shifted_labels.ne(-100)
                target_ids = shifted_labels.masked_fill(~mask, 0)
                token_log_probs = functional.log_softmax(
                    logits.float(), dim=-1
                ).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                sums = (token_log_probs * mask).sum(dim=-1)
                counts = mask.sum(dim=-1)
                means = sums / counts.clamp_min(1)
            synchronize(self.device)
            latency_ms = (time.perf_counter() - started) * 1000.0
            sum_values = [float(value) for value in sums.cpu().tolist()]
            mean_values = [float(value) for value in means.cpu().tolist()]
            selected_scores = mean_values if normalization == "mean" else sum_values
            selected_index = max(
                range(len(candidate_tools)),
                key=lambda index: (selected_scores[index], -index),
            )
            selected_tool = candidate_tools[selected_index]
            raw = answers[selected_index]
            result = PlannerResult(
                tools=[selected_tool],
                strict_ok=True,
                strict_error="",
                raw_output=raw,
                input_tokens=len(prompt_ids),
                output_tokens=len(labels[selected_index]) - len(prompt_ids),
                latency_ms=latency_ms,
                prompt_version=PROMPT_VERSION,
                prompt_hash=prompt_hash(messages, PROMPT_VERSION),
                model_revision=self.model_revision,
            )
            diagnostics = {
                "normalization": normalization,
                "selected_tool": selected_tool,
                "batch_size": len(candidate_tools),
                "scored_tokens": sum(len(value) for value in sequences),
                "scores": [
                    {
                        "tool": tool,
                        "sum_log_probability": sum_value,
                        "mean_log_probability": mean_value,
                    }
                    for tool, sum_value, mean_value in zip(
                        candidate_tools, sum_values, mean_values
                    )
                ],
            }
            return result, diagnostics
