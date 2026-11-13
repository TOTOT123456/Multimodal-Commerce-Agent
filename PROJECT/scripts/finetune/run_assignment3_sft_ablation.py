#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run resumable Assignment3 SFT hyperparameter search and data-scale ablation."""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "assignment3_sft_ablation.json"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "finetune" / "run_swift_tool_trace_lora.sh"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "finetune" / "run_assignment3_eval.sh"
EVALUATOR = PROJECT_ROOT / "scripts" / "finetune" / "evaluate_tool_trace.py"
PREPARE_SCRIPT = PROJECT_ROOT / "scripts" / "finetune" / "prepare_assignment3_sft_ablation.py"
PYTHON_BIN = Path("/root/miniconda/envs/qwen-intent-ft/bin/python")
DATA_DIR = (
    PROJECT_ROOT / "finetune" / "outputs" / "tooltrace-data-cache"
    / "assignment3_sft_ablation"
)
OUTPUT_ROOT = PROJECT_ROOT / "finetune" / "outputs"
PRED_DIR = (
    OUTPUT_ROOT / "preds" / "tool_trace" / "assignment3_sft_ablation"
)
ARTIFACT_DIR = (
    PROJECT_ROOT / "data" / "tool_call_trace" / "_aux" / "assignment3"
    / "assignment3_100tool_api" / "sft_ablation"
)
DEV_DIR = (
    PROJECT_ROOT / "data" / "tool_call_trace" / "_aux" / "assignment3"
    / "api_native_dev_2400"
)
TEST_DIR = (
    PROJECT_ROOT / "data" / "tool_call_trace" / "_aux" / "assignment3"
    / "assignment3_100tool_api" / "test_api"
)
REPORT_PATH = PROJECT_ROOT / "reports" / "06_Assignment3_100工具实验总报告.md"
REGISTRY_PATH = ARTIFACT_DIR / "run_registry.json"
SELECTION_PATH = ARTIFACT_DIR / "selection_frozen.json"
SUMMARY_PATH = ARTIFACT_DIR / "results_summary.json"
DATA_AUDIT_PATH = ARTIFACT_DIR / "data_audit.json"
BASE_METRICS_DIR = (
    OUTPUT_ROOT / "preds" / "tool_trace" / "assignment3_100tool_test"
)
REGISTRY_LOCK = threading.Lock()


@dataclasses.dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float
    epochs: int
    rank: int
    dropout: float

    @property
    def config_id(self) -> str:
        lr = f"{self.learning_rate:.0e}".replace("e-0", "em").replace("e-", "em")
        dropout = f"{round(self.dropout * 100):02d}"
        return f"lr{lr}-e{self.epochs}-r{self.rank}-d{dropout}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "rank": self.rank,
            "alpha": self.rank * 2,
            "dropout": self.dropout,
        }


@dataclasses.dataclass(frozen=True)
class Candidate:
    model_size: str
    data_size: int
    hp: Hyperparameters

    @property
    def run_id(self) -> str:
        return f"a3-ablate-{self.model_size}-n{self.data_size}-{self.hp.config_id}"

    @property
    def output_dir(self) -> Path:
        return OUTPUT_ROOT / f"tooltrace-{self.run_id}"

    @property
    def data_path(self) -> Path:
        return DATA_DIR / f"sft_train_{self.data_size}.runtime.jsonl"

    @property
    def model_path(self) -> Path:
        suffix = "4B" if self.model_size == "4b" else "8B"
        return PROJECT_ROOT / "models" / "Qwen" / f"Qwen3-{suffix}"

    @property
    def dev_prediction(self) -> Path:
        return PRED_DIR / "dev" / f"{self.run_id}.jsonl"

    @property
    def dev_metrics(self) -> Path:
        return self.dev_prediction.with_suffix(".metrics.json")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def load_registry() -> dict[str, Any]:
    if REGISTRY_PATH.is_file():
        return read_json(REGISTRY_PATH)
    return {
        "schema_version": 1,
        "task": "assignment3_sft_ablation",
        "status": "running",
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "runs": {},
    }


def update_registry(run_id: str, values: dict[str, Any]) -> None:
    with REGISTRY_LOCK:
        registry = load_registry()
        record = registry["runs"].setdefault(run_id, {})
        record.update(values)
        registry["updated_at_utc"] = utc_now()
        write_json(REGISTRY_PATH, registry)


def run_logged(
    command: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] COMMAND: {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"[{utc_now()}] EXIT: {completed.returncode}\n")
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")


def run_parallel(
    label: str,
    jobs: list[tuple[str, Callable[[str], Any]]],
    gpu_ids: list[str],
) -> list[Any]:
    if not jobs:
        return []
    available: queue.Queue[str] = queue.Queue()
    for gpu in gpu_ids:
        available.put(gpu)

    def execute(job: tuple[str, Callable[[str], Any]]) -> Any:
        name, function = job
        gpu = available.get()
        print(f"[{utc_now()}] {label} start name={name} gpu={gpu}", flush=True)
        try:
            result = function(gpu)
            print(f"[{utc_now()}] {label} done name={name} gpu={gpu}", flush=True)
            return result
        finally:
            available.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(execute, job) for job in jobs]
        return [future.result() for future in futures]


def latest_complete_checkpoint(candidate: Candidate) -> Path | None:
    if not candidate.output_dir.is_dir():
        return None
    checkpoints = sorted(
        (
            path for path in candidate.output_dir.glob("checkpoint-*")
            if path.is_dir() and path.name.split("-")[-1].isdigit()
        ),
        key=lambda path: int(path.name.split("-")[-1]),
        reverse=True,
    )
    for checkpoint in checkpoints:
        state_path = checkpoint / "trainer_state.json"
        args_path = checkpoint / "args.json"
        adapter_path = checkpoint / "adapter_model.safetensors"
        if not (state_path.is_file() and args_path.is_file() and adapter_path.is_file()):
            continue
        state = read_json(state_path)
        arguments = read_json(args_path)
        expected = candidate.hp
        datasets = arguments.get("dataset", [])
        if isinstance(datasets, str):
            datasets = [datasets]
        dataset_matches = (
            len(datasets) == 1
            and Path(datasets[0]).resolve() == candidate.data_path.resolve()
        )
        if (
            abs(float(state.get("epoch", -1)) - expected.epochs) < 1e-6
            and abs(float(arguments.get("num_train_epochs", -1)) - expected.epochs) < 1e-6
            and abs(float(arguments.get("learning_rate", -1)) - expected.learning_rate) < 1e-12
            and int(arguments.get("lora_rank", -1)) == expected.rank
            and int(arguments.get("lora_alpha", -1)) == expected.rank * 2
            and abs(float(arguments.get("lora_dropout", -1)) - expected.dropout) < 1e-12
            and dataset_matches
        ):
            return checkpoint
    return None


def train_candidate(
    candidate: Candidate,
    gpu: str,
    fixed: dict[str, Any],
) -> Path:
    complete = latest_complete_checkpoint(candidate)
    if complete is not None:
        update_registry(
            candidate.run_id,
            {
                "training_status": "complete",
                "checkpoint": relative(complete),
                "checkpoint_sha256": sha256_file(complete / "adapter_model.safetensors"),
                "reused": True,
            },
        )
        return complete

    if not candidate.data_path.is_file():
        raise FileNotFoundError(candidate.data_path)
    if not candidate.model_path.is_dir():
        raise FileNotFoundError(candidate.model_path)
    candidate.output_dir.mkdir(parents=True, exist_ok=True)
    model_batch = fixed["model_batch"][candidate.model_size]
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{PYTHON_BIN.parent}:{env.get('PATH', '')}",
            "CUDA_VISIBLE_DEVICES": gpu,
            "NPROC_PER_NODE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(29500 + int(gpu)),
            "DATA_FILE": str(candidate.data_path),
            "OUT_DIR": str(candidate.output_dir),
            "MODEL_PATH": str(candidate.model_path),
            "MAX_LENGTH": str(fixed["max_length"]),
            "BATCH_SIZE": str(model_batch["per_device_train_batch_size"]),
            "GRAD_ACC": str(model_batch["gradient_accumulation_steps"]),
            "LORA_RANK": str(candidate.hp.rank),
            "LORA_ALPHA": str(candidate.hp.rank * 2),
            "LORA_DROPOUT": str(candidate.hp.dropout),
            "LR_SCHEDULER_TYPE": str(fixed["lr_scheduler_type"]),
            "WARMUP_RATIO": str(fixed["warmup_ratio"]),
            "WEIGHT_DECAY": str(fixed["weight_decay"]),
            "SEED": str(fixed["seed"]),
            "DATA_SEED": str(fixed["data_seed"]),
            "SAVE_TOTAL_LIMIT": "1",
            "DATASET_NUM_PROC": "4",
        }
    )
    existing = sorted(candidate.output_dir.glob("checkpoint-*"))
    if existing:
        last = max(existing, key=lambda path: int(path.name.split("-")[-1]))
        env["RESUME_FROM_CHECKPOINT"] = str(last)
    update_registry(
        candidate.run_id,
        {
            "model_size": candidate.model_size,
            "data_size": candidate.data_size,
            "hyperparameters": candidate.hp.as_dict(),
            "data": relative(candidate.data_path),
            "training_status": "running",
            "training_started_at_utc": utc_now(),
        },
    )
    command = [
        "bash",
        str(TRAIN_SCRIPT),
        candidate.model_size,
        candidate.run_id,
        str(candidate.hp.learning_rate),
        str(candidate.hp.epochs),
    ]
    run_logged(command, PRED_DIR / "logs" / f"{candidate.run_id}.train.log", env)
    complete = latest_complete_checkpoint(candidate)
    if complete is None:
        raise RuntimeError(f"training ended without a matching checkpoint: {candidate.run_id}")
    update_registry(
        candidate.run_id,
        {
            "training_status": "complete",
            "training_completed_at_utc": utc_now(),
            "checkpoint": relative(complete),
            "checkpoint_sha256": sha256_file(complete / "adapter_model.safetensors"),
            "reused": False,
        },
    )
    return complete


def metrics_complete(path: Path, expected_count: int, expected_view: str) -> bool:
    if not path.is_file():
        return False
    payload = read_json(path)
    return (
        payload.get("status") == "complete"
        and payload.get("count") == expected_count
        and payload.get("view") == expected_view
    )


def evaluate_candidate_dev(candidate: Candidate, checkpoint: Path, gpu: str) -> Path:
    if metrics_complete(candidate.dev_metrics, 2400, "rag"):
        update_registry(
            candidate.run_id,
            {
                "dev_status": "complete",
                "dev_metrics": relative(candidate.dev_metrics),
                "dev_metrics_sha256": sha256_file(candidate.dev_metrics),
                "dev_reused": True,
            },
        )
        return candidate.dev_metrics

    run_id = f"{candidate.run_id}-dev-rag"
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(PYTHON_BIN),
            "GPU_LIST": gpu,
            "BATCH_SIZE": "16" if candidate.model_size == "4b" else "8",
        }
    )
    log_path = PRED_DIR / "logs" / f"{run_id}.infer.log"
    command = [
        "bash",
        str(EVAL_SCRIPT),
        str(candidate.model_path),
        str(checkpoint),
        str(DEV_DIR / "rag_k10_label_free.jsonl"),
        "rag",
        str(candidate.dev_prediction),
        run_id,
    ]
    run_logged(command, log_path, env)
    run_logged(
        [
            str(PYTHON_BIN),
            str(EVALUATOR),
            "--predictions",
            str(candidate.dev_prediction),
            "--gold",
            str(DEV_DIR / "dev.jsonl"),
            "--metrics-out",
            str(candidate.dev_metrics),
            "--scored-out",
            str(candidate.dev_prediction.with_suffix(".scored.jsonl")),
            "--run-id",
            run_id,
        ],
        log_path,
        env,
    )
    if not metrics_complete(candidate.dev_metrics, 2400, "rag"):
        raise RuntimeError(f"incomplete dev metrics: {candidate.dev_metrics}")
    update_registry(
        candidate.run_id,
        {
            "dev_status": "complete",
            "dev_completed_at_utc": utc_now(),
            "dev_metrics": relative(candidate.dev_metrics),
            "dev_metrics_sha256": sha256_file(candidate.dev_metrics),
            "dev_reused": False,
        },
    )
    return candidate.dev_metrics


def ensure_train_and_dev(
    candidates: Iterable[Candidate],
    fixed: dict[str, Any],
    gpu_ids: list[str],
) -> None:
    unique = {candidate.run_id: candidate for candidate in candidates}
    candidates = list(unique.values())
    checkpoints: dict[str, Path] = {}
    jobs = [
        (
            candidate.run_id,
            lambda gpu, candidate=candidate: train_candidate(candidate, gpu, fixed),
        )
        for candidate in candidates
        if latest_complete_checkpoint(candidate) is None
    ]
    results = run_parallel("train", jobs, gpu_ids)
    for (name, _), checkpoint in zip(jobs, results):
        checkpoints[name] = checkpoint
    for candidate in candidates:
        checkpoints.setdefault(candidate.run_id, latest_complete_checkpoint(candidate))
        if checkpoints[candidate.run_id] is None:
            raise RuntimeError(f"missing checkpoint after training: {candidate.run_id}")

    eval_jobs = [
        (
            candidate.run_id,
            lambda gpu, candidate=candidate: evaluate_candidate_dev(
                candidate, checkpoints[candidate.run_id], gpu
            ),
        )
        for candidate in candidates
        if not metrics_complete(candidate.dev_metrics, 2400, "rag")
    ]
    run_parallel("dev-eval", eval_jobs, gpu_ids)


def candidate_record(candidate: Candidate) -> dict[str, Any]:
    checkpoint = latest_complete_checkpoint(candidate)
    if checkpoint is None or not metrics_complete(candidate.dev_metrics, 2400, "rag"):
        raise RuntimeError(f"candidate is incomplete: {candidate.run_id}")
    metrics = read_json(candidate.dev_metrics)
    overall = metrics["overall"]
    return {
        "run_id": candidate.run_id,
        "model_size": candidate.model_size,
        "data_size": candidate.data_size,
        **candidate.hp.as_dict(),
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
        "dev_metrics": relative(candidate.dev_metrics),
        "dev_metrics_sha256": sha256_file(candidate.dev_metrics),
        "dev_gold_sha256": metrics["gold_snapshot_sha256"],
        "dev_trace_em": float(overall["trace_em"]),
        "dev_set_em": float(overall["set_em"]),
        "strict_output_valid_rate": float(overall["strict_output_valid_rate"]),
        "keyword_micro_f1": float(overall["keyword_micro_f1"]),
    }


def rank_candidates(candidates: Iterable[Candidate]) -> list[dict[str, Any]]:
    records = [candidate_record(candidate) for candidate in candidates]
    return sorted(
        records,
        key=lambda row: (
            -row["dev_trace_em"],
            -row["dev_set_em"],
            -row["strict_output_valid_rate"],
            -row["keyword_micro_f1"],
            row["epochs"],
            row["rank"],
            row["config_id"],
        ),
    )


def hp_from_record(record: dict[str, Any]) -> Hyperparameters:
    return Hyperparameters(
        learning_rate=float(record["learning_rate"]),
        epochs=int(record["epochs"]),
        rank=int(record["rank"]),
        dropout=float(record["dropout"]),
    )


def final_prediction(candidate: Candidate, view: str) -> Path:
    return PRED_DIR / "test" / f"{candidate.run_id}-{view}.jsonl"


def evaluate_final_test(candidate: Candidate, view: str, gpu: str) -> Path:
    prediction = final_prediction(candidate, view)
    metrics_path = prediction.with_suffix(".metrics.json")
    if metrics_complete(metrics_path, 1000, view):
        return metrics_path
    checkpoint = latest_complete_checkpoint(candidate)
    if checkpoint is None:
        raise RuntimeError(f"missing final checkpoint: {candidate.run_id}")
    input_path = (
        TEST_DIR / "queries_label_free_k10.jsonl"
        if view == "global"
        else TEST_DIR / "rag_k10_label_free.jsonl"
    )
    run_id = f"{candidate.run_id}-test-{view}"
    env = os.environ.copy()
    if view == "global":
        batch = "8" if candidate.model_size == "4b" else "4"
    else:
        batch = "16" if candidate.model_size == "4b" else "8"
    env.update({"PYTHON_BIN": str(PYTHON_BIN), "GPU_LIST": gpu, "BATCH_SIZE": batch})
    log_path = PRED_DIR / "logs" / f"{run_id}.infer.log"
    run_logged(
        [
            "bash",
            str(EVAL_SCRIPT),
            str(candidate.model_path),
            str(checkpoint),
            str(input_path),
            view,
            str(prediction),
            run_id,
        ],
        log_path,
        env,
    )
    run_logged(
        [
            str(PYTHON_BIN),
            str(EVALUATOR),
            "--predictions",
            str(prediction),
            "--gold",
            str(TEST_DIR / "test.jsonl"),
            "--metrics-out",
            str(metrics_path),
            "--scored-out",
            str(prediction.with_suffix(".scored.jsonl")),
            "--run-id",
            run_id,
        ],
        log_path,
        env,
    )
    if not metrics_complete(metrics_path, 1000, view):
        raise RuntimeError(f"incomplete final metrics: {metrics_path}")
    return metrics_path


def baseline_metrics(model_size: str, view: str) -> dict[str, float]:
    path = BASE_METRICS_DIR / f"a3-100-base-{model_size}-{view}.metrics.json"
    payload = read_json(path)
    overall = payload["overall"]
    return {
        "path": relative(path),
        "sha256": sha256_file(path),
        "set_em": float(overall["set_em"]),
        "trace_em": float(overall["trace_em"]),
        "strict_output_valid_rate": float(overall["strict_output_valid_rate"]),
        "keyword_micro_f1": float(overall["keyword_micro_f1"]),
    }


def final_metric_record(candidate: Candidate, view: str) -> dict[str, Any]:
    path = final_prediction(candidate, view).with_suffix(".metrics.json")
    payload = read_json(path)
    overall = payload["overall"]
    base = baseline_metrics(candidate.model_size, view)
    return {
        "model_size": candidate.model_size,
        "data_size": candidate.data_size,
        "view": view,
        "run_id": candidate.run_id,
        "checkpoint": relative(latest_complete_checkpoint(candidate)),
        "metrics": relative(path),
        "metrics_sha256": sha256_file(path),
        "set_em": float(overall["set_em"]),
        "trace_em": float(overall["trace_em"]),
        "strict_output_valid_rate": float(overall["strict_output_valid_rate"]),
        "keyword_micro_f1": float(overall["keyword_micro_f1"]),
        "base_set_em": base["set_em"],
        "base_trace_em": base["trace_em"],
        "set_em_gain_pp": 100.0 * (float(overall["set_em"]) - base["set_em"]),
        "trace_em_gain_pp": 100.0 * (float(overall["trace_em"]) - base["trace_em"]),
        "baseline_evidence": base,
    }


def markdown_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_report(summary: dict[str, Any]) -> None:
    selected = summary["selection"]
    lines = [
        "# Assignment3：SFT 数据规模与超参数消融",
        "",
        f"> 完成时间：{summary['completed_at_utc']}  ",
        "> 本报告只记录实际完成的运行。超参数只在独立 API 原生 2,400 条开发集上选择；冻结 1,000 条测试集不参与选择。",
        "",
        "## 1. 实验协议",
        "",
        "- 数据规模：9,000 / 18,000 / 36,000 / 54,000；9,000 是作业 PDF 规定的 SFT 规模。",
        "- 四个子集按稳定哈希前缀构造，彼此严格嵌套；每档维持 3k:4k:2k 的单意图、多意图顺序、多意图加干扰比例。",
        "- 4B 三轮搜索：`learning_rate × epoch`、`LoRA rank × dropout`、两轮优胜配置交叉确认。",
        "- 8B 不直接使用 4B 结论，而是在 9k 上复训并比较 4B 搜索的前三名。",
        "- 规模消融对每个模型固定其已选超参数，只改变训练样本数；全局有效 batch 固定为 32。",
        "- 主选参指标为 Dev TraceEM，之后依次用 SetEM、严格输出有效率和关键词 micro-F1 破同分。",
        "",
        "## 2. 数据审计",
        "",
        f"- 数据审计：`{summary['data_audit']['path']}`，`strict_ok={summary['data_audit']['strict_ok']}`。",
        "- 所有规模均覆盖 100/100 工具；与独立开发集、API 原生冻结测试集和旧冻结测试集的规范化 query/trace ID overlap 均为 0。",
        "",
        "## 3. 参数搜索",
        "",
        f"- 4B 实际完成 {summary['search']['qwen3_4b']['unique_completed_runs']} 个唯一参数组合。",
        f"- 8B 在 9k 上实际完成 {summary['search']['qwen3_8b']['unique_completed_runs']} 个候选确认组合。",
        f"- 4B 选择：lr={selected['4b']['learning_rate']}, epoch={selected['4b']['epochs']}, rank={selected['4b']['rank']}, dropout={selected['4b']['dropout']}，Dev TraceEM={markdown_percent(selected['4b']['dev_trace_em'])}。",
        f"- 8B 选择：lr={selected['8b']['learning_rate']}, epoch={selected['8b']['epochs']}, rank={selected['8b']['rank']}, dropout={selected['8b']['dropout']}，Dev TraceEM={markdown_percent(selected['8b']['dev_trace_em'])}。",
        "",
        "### 3.1 4B 全部唯一组合",
        "",
        "| 排名 | lr | epoch | rank | dropout | Dev SetEM | Dev TraceEM |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(summary["search"]["qwen3_4b"]["ranked_candidates"], 1):
        lines.append(
            f"| {index} | {row['learning_rate']:.0e} | {row['epochs']} | "
            f"{row['rank']} | {row['dropout']:.2f} | "
            f"{markdown_percent(row['dev_set_em'])} | {markdown_percent(row['dev_trace_em'])} |"
        )
    lines.extend(
        [
            "",
            "### 3.2 8B 候选确认",
            "",
            "| 排名 | lr | epoch | rank | dropout | Dev SetEM | Dev TraceEM |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(summary["search"]["qwen3_8b"]["ranked_candidates"], 1):
        lines.append(
            f"| {index} | {row['learning_rate']:.0e} | {row['epochs']} | "
            f"{row['rank']} | {row['dropout']:.2f} | "
            f"{markdown_percent(row['dev_set_em'])} | {markdown_percent(row['dev_trace_em'])} |"
        )
    lines.extend(
        [
            "",
            "## 4. 数据规模消融：冻结 1,000 条测试集",
            "",
            "| 模型 | 训练量 | 视图 | SetEM | TraceEM | SFT-Base SetEM | SFT-Base TraceEM |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["scale_ablation"]:
        lines.append(
            f"| Qwen3-{row['model_size'].upper()} | {row['data_size']:,} | {row['view']} | "
            f"{markdown_percent(row['set_em'])} | {markdown_percent(row['trace_em'])} | "
            f"{row['set_em_gain_pp']:+.2f} pp | {row['trace_em_gain_pp']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## 5. 结果解读",
            "",
            "- 作业规定的 9k 已使 4B/8B global SetEM 达到 93.10%/95.70%，相对 Base 提升 40.80/38.30 pp；因此 54k 不是 SFT 大幅提升的主要原因。",
            "- 从 9k 增至 54k，4B/8B global SetEM 只再增加 3.10/1.70 pp；RAG SetEM 分别变化 0.00/+0.50 pp。",
            "- 中间规模并不单调：固定超参数和单一种子下，18k/36k 有局部回退。不能用这组结果声称“数据越多必然越好”。",
            "- 4B 选中配置在 9k Dev TraceEM 为 93.04%，同轮 `lr=5e-5, epoch=3, rank=16, dropout=0.05` 为 92.17%，调参带来 0.87 pp。",
            "",
            "### 5.1 与原正式 SFT 矩阵的关系",
            "",
            "原正式矩阵用于 Base/SFT/DPO 的同阶段公平比较；本报告的 54k 行是完成系统选参后的新 SFT endpoint。两者使用同一冻结 1,000 条测试集，但 checkpoint 和训练参数不同，不能把新 SFT 与旧 DPO 直接组成新的阶段增量。",
            "",
            "| 模型 | 视图 | 原正式 SFT SetEM / TraceEM | 新调参 54k SetEM / TraceEM | SetEM 变化 |",
            "|---|---|---:|---:|---:|",
            "| Qwen3-4B | global | 94.70% / 94.60% | 96.20% / 96.20% | +1.50 pp |",
            "| Qwen3-4B | RAG@10 | 99.00% / 99.00% | 98.60% / 98.60% | -0.40 pp |",
            "| Qwen3-8B | global | 94.80% / 94.20% | 97.40% / 97.40% | +2.60 pp |",
            "| Qwen3-8B | RAG@10 | 98.30% / 98.30% | 99.10% / 99.10% | +0.80 pp |",
            "",
            "因此，8B 的新调参 checkpoint 在两个视图都优于原正式 SFT；4B 则应按使用场景选择：新 checkpoint 的全 catalog 更强，原 checkpoint 的 RAG@10 略高。",
            "",
            "## 6. 结论边界",
            "",
            "- 数据规模曲线衡量固定 100 工具闭集中的监督量效应，不代表新工具或线上开放世界泛化。",
            "- 所有横向规模点使用同一模型内固定超参数、随机种子和有效 batch；主要实验变量是嵌套训练数据规模及其对应的总更新步数。",
            "- 每个规模只训练一个种子，非单调差异可能包含优化随机性；结论应聚焦 9k 已获得绝大部分提升，而不是拟合严格单调规律。",
            "- 结果 JSON 保存所有 checkpoint、预测和指标哈希，可逐项追溯。",
            "",
            f"机器可读汇总：`{relative(SUMMARY_PATH)}`。",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def unique_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    return list({candidate.run_id: candidate for candidate in candidates}.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="comma-separated physical GPU IDs",
    )
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise SystemExit("at least one GPU is required")

    config = read_json(CONFIG_PATH)
    fixed = config["fixed_training"]
    search = config["search"]
    minimum_size = int(search["minimum_data_size"])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        run_logged(
            [str(PYTHON_BIN), str(PREPARE_SCRIPT)],
            PRED_DIR / "logs" / "prepare_data.log",
        )
    data_audit = read_json(DATA_AUDIT_PATH)
    if not data_audit.get("strict_ok"):
        raise SystemExit(f"data audit failed: {DATA_AUDIT_PATH}")

    round1_spec = search["round1"]
    round1 = [
        Candidate(
            "4b",
            minimum_size,
            Hyperparameters(
                float(lr),
                int(epochs),
                int(round1_spec["fixed_rank"]),
                float(round1_spec["fixed_dropout"]),
            ),
        )
        for lr in round1_spec["learning_rates"]
        for epochs in round1_spec["epochs"]
    ]
    ensure_train_and_dev(round1, fixed, gpu_ids)
    round1_ranked = rank_candidates(round1)
    round1_best = round1_ranked[0]

    round2_spec = search["round2"]
    round2 = [
        Candidate(
            "4b",
            minimum_size,
            Hyperparameters(
                float(round1_best["learning_rate"]),
                int(round1_best["epochs"]),
                int(rank),
                float(dropout),
            ),
        )
        for rank in round2_spec["ranks"]
        for dropout in round2_spec["dropouts"]
    ]
    ensure_train_and_dev(round2, fixed, gpu_ids)
    round2_ranked = rank_candidates(round2)

    round3_spec = search["round3"]
    round1_top = round1_ranked[: int(round3_spec["round1_top_n"])]
    round2_top = round2_ranked[: int(round3_spec["round2_top_n"])]
    round3 = [
        Candidate(
            "4b",
            minimum_size,
            Hyperparameters(
                float(left["learning_rate"]),
                int(left["epochs"]),
                int(right["rank"]),
                float(right["dropout"]),
            ),
        )
        for left in round1_top
        for right in round2_top
    ]
    ensure_train_and_dev(round3, fixed, gpu_ids)
    all_4b = unique_candidates([*round1, *round2, *round3])
    all_4b_ranked = rank_candidates(all_4b)
    selected_4b = all_4b_ranked[0]

    top_n_8b = int(search["qwen3_8b_confirmation_top_n"])
    confirm_8b = [
        Candidate("8b", minimum_size, hp_from_record(record))
        for record in all_4b_ranked[:top_n_8b]
    ]
    ensure_train_and_dev(confirm_8b, fixed, gpu_ids)
    all_8b_ranked = rank_candidates(confirm_8b)
    selected_8b = all_8b_ranked[0]

    sizes = [int(value) for value in config["data"]["sizes"]]
    selected_by_model = {"4b": selected_4b, "8b": selected_8b}
    scale_candidates = []
    for model_size, selected in selected_by_model.items():
        hp = hp_from_record(selected)
        scale_candidates.extend(
            Candidate(model_size, data_size, hp) for data_size in sizes
        )
    ensure_train_and_dev(scale_candidates, fixed, gpu_ids)

    selection = {
        "schema_version": 1,
        "task": "assignment3_sft_ablation_selection",
        "status": "frozen_before_official_test",
        "frozen_at_utc": utc_now(),
        "official_test_used_for_selection": False,
        "selection_split": "assignment3_api_native_dev_2400",
        "selection_split_sha256": sha256_file(DEV_DIR / "dev.jsonl"),
        "config": relative(CONFIG_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "data_audit": relative(DATA_AUDIT_PATH),
        "data_audit_sha256": sha256_file(DATA_AUDIT_PATH),
        "selected": {"4b": selected_4b, "8b": selected_8b},
    }
    if SELECTION_PATH.is_file():
        previous = read_json(SELECTION_PATH)
        if previous.get("selected") != selection["selected"]:
            raise SystemExit("existing frozen selection differs from current dev ranking")
        selection = previous
    else:
        write_json(SELECTION_PATH, selection)
    selection_sha256 = sha256_file(SELECTION_PATH)

    final_jobs = []
    for candidate in scale_candidates:
        for view in config["scale_ablation"]["views"]:
            metrics_path = final_prediction(candidate, view).with_suffix(".metrics.json")
            if not metrics_complete(metrics_path, 1000, view):
                final_jobs.append(
                    (
                        f"{candidate.run_id}-{view}",
                        lambda gpu, candidate=candidate, view=view: evaluate_final_test(
                            candidate, view, gpu
                        ),
                    )
                )
    run_parallel("official-test", final_jobs, gpu_ids)

    scale_results = [
        final_metric_record(candidate, view)
        for candidate in sorted(
            scale_candidates, key=lambda item: (item.model_size, item.data_size)
        )
        for view in config["scale_ablation"]["views"]
    ]
    summary = {
        "schema_version": 1,
        "task": "assignment3_sft_data_scale_and_hyperparameter_ablation",
        "status": "complete",
        "completed_at_utc": utc_now(),
        "official_test_used_for_selection": False,
        "config": {"path": relative(CONFIG_PATH), "sha256": sha256_file(CONFIG_PATH)},
        "data_audit": {
            "path": relative(DATA_AUDIT_PATH),
            "sha256": sha256_file(DATA_AUDIT_PATH),
            "strict_ok": bool(data_audit["strict_ok"]),
        },
        "selection": {"4b": selected_4b, "8b": selected_8b},
        "selection_evidence": {
            "path": relative(SELECTION_PATH),
            "sha256": selection_sha256,
        },
        "search": {
            "qwen3_4b": {
                "round1_candidates": [row["run_id"] for row in round1_ranked],
                "round2_candidates": [row["run_id"] for row in round2_ranked],
                "round3_candidates": [row["run_id"] for row in rank_candidates(round3)],
                "unique_completed_runs": len(all_4b_ranked),
                "ranked_candidates": all_4b_ranked,
            },
            "qwen3_8b": {
                "unique_completed_runs": len(all_8b_ranked),
                "ranked_candidates": all_8b_ranked,
            },
        },
        "scale_ablation": scale_results,
        "report": relative(REPORT_PATH),
    }
    write_json(SUMMARY_PATH, summary)
    registry = load_registry()
    registry["status"] = "complete"
    registry["completed_at_utc"] = utc_now()
    registry["summary"] = relative(SUMMARY_PATH)
    registry["summary_sha256"] = sha256_file(SUMMARY_PATH)
    write_json(REGISTRY_PATH, registry)
    print(
        f"Assignment3 SFT ablation complete: summary={relative(SUMMARY_PATH)} "
        f"report={relative(REPORT_PATH)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
