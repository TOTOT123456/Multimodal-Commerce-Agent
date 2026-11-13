#!/usr/bin/env python3
"""运行、检查或删除多轮工具规划会话。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from PROJECT.agent.factory import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_DB,
    RuntimeDevices,
    build_agent,
    load_agent_config,
)
from PROJECT.agent.memory import SQLiteMemoryStore  # noqa: E402


def emit(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def runtime_store(args: argparse.Namespace) -> SQLiteMemoryStore:
    config = load_agent_config(args.config)
    return SQLiteMemoryStore(
        args.db,
        ttl_seconds=int(config["memory"]["session_ttl_seconds"]),
    )


def run_command(args: argparse.Namespace) -> int:
    devices = RuntimeDevices(
        intent=args.intent_device,
        embedding=args.embedding_device,
        reranker=args.reranker_device,
        planner=args.planner_device,
    )
    agent = build_agent(
        config_path=args.config,
        db_path=args.db,
        devices=devices,
        intent_base_model=args.intent_base_model or None,
        planner_base_model=args.planner_base_model or None,
    )
    try:
        if args.interactive:
            for line in sys.stdin:
                query = line.strip()
                if not query:
                    continue
                if query in {"/quit", "/exit"}:
                    break
                emit(agent.run_turn(args.session, query).to_dict())
            return 0
        result = agent.run_turn(args.session, args.query)
        emit(result.to_dict())
        return 0 if result.status in {"ok", "chitchat"} else 1
    finally:
        agent.close()


def inspect_command(args: argparse.Namespace) -> int:
    store = runtime_store(args)
    try:
        if not store.has_session(args.session):
            emit({"conversation_id": args.session, "found": False})
            return 1
        snapshot = store.snapshot(args.session, recent_limit=args.recent_limit)
        payload = {
            "found": True,
            "state": snapshot.state.to_dict(),
            "summary": snapshot.summary.to_dict(),
            "recent_events": [event.to_dict() for event in snapshot.recent_events],
            "long_memory": [record.to_dict() for record in store.memory_records(args.session)],
        }
        emit(payload)
        return 0
    finally:
        store.close()


def delete_command(args: argparse.Namespace) -> int:
    store = runtime_store(args)
    try:
        deleted = store.delete_session(args.session)
        emit({"conversation_id": args.session, "deleted": deleted})
        return 0
    finally:
        store.close()


def purge_command(args: argparse.Namespace) -> int:
    store = runtime_store(args)
    try:
        count = store.purge_expired()
        emit({"expired_sessions_deleted": count})
        return 0
    finally:
        store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="执行单轮或持续读取 stdin")
    run.add_argument("--session", required=True)
    run.add_argument("--query", default="")
    run.add_argument("--interactive", action="store_true")
    run.add_argument("--intent-device", default="cuda:0")
    run.add_argument("--embedding-device", default="cuda:0")
    run.add_argument("--reranker-device", default="cuda:1")
    run.add_argument("--planner-device", default="cuda:2")
    run.add_argument("--intent-base-model", default="")
    run.add_argument("--planner-base-model", default="")
    run.set_defaults(handler=run_command)

    inspect = commands.add_parser("inspect", help="查看一个会话的结构化状态")
    inspect.add_argument("--session", required=True)
    inspect.add_argument("--recent-limit", type=int, default=4)
    inspect.set_defaults(handler=inspect_command)

    delete = commands.add_parser("delete", help="删除一个会话及其事件和记忆")
    delete.add_argument("--session", required=True)
    delete.set_defaults(handler=delete_command)

    purge = commands.add_parser("purge-expired", help="清理 TTL 已到期的会话")
    purge.set_defaults(handler=purge_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run" and not args.interactive and not args.query.strip():
        parser.error("run requires --query unless --interactive is set")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
