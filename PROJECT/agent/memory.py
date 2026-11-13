"""SQLite 会话存储、确定性滚动摘要与会话内长期检索。"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .types import (
    CompactToolState,
    Goal,
    MemoryHit,
    MemoryRecord,
    PlanEvent,
    SessionSnapshot,
    ToolRollingSummary,
    unique_goals,
    utc_now,
)
from .reference_selection import (
    is_implicit_confirmation,
    reference_search_text,
    structural_reference_turn,
    temporally_rerank,
)


class ConcurrentTurnError(RuntimeError):
    """同一会话被另一个调用抢先提交。"""


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def expiry_from_now(ttl_seconds: int) -> str | None:
    if ttl_seconds <= 0:
        return None
    return (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()


class SQLiteMemoryStore:
    """单库持久化事件、紧凑状态和长期记忆，所有读取都带 conversation_id。"""

    def __init__(self, path: str | Path, ttl_seconds: int = 30 * 24 * 3600):
        self.path = str(path)
        self.ttl_seconds = int(ttl_seconds)
        if self.ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,
        )
        if self.path != ":memory:":
            os.chmod(self.path, 0o600)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock:
            schema_version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            if schema_version not in {0, 1}:
                raise RuntimeError(f"unsupported Agent memory schema: {schema_version}")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    conversation_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT
                );

                CREATE TABLE IF NOT EXISTS events (
                    conversation_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, turn_id),
                    FOREIGN KEY (conversation_id) REFERENCES sessions(conversation_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memories (
                    conversation_id TEXT NOT NULL,
                    source_turn_id INTEGER NOT NULL,
                    memory_json TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (conversation_id, source_turn_id),
                    FOREIGN KEY (conversation_id) REFERENCES sessions(conversation_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_events_session_turn
                    ON events(conversation_id, turn_id DESC);
                CREATE INDEX IF NOT EXISTS idx_memories_session_active
                    ON memories(conversation_id, deleted, source_turn_id DESC);
                """
            )
            self._connection.execute("PRAGMA user_version = 1")

    def _ensure_session(self, conversation_id: str) -> sqlite3.Row:
        conversation_id = conversation_id.strip()
        if not conversation_id:
            raise ValueError("conversation id must not be empty")
        row = self._connection.execute(
            "SELECT * FROM sessions WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        now = utc_now()
        if row is not None and row["expires_at"] and row["expires_at"] <= now:
            self._connection.execute(
                "DELETE FROM sessions WHERE conversation_id = ?", (conversation_id,)
            )
            row = None
        if row is None:
            state = CompactToolState(conversation_id=conversation_id)
            summary = ToolRollingSummary(conversation_id=conversation_id)
            self._connection.execute(
                """
                INSERT INTO sessions(
                    conversation_id, version, state_json, summary_json,
                    created_at, updated_at, expires_at
                ) VALUES (?, 0, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    json_dumps(state.to_dict()),
                    json_dumps(summary.to_dict()),
                    now,
                    now,
                    expiry_from_now(self.ttl_seconds),
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM sessions WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
        assert row is not None
        return row

    def snapshot(self, conversation_id: str, recent_limit: int) -> SessionSnapshot:
        if recent_limit <= 0:
            raise ValueError("recent_limit must be positive")
        with self._lock:
            row = self._ensure_session(conversation_id)
            events = self._connection.execute(
                """
                SELECT event_json FROM events
                WHERE conversation_id = ?
                ORDER BY turn_id DESC LIMIT ?
                """,
                (conversation_id, recent_limit),
            ).fetchall()
            recent = [PlanEvent.from_dict(json.loads(value["event_json"])) for value in events]
            recent.reverse()
            return SessionSnapshot(
                state=CompactToolState.from_dict(json.loads(row["state_json"])),
                summary=ToolRollingSummary.from_dict(json.loads(row["summary_json"])),
                recent_events=recent,
            )

    def has_session(self, conversation_id: str) -> bool:
        conversation_id = conversation_id.strip()
        if not conversation_id:
            return False
        now = utc_now()
        with self._lock:
            row = self._connection.execute(
                """
                SELECT 1 FROM sessions
                WHERE conversation_id = ?
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (conversation_id, now),
            ).fetchone()
            return row is not None

    def event(self, conversation_id: str, turn_id: int) -> PlanEvent | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT event_json FROM events
                WHERE conversation_id = ? AND turn_id = ?
                """,
                (conversation_id, int(turn_id)),
            ).fetchone()
            return PlanEvent.from_dict(json.loads(row["event_json"])) if row else None

    def all_events(self, conversation_id: str) -> list[PlanEvent]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT event_json FROM events
                WHERE conversation_id = ? ORDER BY turn_id
                """,
                (conversation_id,),
            ).fetchall()
            return [PlanEvent.from_dict(json.loads(row["event_json"])) for row in rows]

    def memory_records(self, conversation_id: str) -> list[MemoryRecord]:
        now = utc_now()
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT memory_json FROM memories
                WHERE conversation_id = ? AND deleted = 0
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY source_turn_id DESC
                """,
                (conversation_id, now),
            ).fetchall()
            return [MemoryRecord.from_dict(json.loads(row["memory_json"])) for row in rows]

    def commit_turn(
        self,
        event: PlanEvent,
        state: CompactToolState,
        summary: ToolRollingSummary,
        new_memories: list[MemoryRecord],
        expected_version: int,
    ) -> None:
        if state.version != expected_version + 1:
            raise ValueError("new state version must be expected_version + 1")
        if event.conversation_id != state.conversation_id:
            raise ValueError("event and state conversation ids differ")
        if summary.conversation_id != state.conversation_id:
            raise ValueError("summary and state conversation ids differ")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._ensure_session(event.conversation_id)
                actual_version = int(row["version"])
                if actual_version != expected_version:
                    raise ConcurrentTurnError(
                        f"session {event.conversation_id!r} changed: "
                        f"expected version {expected_version}, found {actual_version}"
                    )
                stored_state = CompactToolState.from_dict(json.loads(row["state_json"]))
                if event.turn_id != stored_state.last_turn_id + 1:
                    raise ConcurrentTurnError(
                        f"session {event.conversation_id!r} turn changed: "
                        f"expected {stored_state.last_turn_id + 1}, got {event.turn_id}"
                    )
                self._connection.execute(
                    """
                    INSERT INTO events(conversation_id, turn_id, event_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        event.conversation_id,
                        event.turn_id,
                        json_dumps(event.to_dict()),
                        event.created_at,
                    ),
                )
                for memory in new_memories:
                    if memory.conversation_id != event.conversation_id:
                        raise ValueError("memory crossed the current conversation boundary")
                    self._connection.execute(
                        """
                        INSERT INTO memories(
                            conversation_id, source_turn_id, memory_json, search_text,
                            created_at, expires_at, deleted
                        ) VALUES (?, ?, ?, ?, ?, ?, 0)
                        ON CONFLICT(conversation_id, source_turn_id) DO UPDATE SET
                            memory_json = excluded.memory_json,
                            search_text = excluded.search_text,
                            expires_at = excluded.expires_at,
                            deleted = 0
                        """,
                        (
                            memory.conversation_id,
                            memory.source_turn_id,
                            json_dumps(memory.to_dict()),
                            memory.text,
                            memory.created_at,
                            memory.expires_at,
                        ),
                    )
                now = utc_now()
                cursor = self._connection.execute(
                    """
                    UPDATE sessions SET
                        version = ?, state_json = ?, summary_json = ?,
                        updated_at = ?, expires_at = ?
                    WHERE conversation_id = ? AND version = ?
                    """,
                    (
                        state.version,
                        json_dumps(state.to_dict()),
                        json_dumps(summary.to_dict()),
                        now,
                        expiry_from_now(self.ttl_seconds),
                        event.conversation_id,
                        expected_version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ConcurrentTurnError("session compare-and-swap failed")
                self._connection.execute("COMMIT")
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def delete_session(self, conversation_id: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM sessions WHERE conversation_id = ?", (conversation_id,)
            )
            return cursor.rowcount > 0

    def purge_expired(self) -> int:
        now = utc_now()
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM sessions WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,)
            )
            return cursor.rowcount

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class DeterministicMemoryReducer:
    """把滑出近期窗口的事件变成结构化摘要和独立长期记录。"""

    def __init__(
        self,
        catalog_by_id: dict[str, dict[str, Any]],
        recent_event_limit: int,
        max_summary_goals: int = 20,
        memory_ttl_seconds: int = 30 * 24 * 3600,
    ):
        if recent_event_limit <= 0 or max_summary_goals <= 0:
            raise ValueError("memory reducer limits must be positive")
        self.catalog_by_id = catalog_by_id
        self.recent_event_limit = recent_event_limit
        self.max_summary_goals = max_summary_goals
        self.memory_ttl_seconds = memory_ttl_seconds

    def reduce(
        self,
        previous: ToolRollingSummary,
        all_recent_events: list[PlanEvent],
        newest_turn_id: int,
    ) -> tuple[ToolRollingSummary, list[MemoryRecord]]:
        compact_through = newest_turn_id - self.recent_event_limit
        candidates = [
            event
            for event in all_recent_events
            if previous.covered_through_turn < event.turn_id <= compact_through
        ]
        if not candidates:
            return previous, []

        historical = list(previous.historical_goals)
        cancellations = list(previous.cancelled_goals)
        memories: list[MemoryRecord] = []
        for event in candidates:
            if event.intent_type == "闲聊类":
                continue
            historical.extend(event.resolved_goals)
            cancellations.extend(event.cancelled_goals)
            if not event.resolved_goals:
                continue
            domains = list(
                dict.fromkeys(
                    str(self.catalog_by_id[tool]["domain"])
                    for tool in event.planned_tools
                    if tool in self.catalog_by_id
                )
            )
            memories.append(
                MemoryRecord(
                    conversation_id=event.conversation_id,
                    source_turn_id=event.turn_id,
                    text=event.resolved_query,
                    goals=event.resolved_goals,
                    planned_tools=event.planned_tools,
                    domains=domains,
                    created_at=event.created_at,
                    expires_at=expiry_from_now(self.memory_ttl_seconds),
                )
            )

        historical = unique_goals(historical)[-self.max_summary_goals :]
        cancellations = unique_goals(cancellations)[-self.max_summary_goals :]
        summary = ToolRollingSummary(
            conversation_id=previous.conversation_id,
            version=previous.version + 1,
            covered_through_turn=max(event.turn_id for event in candidates),
            historical_goals=historical,
            cancelled_goals=cancellations,
            updated_at=utc_now(),
        )
        return summary, memories


def fallback_tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    for block in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", block):
            tokens.extend(block)
            tokens.extend(block[index : index + 2] for index in range(max(0, len(block) - 1)))
            tokens.extend(block[index : index + 3] for index in range(max(0, len(block) - 2)))
        else:
            tokens.append(block)
            tokens.extend(part for part in block.split("_") if part)
    return tokens


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    tokenized = [fallback_tokenize(value) for value in documents]
    query_tokens = fallback_tokenize(query)
    count = len(tokenized)
    lengths = [len(value) for value in tokenized]
    average = sum(lengths) / max(1, count)
    frequencies = [Counter(value) for value in tokenized]
    document_frequency: Counter[str] = Counter()
    for value in tokenized:
        document_frequency.update(set(value))
    inverse = {
        term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
    }
    result = []
    for length, frequency in zip(lengths, frequencies):
        score = 0.0
        for term in query_tokens:
            term_frequency = frequency.get(term, 0)
            if not term_frequency:
                continue
            denominator = term_frequency + k1 * (1.0 - b + b * length / max(1.0, average))
            score += inverse.get(term, 0.0) * term_frequency * (k1 + 1.0) / denominator
        result.append(score)
    return result


class LexicalLongTermMemory:
    """先会话隔离，再精确 turn 或 BM25；零词面证据时不返回历史。"""

    def __init__(self, store: SQLiteMemoryStore, reference_strategy: str = "off"):
        if reference_strategy not in {
            "off",
            "raw_bm25",
            "anchor_bm25",
            "anchor_bm25_position",
        }:
            raise ValueError("unknown semantic reference strategy")
        self.store = store
        self.reference_strategy = reference_strategy

    def _event_records(self, conversation_id: str) -> list[MemoryRecord]:
        records = []
        for event in self.store.all_events(conversation_id):
            if not event.resolved_goals:
                continue
            text = event.raw_query
            if event.resolved_query and event.resolved_query != event.raw_query:
                text = f"{event.raw_query}；{event.resolved_query}"
            records.append(
                MemoryRecord(
                    conversation_id=conversation_id,
                    source_turn_id=event.turn_id,
                    text=text,
                    goals=event.resolved_goals,
                    planned_tools=event.planned_tools,
                    domains=[],
                    created_at=event.created_at,
                )
            )
        return records

    def search(
        self,
        conversation_id: str,
        query: str,
        top_k: int,
        exact_turn_ids: list[int] | None = None,
    ) -> list[MemoryHit]:
        if not 1 <= top_k <= 3:
            raise ValueError("long-term memory top_k must be in [1, 3]")
        records = (
            self.store.memory_records(conversation_id)
            if self.reference_strategy == "off"
            else self._event_records(conversation_id)
        )
        if exact_turn_ids:
            by_turn = {record.source_turn_id: record for record in records}
            return [
                MemoryHit(record=by_turn[turn_id], score=1.0, match_reason="exact_turn")
                for turn_id in exact_turn_ids
                if turn_id in by_turn
            ][:top_k]
        if not records:
            return []
        if self.reference_strategy != "off" and is_implicit_confirmation(query):
            latest = max(records, key=lambda record: record.source_turn_id)
            return [
                MemoryHit(
                    record=latest,
                    score=1.0,
                    match_reason="implicit_previous_turn",
                )
            ]
        if self.reference_strategy == "off" and any(
            signal in query for signal in ("最开始", "一开始", "最早")
        ):
            earliest = min(records, key=lambda record: record.source_turn_id)
            return [MemoryHit(record=earliest, score=1.0, match_reason="earliest_turn")]
        search_query = (
            reference_search_text(query)
            if self.reference_strategy in {"anchor_bm25", "anchor_bm25_position"}
            else query
        )
        documents = [(record.source_turn_id, record.text) for record in records]
        scores = bm25_scores(search_query, [text for _, text in documents])
        if self.reference_strategy == "anchor_bm25_position":
            structural = structural_reference_turn(query, documents, bm25_scores)
        else:
            structural = None
        ranked_indices = (
            temporally_rerank(query, documents, scores)
            if self.reference_strategy in {"anchor_bm25", "anchor_bm25_position"}
            else sorted(
                range(len(records)),
                key=lambda index: (-scores[index], -records[index].source_turn_id),
            )
        )
        if structural is not None:
            structural_index = next(
                (
                    index
                    for index, record in enumerate(records)
                    if record.source_turn_id == structural
                ),
                None,
            )
            if structural_index is not None:
                ranked_indices = [
                    structural_index,
                    *(index for index in ranked_indices if index != structural_index),
                ]
        return [
            MemoryHit(
                record=records[index],
                score=scores[index],
                match_reason=(
                    "structural_position"
                    if structural is not None
                    and records[index].source_turn_id == structural
                    else self.reference_strategy
                ),
            )
            for index in ranked_indices
            if scores[index] > 0.0 or records[index].source_turn_id == structural
        ][:top_k]
