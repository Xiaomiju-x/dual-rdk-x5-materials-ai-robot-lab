"""SQLite-backed sparse episodic scene graph.

The graph stores observation metadata only. It has no ROS publisher, serial
interface, actuator API, online learner, or device-control authority.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NODE_KINDS = frozenset({"area", "workstation", "object", "task_event"})
EDGE_RELATIONS = frozenset(
    {"in", "contains", "connected", "reachable", "observed_during"}
)
SCHEMA_VERSION = "x5-episodic-scenegraph/1.0"


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _json_value(value: Any, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            normalized[key] = _json_value(item, f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains unsupported type {type(value).__name__}")


def _json_object(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized = _json_value(value, name)
    assert isinstance(normalized, dict)
    return normalized


def _encode_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _open_database(path: str | Path) -> sqlite3.Connection:
    database = str(path)
    if database != ":memory:":
        target = Path(database).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        database = str(target)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if database != ":memory:":
        connection.execute("PRAGMA journal_mode = WAL")
    return connection


@dataclass(frozen=True, slots=True)
class Pose:
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    frame_id: str = "map"

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_number(self.x, "pose.x"))
        object.__setattr__(self, "y", _finite_number(self.y, "pose.y"))
        object.__setattr__(self, "z", _finite_number(self.z, "pose.z"))
        object.__setattr__(self, "yaw", _finite_number(self.yaw, "pose.yaw"))
        object.__setattr__(
            self,
            "frame_id",
            _required_text(self.frame_id, "pose.frame_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "frame_id": self.frame_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Pose:
        return cls(
            x=value["x"],
            y=value["y"],
            z=value.get("z", 0.0),
            yaw=value.get("yaw", 0.0),
            frame_id=value.get("frame_id", "map"),
        )


@dataclass(frozen=True, slots=True)
class NodeRecord:
    node_id: str
    kind: str
    pose: Pose | None
    source: str
    confidence: float
    timestamp: float
    ttl: float | None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    properties: Mapping[str, Any] = field(default_factory=dict)

    @property
    def expires_at(self) -> float | None:
        return None if self.ttl is None else self.timestamp + self.ttl

    def is_active(self, at: float) -> bool:
        return self.expires_at is None or at < self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "pose": None if self.pose is None else self.pose.to_dict(),
            "source": self.source,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "expires_at": self.expires_at,
            "provenance": dict(self.provenance),
            "properties": dict(self.properties),
        }


@dataclass(frozen=True, slots=True)
class EdgeRecord:
    edge_id: str
    source_node: str
    relation: str
    target_node: str
    source: str
    confidence: float
    timestamp: float
    ttl: float | None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    properties: Mapping[str, Any] = field(default_factory=dict)

    @property
    def expires_at(self) -> float | None:
        return None if self.ttl is None else self.timestamp + self.ttl

    def is_active(self, at: float) -> bool:
        return self.expires_at is None or at < self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node": self.source_node,
            "relation": self.relation,
            "target_node": self.target_node,
            "source": self.source,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "expires_at": self.expires_at,
            "provenance": dict(self.provenance),
            "properties": dict(self.properties),
        }


@dataclass(frozen=True, slots=True)
class GraphNeighbor:
    node: NodeRecord
    edge: EdgeRecord
    direction: str


@dataclass(frozen=True, slots=True)
class TraversalHit:
    node: NodeRecord
    depth: int
    via_edge_id: str | None


class SceneGraph:
    """Persistent sparse graph with temporal and provenance-aware queries."""

    def __init__(
        self,
        database: str | Path = ":memory:",
        *,
        clock: Any = time.time,
    ) -> None:
        self._connection = _open_database(database)
        self._clock = clock
        self._lock = threading.RLock()
        self._create_schema()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS scene_nodes (
                    node_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL CHECK (
                        kind IN ('area', 'workstation', 'object', 'task_event')
                    ),
                    pose_json TEXT,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK (
                        confidence >= 0.0 AND confidence <= 1.0
                    ),
                    observed_at REAL NOT NULL,
                    ttl_s REAL CHECK (ttl_s IS NULL OR ttl_s > 0.0),
                    provenance_json TEXT NOT NULL,
                    properties_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS scene_edges (
                    edge_id TEXT PRIMARY KEY,
                    source_node TEXT NOT NULL REFERENCES scene_nodes(node_id)
                        ON DELETE CASCADE,
                    relation TEXT NOT NULL CHECK (
                        relation IN (
                            'in', 'contains', 'connected', 'reachable',
                            'observed_during'
                        )
                    ),
                    target_node TEXT NOT NULL REFERENCES scene_nodes(node_id)
                        ON DELETE CASCADE,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK (
                        confidence >= 0.0 AND confidence <= 1.0
                    ),
                    observed_at REAL NOT NULL,
                    ttl_s REAL CHECK (ttl_s IS NULL OR ttl_s > 0.0),
                    provenance_json TEXT NOT NULL,
                    properties_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_scene_nodes_kind_source
                    ON scene_nodes(kind, source);
                CREATE INDEX IF NOT EXISTS idx_scene_edges_out
                    ON scene_edges(source_node, relation);
                CREATE INDEX IF NOT EXISTS idx_scene_edges_in
                    ON scene_edges(target_node, relation);
                """
            )
            self._connection.execute(
                """
                INSERT INTO memory_metadata(key, value)
                VALUES('scene_graph_schema', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (SCHEMA_VERSION,),
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SceneGraph:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def upsert_node(
        self,
        node_id: str,
        kind: str,
        *,
        pose: Pose | None,
        source: str,
        confidence: float,
        timestamp: float | None = None,
        ttl: float | None = None,
        provenance: Mapping[str, Any] | None = None,
        properties: Mapping[str, Any] | None = None,
    ) -> NodeRecord:
        node_id = _required_text(node_id, "node_id")
        kind = _required_text(kind, "kind")
        if kind not in NODE_KINDS:
            raise ValueError(f"unsupported node kind: {kind}")
        if pose is not None and not isinstance(pose, Pose):
            raise TypeError("pose must be Pose or None")
        source = _required_text(source, "source")
        confidence = _finite_number(confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        observed_at = _finite_number(
            self._clock() if timestamp is None else timestamp,
            "timestamp",
        )
        ttl_s = None if ttl is None else _finite_number(ttl, "ttl")
        if ttl_s is not None and ttl_s <= 0.0:
            raise ValueError("ttl must be positive")
        provenance_value = _json_object(provenance, "provenance")
        properties_value = _json_object(properties, "properties")

        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO scene_nodes(
                    node_id, kind, pose_json, source, confidence, observed_at,
                    ttl_s, provenance_json, properties_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    kind = excluded.kind,
                    pose_json = excluded.pose_json,
                    source = excluded.source,
                    confidence = excluded.confidence,
                    observed_at = excluded.observed_at,
                    ttl_s = excluded.ttl_s,
                    provenance_json = excluded.provenance_json,
                    properties_json = excluded.properties_json
                """,
                (
                    node_id,
                    kind,
                    None if pose is None else _encode_json(pose.to_dict()),
                    source,
                    confidence,
                    observed_at,
                    ttl_s,
                    _encode_json(provenance_value),
                    _encode_json(properties_value),
                ),
            )
        record = self.get_node(node_id, include_expired=True)
        assert record is not None
        return record

    def upsert_edge(
        self,
        edge_id: str,
        source_node: str,
        relation: str,
        target_node: str,
        *,
        source: str,
        confidence: float,
        timestamp: float | None = None,
        ttl: float | None = None,
        provenance: Mapping[str, Any] | None = None,
        properties: Mapping[str, Any] | None = None,
    ) -> EdgeRecord:
        edge_id = _required_text(edge_id, "edge_id")
        source_node = _required_text(source_node, "source_node")
        target_node = _required_text(target_node, "target_node")
        relation = _required_text(relation, "relation")
        if relation not in EDGE_RELATIONS:
            raise ValueError(f"unsupported edge relation: {relation}")
        source = _required_text(source, "source")
        confidence = _finite_number(confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        observed_at = _finite_number(
            self._clock() if timestamp is None else timestamp,
            "timestamp",
        )
        ttl_s = None if ttl is None else _finite_number(ttl, "ttl")
        if ttl_s is not None and ttl_s <= 0.0:
            raise ValueError("ttl must be positive")
        provenance_value = _json_object(provenance, "provenance")
        properties_value = _json_object(properties, "properties")

        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO scene_edges(
                    edge_id, source_node, relation, target_node, source,
                    confidence, observed_at, ttl_s, provenance_json,
                    properties_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(edge_id) DO UPDATE SET
                    source_node = excluded.source_node,
                    relation = excluded.relation,
                    target_node = excluded.target_node,
                    source = excluded.source,
                    confidence = excluded.confidence,
                    observed_at = excluded.observed_at,
                    ttl_s = excluded.ttl_s,
                    provenance_json = excluded.provenance_json,
                    properties_json = excluded.properties_json
                """,
                (
                    edge_id,
                    source_node,
                    relation,
                    target_node,
                    source,
                    confidence,
                    observed_at,
                    ttl_s,
                    _encode_json(provenance_value),
                    _encode_json(properties_value),
                ),
            )
        record = self.get_edge(edge_id, include_expired=True)
        assert record is not None
        return record

    @staticmethod
    def _node_from_row(row: sqlite3.Row) -> NodeRecord:
        pose_value = json.loads(row["pose_json"]) if row["pose_json"] else None
        return NodeRecord(
            node_id=row["node_id"],
            kind=row["kind"],
            pose=None if pose_value is None else Pose.from_dict(pose_value),
            source=row["source"],
            confidence=float(row["confidence"]),
            timestamp=float(row["observed_at"]),
            ttl=None if row["ttl_s"] is None else float(row["ttl_s"]),
            provenance=json.loads(row["provenance_json"]),
            properties=json.loads(row["properties_json"]),
        )

    @staticmethod
    def _edge_from_row(row: sqlite3.Row) -> EdgeRecord:
        return EdgeRecord(
            edge_id=row["edge_id"],
            source_node=row["source_node"],
            relation=row["relation"],
            target_node=row["target_node"],
            source=row["source"],
            confidence=float(row["confidence"]),
            timestamp=float(row["observed_at"]),
            ttl=None if row["ttl_s"] is None else float(row["ttl_s"]),
            provenance=json.loads(row["provenance_json"]),
            properties=json.loads(row["properties_json"]),
        )

    def get_node(
        self,
        node_id: str,
        *,
        as_of: float | None = None,
        include_expired: bool = False,
    ) -> NodeRecord | None:
        query = "SELECT * FROM scene_nodes WHERE node_id = ?"
        params: list[Any] = [_required_text(node_id, "node_id")]
        if not include_expired:
            at = _finite_number(self._clock() if as_of is None else as_of, "as_of")
            query += " AND (ttl_s IS NULL OR observed_at + ttl_s > ?)"
            params.append(at)
        with self._lock:
            row = self._connection.execute(query, params).fetchone()
        return None if row is None else self._node_from_row(row)

    def get_edge(
        self,
        edge_id: str,
        *,
        as_of: float | None = None,
        include_expired: bool = False,
    ) -> EdgeRecord | None:
        query = "SELECT * FROM scene_edges WHERE edge_id = ?"
        params: list[Any] = [_required_text(edge_id, "edge_id")]
        if not include_expired:
            at = _finite_number(self._clock() if as_of is None else as_of, "as_of")
            query += " AND (ttl_s IS NULL OR observed_at + ttl_s > ?)"
            params.append(at)
        with self._lock:
            row = self._connection.execute(query, params).fetchone()
        return None if row is None else self._edge_from_row(row)

    def find_nodes(
        self,
        *,
        kinds: Iterable[str] | None = None,
        source: str | None = None,
        min_confidence: float = 0.0,
        as_of: float | None = None,
        include_expired: bool = False,
    ) -> list[NodeRecord]:
        minimum = _finite_number(min_confidence, "min_confidence")
        if not 0.0 <= minimum <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        clauses = ["confidence >= ?"]
        params: list[Any] = [minimum]
        selected_kinds = tuple(sorted(set(kinds or ())))
        if selected_kinds:
            invalid = set(selected_kinds) - NODE_KINDS
            if invalid:
                raise ValueError(f"unsupported node kind(s): {sorted(invalid)}")
            placeholders = ",".join("?" for _ in selected_kinds)
            clauses.append(f"kind IN ({placeholders})")
            params.extend(selected_kinds)
        if source is not None:
            clauses.append("source = ?")
            params.append(_required_text(source, "source"))
        if not include_expired:
            at = _finite_number(self._clock() if as_of is None else as_of, "as_of")
            clauses.append("(ttl_s IS NULL OR observed_at + ttl_s > ?)")
            params.append(at)
        query = "SELECT * FROM scene_nodes WHERE " + " AND ".join(clauses)
        query += " ORDER BY observed_at DESC, node_id"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [self._node_from_row(row) for row in rows]

    def neighbors(
        self,
        node_id: str,
        *,
        relations: Iterable[str] | None = None,
        direction: str = "out",
        source: str | None = None,
        as_of: float | None = None,
        include_expired: bool = False,
    ) -> list[GraphNeighbor]:
        node_id = _required_text(node_id, "node_id")
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be out, in, or both")
        selected_relations = tuple(sorted(set(relations or ())))
        invalid = set(selected_relations) - EDGE_RELATIONS
        if invalid:
            raise ValueError(f"unsupported edge relation(s): {sorted(invalid)}")
        at = _finite_number(self._clock() if as_of is None else as_of, "as_of")
        candidates: list[tuple[EdgeRecord, str, str]] = []

        with self._lock:
            for selected_direction in (
                ("out", "source_node", "target_node"),
                ("in", "target_node", "source_node"),
            ):
                label, match_column, neighbor_column = selected_direction
                if direction not in {label, "both"}:
                    continue
                clauses = [f"{match_column} = ?"]
                params: list[Any] = [node_id]
                if selected_relations:
                    placeholders = ",".join("?" for _ in selected_relations)
                    clauses.append(f"relation IN ({placeholders})")
                    params.extend(selected_relations)
                if source is not None:
                    clauses.append("source = ?")
                    params.append(_required_text(source, "source"))
                if not include_expired:
                    clauses.append("(ttl_s IS NULL OR observed_at + ttl_s > ?)")
                    params.append(at)
                rows = self._connection.execute(
                    "SELECT * FROM scene_edges WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY edge_id",
                    params,
                ).fetchall()
                for row in rows:
                    candidates.append(
                        (self._edge_from_row(row), row[neighbor_column], label)
                    )

        result: list[GraphNeighbor] = []
        for edge, neighbor_id, selected_direction in candidates:
            neighbor = self.get_node(
                neighbor_id,
                as_of=at,
                include_expired=include_expired,
            )
            if neighbor is not None:
                result.append(
                    GraphNeighbor(
                        node=neighbor,
                        edge=edge,
                        direction=selected_direction,
                    )
                )
        return result

    def traverse(
        self,
        start_node: str,
        *,
        relations: Sequence[str] | None = None,
        direction: str = "out",
        max_depth: int = 3,
        source: str | None = None,
        as_of: float | None = None,
    ) -> list[TraversalHit]:
        if isinstance(max_depth, bool) or not isinstance(max_depth, int):
            raise TypeError("max_depth must be an integer")
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        at = _finite_number(self._clock() if as_of is None else as_of, "as_of")
        start = self.get_node(start_node, as_of=at)
        if start is None:
            return []
        hits = [TraversalHit(node=start, depth=0, via_edge_id=None)]
        visited = {start.node_id}
        frontier = [(start.node_id, 0)]
        while frontier:
            current_id, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for neighbor in self.neighbors(
                current_id,
                relations=relations,
                direction=direction,
                source=source,
                as_of=at,
            ):
                if neighbor.node.node_id in visited:
                    continue
                visited.add(neighbor.node.node_id)
                next_depth = depth + 1
                hits.append(
                    TraversalHit(
                        node=neighbor.node,
                        depth=next_depth,
                        via_edge_id=neighbor.edge.edge_id,
                    )
                )
                frontier.append((neighbor.node.node_id, next_depth))
        return hits

    @staticmethod
    def safety_boundary() -> dict[str, bool]:
        return {
            "shadow_only": True,
            "controls_devices": False,
            "online_training": False,
            "publishes_ros": False,
        }
