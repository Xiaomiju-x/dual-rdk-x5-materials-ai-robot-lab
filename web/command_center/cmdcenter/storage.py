"""SQLite storage policy and bootstrap contract for the command center.

Importing this module is side-effect free. Connections, schema creation, and
default configuration seeding only happen through explicit function calls.
"""

from __future__ import annotations

import os
import sqlite3
import time


SQLITE_TIMEOUT_S = 10
SQLITE_JOURNAL_MODE = "WAL"
SQLITE_SYNCHRONOUS = "NORMAL"

SCHEMA_TABLES = (
    "samples",
    "kpi_samples",
    "events",
    "alarms",
    "maintenance",
    "workorders",
    "wo_log",
    "eln",
    "app_metrics",
    "alert_rules",
    "silences",
    "oncall",
    "notifications",
    "logs",
    "ncr",
    "capa",
    "esign",
    "pm_schedule",
    "spares",
    "releases",
    "config",
    "samples_hourly",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS samples(
  ts INTEGER NOT NULL, sys TEXT NOT NULL, serving TEXT,
  real_ms INTEGER, mirror_ms INTEGER);
CREATE INDEX IF NOT EXISTS idx_samples_sys_ts ON samples(sys, ts);
CREATE TABLE IF NOT EXISTS kpi_samples(
  ts INTEGER NOT NULL, source TEXT, predictions INTEGER,
  ci_coverage_pct REAL, ci_narrowing_pct REAL,
  audit_valid INTEGER, audit_total INTEGER, llm_up INTEGER, llm_total INTEGER);
CREATE INDEX IF NOT EXISTS idx_kpi_ts ON kpi_samples(ts);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL,
  sys TEXT, kind TEXT, severity TEXT, message TEXT);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS alarms(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  rule TEXT NOT NULL, sys TEXT, severity TEXT, message TEXT,
  ts_raised INTEGER NOT NULL, ts_cleared INTEGER, ts_ack INTEGER, ack_by TEXT);
CREATE INDEX IF NOT EXISTS idx_alarms_open ON alarms(ts_cleared);
CREATE TABLE IF NOT EXISTS maintenance(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL,
  asset TEXT NOT NULL, author TEXT, note TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_maint_asset ON maintenance(asset, ts);
CREATE TABLE IF NOT EXISTS workorders(
  id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT UNIQUE,
  formula TEXT NOT NULL, dop_symbol TEXT, dop_site TEXT, dop_pct REAL,
  created_ts INTEGER, created_by TEXT,
  stage INTEGER DEFAULT 0, state TEXT DEFAULT 'open',
  trace_id TEXT, verdict TEXT, pred_summary TEXT, pred_source TEXT,
  lambda_obs REAL, close_note TEXT);
CREATE TABLE IF NOT EXISTS wo_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, wo INTEGER,
  author TEXT, action TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS idx_wolog_wo ON wo_log(wo, id);
CREATE TABLE IF NOT EXISTS eln(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL, updated_ts INTEGER,
  author TEXT, title TEXT NOT NULL, formula TEXT, tags TEXT, body TEXT);
CREATE INDEX IF NOT EXISTS idx_eln_ts ON eln(ts);
CREATE TABLE IF NOT EXISTS app_metrics(
  ts INTEGER NOT NULL, rss_kb INTEGER, threads INTEGER, uptime_s INTEGER,
  req_total INTEGER, req_4xx INTEGER, req_5xx INTEGER,
  p50_ms REAL, p95_ms REAL, db_bytes INTEGER, samples_rows INTEGER);
CREATE INDEX IF NOT EXISTS idx_appm_ts ON app_metrics(ts);
CREATE TABLE IF NOT EXISTS alert_rules(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, sys TEXT,
  metric TEXT NOT NULL, op TEXT NOT NULL, threshold REAL NOT NULL,
  for_n INTEGER DEFAULT 1, severity TEXT DEFAULT 'warn', channel TEXT,
  enabled INTEGER DEFAULT 1, created_by TEXT, ts INTEGER);
CREATE TABLE IF NOT EXISTS silences(
  id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT, reason TEXT,
  ts_start INTEGER, ts_end INTEGER, created_by TEXT, ts INTEGER);
CREATE TABLE IF NOT EXISTS oncall(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, contact TEXT,
  ts_start INTEGER, ts_end INTEGER, created_by TEXT, ts INTEGER);
CREATE TABLE IF NOT EXISTS notifications(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, rule TEXT, channel TEXT,
  status TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS idx_notif_ts ON notifications(ts);
CREATE TABLE IF NOT EXISTS logs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, req_id TEXT, method TEXT,
  route TEXT, status INTEGER, ms REAL, usr TEXT, role TEXT, ip TEXT, msg TEXT);
CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs(id);
CREATE INDEX IF NOT EXISTS idx_logs_req ON logs(req_id);
CREATE TABLE IF NOT EXISTS ncr(
  id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT, batch TEXT, defect TEXT,
  severity TEXT, raised_by TEXT, ts INTEGER, status TEXT DEFAULT 'open', capa_id INTEGER);
CREATE TABLE IF NOT EXISTS capa(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ncr_id INTEGER, root_cause TEXT,
  action TEXT, owner TEXT, due INTEGER, status TEXT DEFAULT 'open', ts INTEGER);
CREATE TABLE IF NOT EXISTS esign(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, obj_type TEXT, obj_id TEXT,
  signer TEXT, role TEXT, meaning TEXT, reason TEXT, hash TEXT);
CREATE TABLE IF NOT EXISTS pm_schedule(
  id INTEGER PRIMARY KEY AUTOINCREMENT, asset TEXT NOT NULL, task TEXT,
  interval_days INTEGER, last_done INTEGER, next_due INTEGER, enabled INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS spares(
  id INTEGER PRIMARY KEY AUTOINCREMENT, part TEXT NOT NULL, asset TEXT,
  qty INTEGER, min_qty INTEGER, unit TEXT);
CREATE TABLE IF NOT EXISTS releases(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ver TEXT, ts INTEGER, files TEXT,
  sha TEXT, notes TEXT, by TEXT);
CREATE TABLE IF NOT EXISTS config(
  key TEXT PRIMARY KEY, value TEXT, type TEXT, updated_by TEXT, ts INTEGER);
CREATE TABLE IF NOT EXISTS samples_hourly(
  hour INTEGER NOT NULL, sys TEXT NOT NULL, avg_real_ms REAL, avg_mirror_ms REAL,
  n_real INTEGER, n_mirror INTEGER, n_down INTEGER, n INTEGER,
  PRIMARY KEY(hour, sys));
"""


def resolve_db_path(app_file):
    """Resolve the legacy database location without opening the database."""
    return os.environ.get("XRD_CMD_DB_PATH") or os.path.join(
        os.path.dirname(os.path.abspath(app_file)), "data.db"
    )


def connect(db_path):
    """Open one connection with the command-center concurrency policy."""
    con = sqlite3.connect(os.fspath(db_path), timeout=SQLITE_TIMEOUT_S)
    con.execute(f"PRAGMA journal_mode={SQLITE_JOURNAL_MODE}")
    con.execute(f"PRAGMA synchronous={SQLITE_SYNCHRONOUS}")
    return con


def initialize(db_path):
    """Create the 22-table schema and apply idempotent legacy migrations."""
    con = connect(db_path)
    try:
        con.executescript(SCHEMA_SQL)
        try:
            columns = [row[1] for row in con.execute("PRAGMA table_info(ncr)").fetchall()]
            if "capa_id" not in columns:
                con.execute("ALTER TABLE ncr ADD COLUMN capa_id INTEGER")
        except Exception:
            pass
        con.commit()
    finally:
        con.close()


def seed_defaults(db_path, now=None):
    """Seed operational configuration only when each destination table is empty."""
    con = connect(db_path)
    try:
        timestamp = int(time.time()) if now is None else int(now)
        if con.execute("SELECT COUNT(*) FROM alert_rules").fetchone()[0] == 0:
            rules = [
                ("服务双路径离线", None, "serving_down", "==", 1, 1, "crit", ""),
                ("HTTP 5xx 错误率超标", None, "req_err_pct", ">", 5, 2, "warn", ""),
                ("进程内存超 400MB", None, "rss_mb", ">", 400, 3, "warn", ""),
                ("审计链断裂", "lab", "audit_broken", "==", 1, 1, "crit", ""),
            ]
            for name, system, metric, op, threshold, for_n, severity, channel in rules:
                con.execute(
                    "INSERT INTO alert_rules(name,sys,metric,op,threshold,for_n,severity,channel,"
                    "enabled,created_by,ts) VALUES(?,?,?,?,?,?,?,?,1,'system',?)",
                    (name, system, metric, op, threshold, for_n, severity, channel, timestamp),
                )
        if con.execute("SELECT COUNT(*) FROM pm_schedule").fetchone()[0] == 0:
            schedules = [
                ("lab", "BPU 散热/风扇点检 + 日志清理"),
                ("car", "底盘电机/雷达校准点检"),
                ("arm", "六轴关节回零 + 夹爪行程点检"),
            ]
            for asset, task in schedules:
                con.execute(
                    "INSERT INTO pm_schedule(asset,task,interval_days,last_done,next_due,enabled)"
                    " VALUES(?,?,?,?,?,1)",
                    (asset, task, 90, timestamp, timestamp + 90 * 86400),
                )
        if con.execute("SELECT COUNT(*) FROM spares").fetchone()[0] == 0:
            spares = [
                ("MG996R 25T 舵机", "arm", 6, 2, "个"),
                ("SG90 微舵机", "arm", 2, 1, "个"),
                ("12V/5A DC5525 适配器", "arm", 2, 1, "个"),
                ("钕磁铁 φ8×3mm", "arm", 32, 8, "颗"),
                ("USB Web Camera 720p", "arm", 2, 1, "个"),
                ("MG996R 备件", "arm", 0, 2, "个"),
            ]
            for part, asset, quantity, minimum, unit in spares:
                con.execute(
                    "INSERT INTO spares(part,asset,qty,min_qty,unit) VALUES(?,?,?,?,?)",
                    (part, asset, quantity, minimum, unit),
                )
        con.commit()
    finally:
        con.close()
