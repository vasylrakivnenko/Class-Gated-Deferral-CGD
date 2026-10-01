"""
Signed-in users, their per-day question limits, and a log of their requests,
in SQLite. Used by ask_ui.py when it runs behind the sign-in proxy.

The request log keeps who asked, when, and how the question was answered
(path, reader, latency, LLM calls), and since 2026-09-30 what was asked and
answered: the question, the answer with its confidence and evidence, and the
document. Each request has an id ("req_..."); each document is stored once,
under an id made from its text ("doc_..."), so asking again about the same
text doesn't store it again. When Tier 2 ran, which reader it used (router/tier2.py's
option id; "standard" / "priority" before 2026-10-01 meant Fireworks') and its timings
(tier2_*), to compare them on live traffic:
    SELECT tier2, COUNT(*), AVG(tier2_ms), AVG(tier2_server_ms) FROM requests
    WHERE tier2 IS NOT NULL GROUP BY tier2;
`settings` keeps what the admin page sets that must survive a restart: the tier switches
(router/stages.py) and the Tier 2 reader (router/tier2.py).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import threading
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, first_seen TEXT, last_seen TEXT);
CREATE TABLE IF NOT EXISTS usage (email TEXT, day TEXT, questions INTEGER, PRIMARY KEY (email, day));
CREATE TABLE IF NOT EXISTS limits (email TEXT PRIMARY KEY, daily_limit INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS requests (
    ts TEXT NOT NULL, day TEXT NOT NULL, email TEXT NOT NULL,
    status TEXT NOT NULL,          -- ok | error | limited
    reader TEXT, path TEXT, answered_by TEXT, ms INTEGER, llm_calls INTEGER
);
CREATE INDEX IF NOT EXISTS requests_day ON requests (day);
CREATE INDEX IF NOT EXISTS requests_email ON requests (email);
CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, text TEXT NOT NULL, first_seen TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY, value TEXT NOT NULL,  -- value is JSON
    updated_at TEXT, updated_by TEXT            -- which admin last changed it
);
"""
# Added to `requests` later, so also to a database made before them. Rows
# logged earlier keep NULL here.
REQUEST_CONTENT = (
    ("id", "TEXT"),  # new_request_id()
    ("question", "TEXT"),
    ("document_id", "TEXT"),  # documents.id
    ("answer", "TEXT"),
    ("confidence", "REAL"),
    ("evidence", "TEXT"),  # JSON: [{"text", "label", "p"}]
    # Tier 2 (router/reader.py), when it ran: to compare the readers on live traffic
    ("tier2", "TEXT"),  # router/tier2.py's option id; before 2026-10-01 "standard" | "priority" (Fireworks)
    ("tier2_ms", "REAL"),  # the reader call's round trip from this server
    ("tier2_server_ms", "REAL"),  # the server's own time: Fireworks' reply headers, or our GPU gateway's
    ("tier2_result", "TEXT"),  # what came of it: answered and checked, not stated, dropped, unavailable
    ("tier2_usage", "TEXT"),  # JSON: tokens, time to first token, attempts
    # Which tiers were on (router/stages.py `mask`), so "free classifier share" and the
    # latency numbers can be read against the configuration that produced them.
    ("stages", "TEXT"),
)


LEGACY_TIER2 = {"standard": "fireworks-standard", "priority": "fireworks-priority"}  # as logged before 2026-10-01


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def new_request_id() -> str:
    return "req_" + uuid.uuid4().hex


def document_id(text: str) -> str:
    """The same for the same text, so each document is stored once."""
    return "doc_" + hashlib.sha256(text.encode()).hexdigest()[:16]


class Usage:
    def __init__(self, path: Path, daily_limit: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_limit = daily_limit
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.executescript(SCHEMA)
        have = {row[1] for row in self._db.execute("PRAGMA table_info(requests)")}
        with self._db:
            for name, kind in REQUEST_CONTENT:
                if name not in have:
                    self._db.execute(f"ALTER TABLE requests ADD COLUMN {name} {kind}")
            self._db.execute("CREATE UNIQUE INDEX IF NOT EXISTS requests_id ON requests (id)")
            self._db.execute("CREATE INDEX IF NOT EXISTS requests_document ON requests (document_id)")
            # `settings` was made with only key and value on 2026-10-01 (the Tier 2 switch)
            have = {row[1] for row in self._db.execute("PRAGMA table_info(settings)")}
            for name in ("updated_at", "updated_by"):
                if name not in have:
                    self._db.execute(f"ALTER TABLE settings ADD COLUMN {name} TEXT")

    @staticmethod
    def _today() -> str:
        return _now().date().isoformat()

    def _limit(self, email: str) -> int:
        row = self._db.execute("SELECT daily_limit FROM limits WHERE email = ?", (email,)).fetchone()
        return row[0] if row else self.daily_limit

    def _used(self, email: str, day: str) -> int:
        row = self._db.execute("SELECT questions FROM usage WHERE email = ? AND day = ?", (email, day)).fetchone()
        return row[0] if row else 0

    def seen(self, email: str) -> None:
        now = _now().isoformat(timespec="seconds")
        with self._lock, self._db:
            self._db.execute("INSERT INTO users VALUES (?, ?, ?) ON CONFLICT(email) DO UPDATE SET last_seen = ?",
                             (email, now, now, now))

    def take(self, email: str) -> bool:
        """Count one question against today's limit; False if none are left."""
        day = self._today()
        with self._lock, self._db:
            if self._used(email, day) >= self._limit(email):
                return False
            self._db.execute("INSERT INTO usage VALUES (?, ?, 1) ON CONFLICT(email, day) DO UPDATE SET questions = questions + 1",
                             (email, day))
        return True

    def status(self, email: str) -> dict:
        with self._lock:
            limit, used = self._limit(email), self._used(email, self._today())
        return {"user": email, "daily_limit": limit, "remaining": max(0, limit - used)}

    def log(self, email: str, status: str, reader: str | None = None, answer: dict | None = None, *,
            request_id: str | None = None, question: str | None = None, document: str | None = None,
            stages: str | None = None) -> None:
        now = _now()
        ts = now.isoformat(timespec="seconds")
        a = answer or {}
        doc_id = document_id(document) if document else None
        evidence = json.dumps(a["evidence"], default=str) if "evidence" in a else None
        t2 = a.get("reader") or {}
        t2_usage = t2.get("usage") or {}
        with self._lock, self._db:
            if doc_id:
                self._db.execute("INSERT OR IGNORE INTO documents VALUES (?, ?, ?)", (doc_id, document, ts))
            self._db.execute(
                "INSERT INTO requests (ts, day, email, status, reader, path, answered_by, ms, llm_calls, "
                "id, question, document_id, answer, confidence, evidence, "
                "tier2, tier2_ms, tier2_server_ms, tier2_result, tier2_usage, stages) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    ts, now.date().isoformat(), email, status, reader,
                    a.get("path"), a.get("answered_by"), a.get("ms"), a.get("llm_calls"),
                    request_id, question, doc_id, a.get("answer"), a.get("confidence"), evidence,
                    (t2_usage.get("option") or t2_usage.get("service_tier")) if t2 else None,
                    round(t2["ms"], 1) if t2.get("ms") else None,
                    round(t2_usage["server_s"] * 1000, 1) if "server_s" in t2_usage else None,
                    t2.get("reason") or None,
                    json.dumps({k: v for k, v in t2_usage.items() if k not in ("service_tier", "option")})
                    if t2_usage else None,
                    stages))

    def tier2_stats(self, days: int = 7) -> dict:
        """Per Tier 2 reader over the last `days`: reads, how they ended, round-trip percentiles."""
        start = (_now().date() - dt.timedelta(days=days - 1)).isoformat()
        with self._lock:
            rows = self._db.execute("SELECT tier2, tier2_ms, tier2_result FROM requests WHERE day >= ? AND tier2 IS NOT NULL",
                                    (start,)).fetchall()
        by = {}
        for option, ms, result in rows:
            option = LEGACY_TIER2.get(option, option)
            s = by.setdefault(option, {"reads": 0, "answered": 0, "unavailable": 0, "ms": []})
            s["reads"] += 1
            s["answered"] += (result or "").startswith("copied")
            s["unavailable"] += (result or "").startswith("the reader was unavailable")
            if ms is not None:
                s["ms"].append(ms)
        pct = lambda xs, p: round(sorted(xs)[min(len(xs) - 1, int(len(xs) * p))]) if xs else None
        return {o: {"reads": s["reads"], "answered": s["answered"], "unavailable": s["unavailable"],
                    "p50_ms": pct(s["ms"], .5), "p90_ms": pct(s["ms"], .9)} for o, s in by.items()}

    def set_limit(self, email: str, daily_limit: int | None) -> None:
        """A per-user limit (0 blocks the user); None goes back to the default."""
        with self._lock, self._db:
            if daily_limit is None:
                self._db.execute("DELETE FROM limits WHERE email = ?", (email,))
            else:
                self._db.execute("INSERT INTO limits VALUES (?, ?) ON CONFLICT(email) DO UPDATE SET daily_limit = excluded.daily_limit",
                                 (email, daily_limit))

    def get_setting(self, key: str, default=None):
        """A stored JSON setting, `default` if it was never set or can't be read."""
        with self._lock:
            row = self._db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return default

    def set_setting(self, key: str, value, by: str = "") -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO settings (key, value, updated_at, updated_by) VALUES (?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
                "value = excluded.value, updated_at = excluded.updated_at, updated_by = excluded.updated_by",
                (key, json.dumps(value), _now().isoformat(timespec="seconds"), by))

    def setting_meta(self, key: str) -> dict:
        with self._lock:
            row = self._db.execute("SELECT updated_at, updated_by FROM settings WHERE key = ?", (key,)).fetchone()
        return {"updated_at": row[0], "updated_by": row[1]} if row else {"updated_at": None, "updated_by": None}

    def stats(self, days: int = 30) -> dict:
        """Everything the admin dashboard shows."""
        today = self._today()
        start = (_now().date() - dt.timedelta(days=days - 1)).isoformat()
        week = (_now().date() - dt.timedelta(days=6)).isoformat()
        q = lambda sql, *args: self._db.execute(sql, args).fetchall()
        with self._lock:
            (users_total,), = q("SELECT COUNT(*) FROM users")
            (users_new_today,), = q("SELECT COUNT(*) FROM users WHERE substr(first_seen, 1, 10) = ?", today)
            (active_today,), = q("SELECT COUNT(DISTINCT email) FROM requests WHERE day = ?", today)
            (ok_today, err_today, lim_today, ms_today), = q(
                "SELECT COALESCE(SUM(status = 'ok'), 0), COALESCE(SUM(status = 'error'), 0), COALESCE(SUM(status = 'limited'), 0), "
                "AVG(CASE WHEN status = 'ok' THEN ms END) FROM requests WHERE day = ?", today)
            (ok_total,), = q("SELECT COUNT(*) FROM requests WHERE status = 'ok'")
            (ok_week, free_week, ms_week), = q(
                "SELECT COUNT(*), COALESCE(SUM(path = 'classifier'), 0), AVG(ms) FROM requests WHERE status = 'ok' AND day >= ?", week)
            by_day = {d: {"classifier": c, "llm": l, "errors": e, "limited": m} for d, c, l, e, m in q(
                "SELECT day, SUM(status = 'ok' AND path = 'classifier'), SUM(status = 'ok' AND path != 'classifier'), "
                "SUM(status = 'error'), SUM(status = 'limited') FROM requests WHERE day >= ? GROUP BY day", start)}
            users = [dict(zip(("email", "first_seen", "last_seen", "today", "limit", "custom_limit", "week", "total", "avg_ms", "errors"), r)) for r in q(
                """SELECT u.email, u.first_seen, u.last_seen,
                          COALESCE((SELECT questions FROM usage WHERE email = u.email AND day = ?), 0),
                          COALESCE(l.daily_limit, ?), l.daily_limit IS NOT NULL,
                          (SELECT COUNT(*) FROM requests WHERE email = u.email AND status = 'ok' AND day >= ?),
                          (SELECT COUNT(*) FROM requests WHERE email = u.email AND status = 'ok'),
                          (SELECT AVG(ms) FROM requests WHERE email = u.email AND status = 'ok'),
                          (SELECT COUNT(*) FROM requests WHERE email = u.email AND status = 'error')
                   FROM users u LEFT JOIN limits l ON l.email = u.email
                   ORDER BY u.last_seen DESC""", today, self.daily_limit, week)]
            recent = [dict(zip(("ts", "email", "status", "reader", "path", "answered_by", "ms", "llm_calls", "tier2", "stages"), r))
                      for r in q("SELECT ts, email, status, reader, path, answered_by, ms, llm_calls, tier2, stages "
                                 "FROM requests ORDER BY ts DESC, rowid DESC LIMIT 25")]
            for r in recent:
                r["tier2"] = LEGACY_TIER2.get(r["tier2"], r["tier2"])
        daily = []
        for i in range(days):
            d = (_now().date() - dt.timedelta(days=days - 1 - i)).isoformat()
            daily.append({"day": d, **by_day.get(d, {"classifier": 0, "llm": 0, "errors": 0, "limited": 0})})
        return {
            "today": today, "default_limit": self.daily_limit,
            "tiles": {
                "users_total": users_total, "users_new_today": users_new_today, "active_today": active_today,
                "questions_today": ok_today, "errors_today": err_today, "limited_today": lim_today,
                "avg_ms_today": round(ms_today) if ms_today is not None else None,
                "questions_total": ok_total, "questions_week": ok_week,
                "free_share_week": round(free_week / ok_week, 3) if ok_week else None,
                "avg_ms_week": round(ms_week) if ms_week is not None else None,
            },
            "daily": daily,
            "users": [{**u, "custom_limit": bool(u["custom_limit"]), "avg_ms": round(u["avg_ms"]) if u["avg_ms"] is not None else None} for u in users],
            "recent": recent,
        }
