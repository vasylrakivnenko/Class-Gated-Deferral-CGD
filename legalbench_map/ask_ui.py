"""
A local web page for the router harness (router/ui.html), styled after the
TypeSafe playground: paste a document, ask a yes/no question, see the answer
with its confidence and how it was reached. Listens on 127.0.0.1 only; the
Jev key stays in this process.

Usage:
    /Users/vasyl/zadumai/.venv/bin/python /Users/vasyl/zadumai/legalbench_map/ask_ui.py [--port 8765]
then open http://127.0.0.1:8765

Behind a sign-in proxy (oauth2-proxy on router.zadum.ai), run with
--require-user: every request must carry the proxy's X-Forwarded-Email
header, and each user gets --daily-limit questions per UTC day, counted in
--usage-db. The proxy is the only thing that can reach 127.0.0.1, and it
overwrites the header, so it cannot be forged from outside. Each --admin
email can open /admin: users, requests per day, and per-user limits.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from router.bank import Bank
from router.harness import Harness
from router.systemone import SystemOne, SystemOneError
from router.usage import Usage

PAGE = Path(__file__).resolve().parent / "router" / "ui.html"
ADMIN_PAGE = Path(__file__).resolve().parent / "router" / "admin.html"
USER_HEADER = "X-Forwarded-Email"


class App:
    def __init__(self, usage: Usage | None = None, admins: frozenset = frozenset()):
        self.bank = Bank()
        self.jev = SystemOne.jev()
        self.kev = SystemOne.kev()
        self.usage = usage  # None: no sign-in, no limits (local use)
        self.admins = admins
        self._lock = threading.Lock()  # one question at a time: the clients' call counters are shared

    def ask(self, question: str, document: str, reader: str) -> dict:
        llm = self.kev if reader == "kev" else self.jev
        with self._lock:
            start = time.perf_counter()
            answer = Harness(self.jev, llm, self.bank).answer(question, document)
            elapsed = time.perf_counter() - start
        return {**answer.to_dict(), "seconds": round(elapsed, 1), "ms": round(elapsed * 1000)}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _user(self) -> str | None:
            """The signed-in email, or None after answering 401 when sign-in is required."""
            if app.usage is None:
                return ""
            email = (self.headers.get(USER_HEADER) or "").strip().lower()
            if not email:
                self._json(401, {"error": "Not signed in."})
                return None
            app.usage.seen(email)
            return email

        def _admin(self) -> str | None:
            """The signed-in admin's email, or None after answering 401/403."""
            email = self._user()
            if email is None:
                return None
            if app.usage is None or email not in app.admins:
                self._json(403, {"error": "Admins only."})
                return None
            return email

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/me":
                email = self._user()
                if email is not None:
                    me = app.usage.status(email) if app.usage else {"user": None}
                    self._json(200, {**me, "admin": email in app.admins})
            elif path == "/admin":
                if self._admin() is not None:
                    self._send(200, ADMIN_PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/admin/stats":
                if self._admin() is not None:
                    self._json(200, app.usage.stats())
            elif path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")  # re-read: edits show on refresh
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/api/admin/limit":
                if self._admin() is None:
                    return
                req = self._body()
                email, limit = (req.get("email") or "").strip().lower(), req.get("daily_limit")
                if not email or not (limit is None or (isinstance(limit, int) and 0 <= limit <= 100_000)):
                    self._json(400, {"error": "Send an email and a daily_limit from 0 to 100000, or null for the default."})
                    return
                app.usage.set_limit(email, limit)
                self._json(200, app.usage.status(email))
                return
            if self.path != "/api/ask":
                self.send_error(404)
                return
            email = self._user()
            if email is None:
                return
            req = self._body()
            question, document = (req.get("question") or "").strip(), (req.get("document") or "").strip()
            reader = "kev" if req.get("reader") == "kev" else "jev"
            if not question or not document:
                self._json(400, {"error": "Enter both a document and a question."})
                return
            if app.usage and not app.usage.take(email):
                app.usage.log(email, "limited", reader)
                self._json(429, {"error": f"You've used all {app.usage.status(email)['daily_limit']} questions for today. The limit resets at 00:00 UTC.",
                                 **app.usage.status(email)})
                return
            try:
                result = app.ask(question, document, reader)
            except SystemOneError as e:
                if app.usage:
                    app.usage.log(email, "error", reader)
                self._json(502, {"error": f"LLM call failed: {e}"})
                return
            if app.usage:
                app.usage.log(email, "ok", reader, result)
            self._json(200, {**result, **(app.usage.status(email) if app.usage else {})})

        def _body(self) -> dict:
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

        def _json(self, code: int, obj: dict):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _send(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--require-user", action="store_true", help=f"require the sign-in proxy's {USER_HEADER} header and enforce --daily-limit")
    p.add_argument("--daily-limit", type=int, default=50, help="questions per user per UTC day (default: 50)")
    p.add_argument("--usage-db", type=Path, default=Path.home() / ".local" / "state" / "zadum-router" / "usage.db")
    p.add_argument("--admin", action="append", default=[], help="email allowed to open /admin (repeatable; needs --require-user)")
    args = p.parse_args()
    usage = Usage(args.usage_db, args.daily_limit) if args.require_user else None
    admins = frozenset(a.strip().lower() for a in args.admin)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(App(usage, admins)))
    print(f"Router playground: http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
