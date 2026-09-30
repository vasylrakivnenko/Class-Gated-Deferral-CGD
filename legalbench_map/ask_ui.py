"""
A local web page for the router harness (router/ui.html), styled after the
TypeSafe playground: paste a document, ask a yes/no question, see the answer
with its confidence and how it was reached. Listens on 127.0.0.1 only; the
Jev key stays in this process.

Usage:
    /Users/vasyl/zadumai/.venv/bin/python /Users/vasyl/zadumai/legalbench_map/ask_ui.py [--port 8765]
then open http://127.0.0.1:8765
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

PAGE = Path(__file__).resolve().parent / "router" / "ui.html"


class App:
    def __init__(self):
        self.bank = Bank()
        self.jev = SystemOne.jev()
        self.kev = SystemOne.kev()
        self._lock = threading.Lock()  # one question at a time: the clients' call counters are shared

    def ask(self, question: str, document: str, reader: str) -> dict:
        llm = self.kev if reader == "kev" else self.jev
        with self._lock:
            start = time.time()
            answer = Harness(self.jev, llm, self.bank).answer(question, document)
        return {**answer.to_dict(), "seconds": round(time.time() - start, 1)}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")  # re-read: edits show on refresh

        def do_POST(self):
            if self.path != "/api/ask":
                self.send_error(404)
                return
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            question, document = (req.get("question") or "").strip(), (req.get("document") or "").strip()
            if not question or not document:
                self._json(400, {"error": "Enter both a document and a question."})
                return
            try:
                self._json(200, app.ask(question, document, req.get("reader", "jev")))
            except SystemOneError as e:
                self._json(502, {"error": f"LLM call failed: {e}"})

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
    args = p.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(App()))
    print(f"Router playground: http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
