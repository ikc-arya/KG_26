"""Aninext web UI — local preview with live reload (development only; Pages serves docs/).

Serves index.html straight from this folder (so edits show without re-exporting) and the
exported data from <repo>/docs/data/. A tiny poller injected into the page reloads it
whenever index.html changes on disk.

Run:  python Apply/src/web/serve.py            # -> http://127.0.0.1:8026
      (VS Code: Cmd+Shift+P -> "Simple Browser: Show" -> that URL, to keep it beside the code)
"""

from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent
DOCS = WEB.parents[2] / "docs"
PAGE = WEB / "index.html"

RELOAD = b"""<script>(() => { let t0;
  setInterval(async () => { try {
    const t = await (await fetch("/__mtime", { cache: "no-store" })).text();
    if (t0 && t !== t0) location.reload(); t0 = t; } catch {} }, 700); })();</script>"""


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?")[0].split("#")[0]
        if path == "/__mtime":
            return self._send(str(PAGE.stat().st_mtime_ns).encode(), "text/plain")
        if path in ("/", "/index.html"):
            return self._send(PAGE.read_bytes().replace(b"</body>", RELOAD + b"</body>"), "text/html; charset=utf-8")
        super().do_GET()

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        if "/__mtime" not in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8026)
    args = p.parse_args()
    if not (DOCS / "data" / "meta.json").exists():
        raise SystemExit("no exported data yet: run python Apply/src/web/export_web.py first")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, directory=str(DOCS)))
    print(f"Aninext preview -> http://127.0.0.1:{args.port}  (live reload on {PAGE.name} save)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
