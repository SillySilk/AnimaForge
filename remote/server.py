"""Remote dashboard: run Anima Forge's batch queue from another device.

A small web server meant to be reached privately over Tailscale (tailscale serve
proxies HTTPS to it). It queues runs on a folder of images using a saved set's
settings, starts and stops the headless runner, and shows progress, loss, the log,
preview images and the finished LoRA. See remote/control.py for the rules.

Answers only loopback and this PC's own Tailscale name. Changes are JSON POSTs.

Run: .venv\\Scripts\\pythonw.exe -m remote.server
"""
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from core.batch import DONE
from core.gpu_check import resident_gpu_apps
from remote import control, folders

PORT = 7802
_PAGE = Path(__file__).with_name("page.html")
_GPU_CACHE_SECONDS = 10


def tailscale_name() -> str:
    """This PC's Tailscale DNS name, or "" when Tailscale isn't running here."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        out = subprocess.run(["tailscale", "status", "--json"], capture_output=True,
                             text=True, timeout=15, creationflags=flags)
        return (json.loads(out.stdout or "{}").get("Self", {}).get("DNSName") or "").rstrip(".")
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


class Dashboard:
    """Everything the handler needs; tests swap the pieces that touch the machine."""

    def __init__(self, paths: control.Paths, thumb_dir: Path, gpu_apps=resident_gpu_apps,
                 start=control.start_runner, stop=control.stop_runner,
                 find=control.find_app_processes):
        self.paths = paths
        self.thumb_dir = Path(thumb_dir)
        self._gpu_apps = gpu_apps
        self._start, self._stop, self._find = start, stop, find
        self._gpu_cache = (0.0, [])
        self._lock = threading.Lock()     # one queue change / start / stop at a time

    def gpu_apps(self) -> list:
        at, apps = self._gpu_cache
        if time.monotonic() - at > _GPU_CACHE_SECONDS:
            apps = list(self._gpu_apps())
            self._gpu_cache = (time.monotonic(), apps)
        return apps

    def state(self) -> dict:
        gui, runners = self._find(self.paths.root)
        running = bool(runners)
        rows = control.run_rows(self.paths, running)
        return {"running": running, "app_open": bool(gui), "gpu_apps": self.gpu_apps(),
                "runs": rows,
                "has_work": any(r["state"] != DONE for r in rows)}

    def run_detail(self, index: int):
        run = control.queue_run(self.paths, index)
        if run is None:
            return None
        return {"loss": control.loss_points(self.paths, run),
                "files": control.run_files(self.paths, run),
                "log": control.log_tail(self.paths, run)}

    def _busy(self):
        """Why the queue can't be changed right now, or ""."""
        gui, runners = self._find(self.paths.root)
        if runners:
            return "The queue is running. Stop it first."
        if gui:
            return "Anima Forge is open on the PC. Close it first (it holds the queue)."
        return ""

    def add(self, body: dict):
        with self._lock:
            busy = self._busy()
            if busy:
                return False, busy
            return control.add_run(self.paths, body.get("folder", ""), body.get("name", ""),
                                   body.get("trigger", ""), body.get("template", ""))

    def remove(self, body: dict):
        with self._lock:
            busy = self._busy()
            if busy:
                return False, busy
            return control.remove_run(self.paths, int(body.get("index", -1)))

    def start(self, body: dict):
        with self._lock:
            busy = self._busy()
            if busy:
                return False, busy
            rows = control.run_rows(self.paths, False)
            if not any(r["state"] != DONE for r in rows):
                return False, "Nothing is waiting in the queue."
            apps = self.gpu_apps()
            if apps and not body.get("force"):
                return False, "gpu:" + ", ".join(apps)
            self._start(self.paths)
            return True, "Started."

    def stop(self, _body: dict):
        with self._lock:
            n = self._stop(self.paths)
            return (True, "Stopped.") if n else (False, "Nothing was running.")

    def close_app(self, _body: dict):
        """Close the desktop app (its unsaved changes are lost — the page says so)."""
        with self._lock:
            gui, _runners = self._find(self.paths.root)
            for p in gui:
                try:
                    p.kill()
                except Exception:
                    pass
            return (True, "Closed Anima Forge.") if gui else (False, "It wasn't open.")


def make_handler(dash: Dashboard, allowed_hosts, refresh_host=None):
    """`refresh_host` re-reads this PC's Tailscale name when an unknown *.ts.net name
    knocks: Tailscale may not have been up yet when the dashboard started at logon."""
    allowed = {h.lower() for h in allowed_hosts if h} | {"127.0.0.1", "localhost"}
    refreshed = {"at": 0.0}
    actions = {"/api/queue": dash.add, "/api/remove": dash.remove, "/api/start": dash.start,
               "/api/stop": dash.stop, "/api/close-app": dash.close_app}

    def host_name(value: str) -> str:
        value = (value or "").lower()
        value = re.sub(r"^[a-z]+://", "", value).split("/")[0]
        return value.rsplit(":", 1)[0] if value.count(":") == 1 else value

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype="text/plain; charset=utf-8",
                  head=False, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if not head:
                self.wfile.write(body)

        def _json(self, code: int, data, head=False):
            self._send(code, json.dumps(data).encode("utf-8"),
                       "application/json; charset=utf-8", head)

        def _host_ok(self) -> bool:
            name = host_name(self.headers.get("Host"))
            if name in allowed:
                return True
            if refresh_host and name.endswith(".ts.net") \
                    and time.monotonic() - refreshed["at"] > 30:
                refreshed["at"] = time.monotonic()
                fresh = (refresh_host() or "").lower()
                if fresh:
                    allowed.add(fresh)
            return name in allowed

        def do_HEAD(self):
            self.do_GET(head=True)

        def do_GET(self, head=False):
            if not self._host_ok():
                return self._send(403, b"Forbidden", head=head)
            parts = urlsplit(self.path)
            path, q = parts.path, parse_qs(parts.query)
            arg = lambda k: (q.get(k) or [""])[0]
            try:
                if path == "/":
                    return self._send(200, _PAGE.read_bytes(), "text/html; charset=utf-8", head)
                if path == "/api/state":
                    return self._json(200, dash.state(), head)
                if path == "/api/sets":
                    return self._json(200, control.list_templates(dash.paths), head)
                if path == "/api/folders":
                    listing = folders.browse(arg("path"))
                    if listing is None:
                        return self._json(404, {"message": "That folder isn't there."}, head)
                    return self._json(200, listing, head)
                if path == "/api/run":
                    detail = dash.run_detail(int(arg("index") or -1))
                    if detail is None:
                        return self._json(404, {"message": "That run is no longer queued."}, head)
                    return self._json(200, detail, head)
                if path == "/thumb":
                    return self._thumb(Path(unquote(arg("path"))), head)
                if path == "/run-thumb" or path == "/run-file":
                    run = control.queue_run(dash.paths, int(arg("index") or -1))
                    target = control.run_file(dash.paths, run, arg("name")) if run else None
                    if target is None:
                        return self._send(404, b"Not found", head=head)
                    if path == "/run-thumb":
                        return self._thumb(target, head)
                    return self._send_file(target, "download" in q, head)
            except (ConnectionError, OSError):
                return
            except ValueError:
                return self._send(400, b"Bad request", head=head)
            return self._send(404, b"Not found", head=head)

        def do_POST(self):
            if not self._host_ok():
                return self._send(403, b"Forbidden")
            origin = self.headers.get("Origin")
            if origin and host_name(origin) != host_name(self.headers.get("Host")):
                return self._send(403, b"Forbidden")
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._send(415, b"JSON only")
            action = actions.get(urlsplit(self.path).path)
            if action is None:
                return self._send(404, b"Not found")
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
                ok, message = action(body if isinstance(body, dict) else {})
            except (ValueError, TypeError):
                return self._json(400, {"ok": False, "message": "Bad request."})
            return self._json(200 if ok else 409, {"ok": ok, "message": message})

        def _thumb(self, image: Path, head: bool):
            if image.suffix.lower() not in control.IMAGE_EXTS or not image.is_absolute() \
                    or not image.is_file():
                return self._send(404, b"Not found", head=head)
            try:
                thumb = folders.thumbnail(image, dash.thumb_dir)
            except Exception:            # unreadable image: say so, don't hang up
                return self._send(500, b"Could not make a thumbnail", head=head)
            return self._send(200, thumb.read_bytes(), "image/webp", head,
                              {"Cache-Control": "max-age=86400"})

        def _send_file(self, target: Path, download: bool, head: bool):
            size = target.stat().st_size
            start, end, code = 0, size - 1, 200
            m = re.match(r"^bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
            if m and size:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:                                  # bytes=-N: the last N bytes
                    start = max(0, size - int(m.group(2) or 0))
                if start >= size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                end, code = min(end, size - 1), 206
            self.send_response(code)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0]
                             or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            disposition = "attachment" if download else "inline"
            self.send_header("Content-Disposition",
                             f"{disposition}; filename*=UTF-8''{quote(target.name)}")
            self.end_headers()
            if head:
                return
            with open(target, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

    return Handler


def main():
    os.chdir(control.APP_ROOT)
    paths = control.default_paths()
    thumbs = Path(os.environ.get("LOCALAPPDATA", control.APP_ROOT)) / "AnimaForgeRemote" / "thumbs"
    host = tailscale_name()
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT),
                                make_handler(Dashboard(paths, thumbs), {host}, tailscale_name))
    print(f"Anima Forge remote dashboard on http://127.0.0.1:{PORT}"
          + (f" and {host}" if host else ""), file=sys.stderr)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
