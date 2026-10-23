"""Novel-view renderer for the splat: headless Chromium driving web/render.html (Spark).

    with SplatRenderer("data/alma_v2/brush_30k_masked/splat_30000.ply") as r:
        img = r.render(view)          # any object with R, t, K, width, height (scene.View)

The page and the splat are served from the repo root by a throwaway local HTTP
server, so the same Spark renderer serves the agent here and the web demo later.
"""

import base64
import functools
import http.server
import socketserver
import threading
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class SplatRenderer:
    def __init__(self, splat_path, headless=True, timeout_ms=600_000, mesh_options=None):
        self.splat_url = "/" + str(Path(splat_path).resolve().relative_to(ROOT))
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.mesh_options = mesh_options or {}

    def __enter__(self):
        handler = functools.partial(_QuietHandler, directory=str(ROOT))
        self._httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
        self._httpd.daemon_threads = True
        port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=self.headless,
            args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"])
        self.page = self._browser.new_page()
        self.page.set_default_timeout(self.timeout_ms)
        self.page.on("pageerror", lambda e: print(f"[page error] {e}"))
        self.page.goto(f"http://127.0.0.1:{port}/web/render.html")
        self.page.wait_for_function("window.twin && window.twin.ready", timeout=30_000)
        self.num_splats = self.page.evaluate("([u, o]) => window.twin.load(u, o)", [self.splat_url, self.mesh_options])
        return self

    def __exit__(self, *exc):
        self._browser.close()
        self._pw.stop()
        self._httpd.shutdown()

    def configure(self, **opts):
        self.page.evaluate("o => window.twin.configure(o)", opts)

    def render(self, view, near=0.01, far=500.0, settle_frames=2):
        """BGR uint8 image from a pinhole camera (R, t world->camera; K)."""
        cam = {"R": np.asarray(view.R, float).ravel().tolist(), "t": np.asarray(view.t, float).tolist(),
               "fx": float(view.K[0, 0]), "fy": float(view.K[1, 1]),
               "cx": float(view.K[0, 2]), "cy": float(view.K[1, 2]),
               "width": int(view.width), "height": int(view.height), "near": near, "far": far}
        url = self.page.evaluate("([c, n]) => window.twin.render(c, n)", [cam, settle_frames])
        data = base64.b64decode(url.split(",", 1)[1])
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
