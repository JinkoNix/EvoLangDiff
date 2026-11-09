"""
visualizer.py — Continuous Spatial Diffusion Server & Dialect Observer.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import numpy as np

from world_grid import WorldGrid
from lexicon import LexiconGenerator
from probe import DialectProbe


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


class DiffusionSimulationState:
    def __init__(self, size: int = 64):
        self.size = size
        self.grid = WorldGrid(size=size, seed=42)
        self.proto_lexicon = LexiconGenerator.populate_proto_lexicon(population=0.50)
        self.events: list[str] = [
            f"Primordial Cradle founded at ({self.size//2}, {self.size//2}) with {len(self.proto_lexicon)} distinct ancestral roots.",
            "Continuous 2D Planetary Manifold initialized."
        ]

    def step(self, epochs: int = 1):
        for _ in range(epochs):
            self.grid.step(dt=1.0)

        t = self.grid.epoch
        v_estimate = int(59 + 180.0 * np.sqrt(t / 100.0) + 0.35 * t)
        active_polities = len(self.grid.polity_centers)
        self.events.append(
            f"Epoch {t}: Tensors diffused. Active Civilizations: {active_polities} | Lexicon capacity: ~{v_estimate} words."
        )
        if len(self.events) > 30:
            self.events = self.events[-30:]

    def get_grid_state(self) -> dict:
        h_sum = np.sum(self.grid.hegemony, axis=-1, keepdims=True)
        h_norm = np.where(h_sum > 0.01, self.grid.hegemony / np.maximum(0.01, h_sum), 0.0)
        dom_hegemony = np.argmax(self.grid.hegemony, axis=-1)
        dom_strength = np.max(h_norm, axis=-1)

        # Mapping polity IDs to their dynamic linguistic typology colors
        color_map = {p["id"]: p.get("color", [200, 100, 100]) for p in self.grid.polity_centers}

        return {
            "epoch": self.grid.epoch,
            "size": self.size,
            "events": self.events,
            "elevation": np.round(self.grid.elev, 2).tolist(),
            "population": np.round(self.grid.pop, 3).tolist(),
            "infrastructure": np.round(self.grid.infrastructure, 2).tolist(),
            "prestige": np.round(self.grid.prestige, 2).tolist(),
            "humidity": np.round(self.grid.hum, 2).tolist(),
            "temperature": np.round(self.grid.temp, 2).tolist(),
            "vegetation": np.round(self.grid.veg, 2).tolist(),
            "permeability": np.round(self.grid.permeability, 2).tolist(),
            "complexity": np.round(self.grid.typology[..., 2], 2).tolist(),
            "head_directionality": np.round(self.grid.typology[..., 0], 2).tolist(),
            "synthesis_index": np.round(self.grid.typology[..., 1], 2).tolist(),
            "hegemony": dom_hegemony.tolist(),
            "hegemony_strength": np.round(dom_strength, 2).tolist(),
            "polities": self.grid.polity_centers,
            "polity_colors": color_map,
        }

    def probe_coordinate(self, x: int, y: int) -> dict:
        return DialectProbe.evoke_pixel_dialect(
            self.grid,
            x=x,
            y=y,
            proto_roots=self.proto_lexicon,
            epoch=self.grid.epoch,
        )


STATE = DiffusionSimulationState(size=64)


class EvoLangDiffHandler(BaseHTTPRequestHandler):
    def _set_json(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._set_json()
            payload = json.dumps(STATE.get_grid_state(), default=str)
            self.wfile.write(payload.encode("utf-8"))

        elif parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            base = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
            html_p = os.path.join(base, "index.html")
            if os.path.exists(html_p):
                with open(html_p, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"<h1>index.html not found. Place it in the project root.</h1>")
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        data = json.loads(body) if body else {}

        if parsed.path == "/api/step":
            epochs = int(data.get("epochs", 1))
            STATE.step(epochs)
            self._set_json()
            self.wfile.write(json.dumps(STATE.get_grid_state(), default=str).encode("utf-8"))

        elif parsed.path == "/api/probe":
            x = int(data.get("x", 0))
            y = int(data.get("y", 0))
            res = STATE.probe_coordinate(x, y)
            self._set_json()
            self.wfile.write(json.dumps(res, default=str).encode("utf-8"))

        else:
            self.send_error(404, "Endpoint not found")


def start_server(port: int = 8008):
    server = ReusableHTTPServer(("", port), EvoLangDiffHandler)
    print(f"\n==================================================================")
    print(f"  EVOLANGDIFF CONTINUOUS MANIFOLD OBSERVER RUNNING")
    print(f"  Open your browser at: http://localhost:{port}")
    print(f"==================================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()


if __name__ == "__main__":
    start_server()