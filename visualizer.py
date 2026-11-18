"""
visualizer.py — Continuous Spatial Diffusion Server & Dialect Observer.
Includes zero-copy binary Float32Array streaming (/api/state_bin, /api/step_bin)
for 24–60+ FPS browser rendering alongside standard JSON endpoints.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import json
import os
import struct
import threading
from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import numpy as np

from world_grid import WorldGrid
from lexicon import LexiconGenerator
from probe import DialectProbe


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class DiffusionSimulationState:
    def __init__(self, size: int = 64):
        self.size = size
        self.lock = threading.Lock()
        self.proto_lexicon = LexiconGenerator.populate_proto_lexicon(population=0.50)
        self.grid = WorldGrid(size=size, seed=42, proto_roots=self.proto_lexicon)
        self.events: list[str] = [
            f"Primordial Hearth founded with {len(self.proto_lexicon)} ancestral roots.",
            "Continuous 2D Planetary Manifold initialized."
        ]

    def step(self, epochs: int = 1):
        with self.lock:
            for _ in range(epochs):
                self.grid.step(dt=1.0)

            t = self.grid.epoch
            active_words = len(self.grid.lexical_manifold)
            active_polities = len(self.grid.polities)
            self.events.append(
                f"Epoch {t}: Active Sovereignties: {active_polities} | Global Living Roots: {active_words}."
            )
            if len(self.events) > 30:
                self.events = self.events[-30:]

    def _build_grid_metadata(self) -> tuple[dict, np.ndarray, np.ndarray]:
        h_max = np.max(self.grid.hegemony, axis=-1)
        h_sum = np.sum(self.grid.hegemony, axis=-1, keepdims=True)
        h_norm = np.where(h_sum > 0.01, self.grid.hegemony / np.maximum(0.01, h_sum), 0.0)

        dom_hegemony = np.where(h_max >= 0.05, np.argmax(self.grid.hegemony, axis=-1), -1)

        has_pop = (self.grid.pop > 0.008) & (self.grid.elev > 0.0)
        if has_pop.any() and len(self.grid.polities) > 0:
            unclaimed_pop = has_pop & (dom_hegemony < 0)
            if unclaimed_pop.any():
                active_p = self.grid.polities
                dists = np.stack([p.dist_cells for p in active_p], axis=-1)
                nearest_idx = np.argmin(dists, axis=-1)
                nearest_ids = np.array([p.id for p in active_p])[nearest_idx]
                dom_hegemony = np.where(unclaimed_pop, nearest_ids, dom_hegemony)

        dom_strength = np.max(h_norm, axis=-1)

        color_map = {p.id: p.color for p in self.grid.polities}
        serialized_polities = [
            {
                "id": p.id,
                "name": p.name or f"Domain #{p.id}",
                "literal_meaning": p.literal_meaning or "Autonomous",
                "x": p.x,
                "y": p.y,
                "prestige": round(p.prestige, 2),
                "color": p.color,
                "government_type": p.government_type,
                "centralization": round(p.centralization, 2),
                "institutionalization": round(
                    min(1.0, 0.25 + 0.45 * p.centralization + (0.30 if p.is_standardized else 0.0)),
                    2,
                ),
                "is_standardized": p.is_standardized,
                "standard_epoch": p.standard_epoch,
            }
            for p in self.grid.polities
        ]

        pending_city_events = list(self.grid.new_city_events)
        self.grid.new_city_events.clear()

        meta = {
            "epoch": self.grid.epoch,
            "size": self.size,
            "events": self.events,
            "polities": serialized_polities,
            "polity_colors": color_map,
            "landmarks": [lm.to_dict() for lm in self.grid.landmarks.values()],
            "rivers": self.grid.river_paths,
            "lakes": [lk.to_dict() for lk in self.grid.lakes],
            "new_city_events": pending_city_events,
        }
        return meta, dom_hegemony, dom_strength

    def get_grid_state(self) -> dict:
        with self.lock:
            meta, dom_hegemony, dom_strength = self._build_grid_metadata()
            g = self.grid
            meta.update({
                "elevation": np.round(g.elev, 2).tolist(),
                "population": np.round(g.pop, 3).tolist(),
                "migration_vx": np.round(g.migration_vx, 2).tolist(),
                "migration_vy": np.round(g.migration_vy, 2).tolist(),
                "infrastructure": np.round(g.infrastructure, 2).tolist(),
                "prestige": np.round(g.prestige, 2).tolist(),
                "humidity": np.round(g.hum, 2).tolist(),
                "temperature": np.round(g.temp, 2).tolist(),
                "vegetation": np.round(g.veg, 2).tolist(),
                "fauna": np.round(g.fauna, 2).tolist(),
                "permeability": np.round(g.permeability, 2).tolist(),
                "attention": np.round(g.attention, 2).tolist(),
                "complexity": np.round(g.typology[..., 2], 2).tolist(),
                "head_directionality": np.round(g.typology[..., 0], 2).tolist(),
                "synthesis_index": np.round(g.typology[..., 1], 2).tolist(),
                "hegemony": dom_hegemony.tolist(),
                "hegemony_strength": np.round(dom_strength, 2).tolist(),
            })
            return meta

    def get_grid_state_bin(self) -> bytes:
        with self.lock:
            meta, dom_hegemony, dom_strength = self._build_grid_metadata()
            g = self.grid
            planes = np.stack([
                g.elev, g.pop, g.migration_vx, g.migration_vy,
                g.infrastructure, g.prestige, g.hum, g.temp,
                g.veg, g.fauna, g.permeability,
                g.typology[..., 2], g.typology[..., 0], g.typology[..., 1],
                dom_hegemony.astype(np.float32), dom_strength.astype(np.float32),
                g.attention[..., 0], g.attention[..., 1], g.attention[..., 2],
                g.attention[..., 3], g.attention[..., 4], g.attention[..., 5], g.attention[..., 6],
            ], axis=0).astype(np.float32, copy=False)

        header = json.dumps(meta, default=str).encode("utf-8")
        pad = (-len(header)) & 3
        return struct.pack("<I", len(header)) + header + (b"\x00" * pad) + planes.tobytes()

    def probe_coordinate(self, x: int, y: int) -> dict:
        with self.lock:
            return DialectProbe.evoke_pixel_dialect(
                self.grid,
                x=x,
                y=y,
                epoch=self.grid.epoch,
            )


STATE = DiffusionSimulationState(size=64)


class EvoLangDiffHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress per-frame console spam at high FPS
        return

    def _set_json(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _send_binary(self, payload: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._set_json()
            payload = json.dumps(STATE.get_grid_state(), default=str)
            self.wfile.write(payload.encode("utf-8"))

        elif parsed.path == "/api/state_bin":
            self._send_binary(STATE.get_grid_state_bin())

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

        elif parsed.path == "/api/step_bin":
            epochs = int(data.get("epochs", 1))
            STATE.step(epochs)
            self._send_binary(STATE.get_grid_state_bin())

        elif parsed.path == "/api/probe":
            x = int(data.get("x", 0))
            y = int(data.get("y", 0))
            res = STATE.probe_coordinate(x, y)
            self._set_json()
            self.wfile.write(json.dumps(res, default=str).encode("utf-8"))

        else:
            self.send_error(404, "Endpoint not found")


def start_server(port: int = 8008):
    server = ThreadedHTTPServer(("", port), EvoLangDiffHandler)
    print("\n==================================================================")
    print("  EVOLANGDIFF CONTINUOUS MANIFOLD OBSERVER RUNNING")
    print(f"  Open your browser at: http://localhost:{port}")
    print("==================================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()


if __name__ == "__main__":
    start_server()