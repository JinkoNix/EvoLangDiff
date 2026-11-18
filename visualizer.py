"""
visualizer.py — Continuous Spatial Diffusion Server & Dialect Observer.
Includes zero-copy binary Float32Array streaming (/api/state_bin, /api/step_bin)
and on-demand RIFF/WAVE speech audio synthesis (/api/audio).
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import json
import os
import struct
import threading
from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
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

    def get_audio_wav(
        self,
        x: int | None = None,
        y: int | None = None,
        word_id: int | None = None,
        phoneme: str | None = None,
        stage_idx: int | None = None,
    ) -> bytes:
        """Synthesizes playable 16-bit 16kHz mono RIFF/WAVE audio on-demand."""
        with self.lock:
            # 1. Phoneme symbol playback
            if phoneme:
                from phonetics import _CANONICAL_ATTRACTORS
                from acoustics import AcousticEngine
                clean_sym = phoneme.replace("ː", "").replace("˥", "").replace("˦", "").replace("˧", "").replace("˨", "").replace("˩", "").strip("/")
                base_sym = clean_sym.replace("ʰ", "")
                attractor_dict = dict(_CANONICAL_ATTRACTORS)

                if base_sym in attractor_dict:
                    from phonetics import Phoneme
                    g = list(attractor_dict[base_sym])
                    if "ʰ" in clean_sym:
                        g[7] = 0.75
                    dur = 1.35 if "ː" in phoneme else 1.0
                    p = Phoneme(g, duration=dur)
                    return AcousticEngine.synthesize_wav_pcm([p.vector], base_f0=120.0)
                return b""

            # 2. Localized dialect word playback (current or historical checkpoint)
            if word_id is not None and x is not None and y is not None:
                from acoustics import AcousticEngine
                from phonetics import Phoneme, repair_phonemes

                cache_words = getattr(self.grid, "_probe_words_cache", None)
                w = None
                if cache_words and (x, y, self.grid.epoch) in cache_words:
                    w = cache_words[(x, y, self.grid.epoch)].get(word_id)

                if not w:
                    w = next((item for item in self.grid.lexical_manifold if item.id == word_id), None)

                if w:
                    # If requesting a historical ancestor checkpoint:
                    if stage_idx is not None and 0 <= stage_idx < len(w.history):
                        historical_tensor = w.history[stage_idx].tensor
                        return AcousticEngine.synthesize_wav_pcm(list(historical_tensor), base_f0=120.0)

                    if w.phonemes:
                        return AcousticEngine.synthesize_wav_pcm([p.vector for p in w.phonemes], base_f0=120.0)

            return b""


STATE = DiffusionSimulationState(size=64)


class EvoLangDiffHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
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

    def _send_audio_wav(self, wav_bytes: bytes):
        if not wav_bytes:
            self.send_error(404, "Audio synthesis failed")
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.end_headers()
        self.wfile.write(wav_bytes)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._set_json()
            payload = json.dumps(STATE.get_grid_state(), default=str)
            self.wfile.write(payload.encode("utf-8"))

        elif parsed.path == "/api/state_bin":
            self._send_binary(STATE.get_grid_state_bin())

        elif parsed.path == "/api/audio":
            params = parse_qs(parsed.query)
            x = int(params["x"][0]) if "x" in params else None
            y = int(params["y"][0]) if "y" in params else None
            word_id = int(params["word_id"][0]) if "word_id" in params else None
            phoneme = params["phoneme"][0] if "phoneme" in params else None
            stage_idx = int(params["stage_idx"][0]) if "stage_idx" in params else None
            wav = STATE.get_audio_wav(x=x, y=y, word_id=word_id, phoneme=phoneme, stage_idx=stage_idx)
            self._send_audio_wav(wav)

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

        elif parsed.path == "/api/audio":
            x = data.get("x")
            y = data.get("y")
            word_id = data.get("word_id")
            phoneme = data.get("phoneme")
            wav = STATE.get_audio_wav(x=x, y=y, word_id=word_id, phoneme=phoneme)
            self._send_audio_wav(wav)

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