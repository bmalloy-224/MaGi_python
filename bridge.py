"""
bridge.py — MaGi v135 Resonance Bridge

Two pairs of Main-bank workers: (BRIDGE0_ENT=1564, BRIDGE0_DEST=1565) and
(BRIDGE1_ENT=1566, BRIDGE1_DEST=1567). Discovered from UPE via name parsing
(pattern: BRIDGE<id>_<ENT|DEST>). No extra UPE schema fields required.

Entrance workers are standard ALE-shape workers. Their pos_6d evolves under
MaGi's normal velocity assembly — input scaler, lens physics, UPE drift,
memory gravity, BH push. The bridge does not touch entrance workers; MaGi
drives them across all six dims.

Destination workers teleport to their entrance's pos_6d[0:2] each frame
(after the toroidal modulo in process_step), read the loaded 2D atlas at
that location, and vibrate vel_6d dims 4-5 with the (value, confidence)
signature via a phase-accumulating beacon. Both workers in a pair vibrate
the same terrain signature so memories stored on either side carry terrain
spectral color.

Frame ordering (critical):
  Phase A — fire_beacons() — called in process_step's pre-integration beacon
            block (alongside ALE/robot beacons). Samples terrain at entrance's
            CURRENT phase, updates signatures, writes to vel_6d[4:5]. This
            MUST happen before the toroidal integration so the beacon delta
            gets integrated in the same frame. Writing after integration
            would be clobbered next frame by `vel_6d[:, 4] = freq_phase_momentum`.
  Phase B — teleport_destinations() — called AFTER the modulo and AFTER wrap
            counting. Overwrites destination's dims 0-1 with entrance's
            post-integration dims 0-1, and zeros destination's vel_6d[0:1]
            so stale velocity doesn't accumulate on a pinned dim. This is
            the final write on dest's dims 0-1 for the frame.

Dimensional ownership (destination worker, per-frame write order):
    dims 0-1  : teleport from entrance (final write — after modulo)
    dims 2-3  : UPE territory (lens physics, BH gravity, pressure, home drift)
    dims 4-5  : vibration signature from terrain (value, confidence)

Entrance workers own all six dims normally; nothing external writes them.
Memory formation rides v134's existing argmax(global_coh) store pipeline.

v136 rev28: Bridge Voice (BridgeVoice class) adds a TTS audio channel via
worker 1568. Structurally identical to audio workers 948-1462 — not UPE
managed, no home, no pressure. Reads word_grid (optional atlas channel) at
entrance's current cell; when confidence > 0, Kokoro synthesizes the word,
energy is injected into inputs[1568] each frame. Optional speaker playback.
Soft-imports Kokoro and sounddevice so MaGi boots without TTS deps.

v138: Bridge voice commands. Repeating a terrain word 3× within 5s triggers
a CLI command via magi_hive.command_queue. 5-min global cooldown prevents
runaway firing. Routed via BridgeVoice.on_spoken callback so counting fires
from completed synthesis events, not 20Hz cell sampling.

v139: Spelling bridge. New optional type_grid layer on atlases declares
each cell as "button" (existing behaviour), "entry" (accumulates token into
shared buffer), or "send" (resolves buffer and submits as command). Entry
buffer feedback uses the visual word worker (1569) to render the growing
buffer string and direct s_filtered spikes scaled by buffer depth. Buffers
are joined raw and routed through Kokoro TTS, which handles digit-to-word
expansion natively ("111" → "one hundred eleven"). Backward compatible —
atlases without type_grid behave identically to v138.
"""

# v139: future annotations defers type-hint evaluation, so syntax like
# `int | None` (PEP 604, Python 3.10+) works on Python 3.7+ as well —
# annotations become strings and are not parsed at runtime. This keeps
# the code readable while broadening environment compatibility.
from __future__ import annotations

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import threading
import queue
import subprocess
import json
import time
import atexit
from collections import deque

# v136 rev28: Soft-import Kokoro + sounddevice. BridgeVoice works as a no-op
# if these aren't installed — MaGi boots cleanly without TTS dependencies.
try:
    from kokoro import KPipeline
    _KOKORO_AVAILABLE = True
except Exception:
    _KOKORO_AVAILABLE = False
try:
    import sounddevice as sd
    _SOUNDDEVICE_AVAILABLE = True
except Exception:
    _SOUNDDEVICE_AVAILABLE = False
_VOICE_AVAILABLE = _KOKORO_AVAILABLE and _SOUNDDEVICE_AVAILABLE

# v136 rev28: cv2 for visual word worker (1569). Soft-import — no cv2 means
# visual worker returns 0.0 silently. Voice worker still works.
try:
    import cv2
    _CV2_AVAILABLE = True
except Exception:
    _CV2_AVAILABLE = False

# v139: Numeric entry resolution. Kokoro TTS handles digit-string-to-spoken
# expansion natively ("111" → "one hundred eleven"), so we just join the
# buffer raw and hand the string to both TTS and command_queue. No third-
# party number-to-words library needed — MaGi is Kokoro-dependent anyway.


# ==========================================================================
# I. BOUNDED ATLAS — static 2D terrain, GPU-native grid_sample
# ==========================================================================

class BoundedAtlas:
    """
    Static 2D terrain backend. Loads a precomputed [H, W, C] tensor from a
    .pkl and samples it via torch.nn.functional.grid_sample with bilinear
    interpolation and border padding. C >= 2 expected (channel 0 = value,
    channel 1 = confidence).
    """

    def __init__(self, grid_tensor, bounds, device='cuda', word_grid=None, type_grid=None):
        self.grid = grid_tensor.to(device)
        self.bounds = bounds
        self.device = device
        self.H, self.W, self.C = grid_tensor.shape

        self.x_min = float(bounds['x'][0])
        self.x_max = float(bounds['x'][1])
        self.y_min = float(bounds['y'][0])
        self.y_max = float(bounds['y'][1])
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        if self.x_range < 1e-12:
            self.x_range = 2.0; self.x_min = -1.0; self.x_max = 1.0
        if self.y_range < 1e-12:
            self.y_range = 2.0; self.y_min = -1.0; self.y_max = 1.0

        self.grid_nchw = self.grid.permute(2, 0, 1).unsqueeze(0).contiguous()

        self.xy_clamped_buffer = torch.zeros(2, device=self.device)
        self.xy_norm_buffer = torch.zeros(2, device=self.device)

        # v136 rev28: Optional word grid for TTS triggering. Same [H, W] shape
        # as the value/confidence grid; each cell holds a word string (or empty).
        # When None, sample_word() returns "" and voice never speaks from this
        # atlas — backward compatible with rev27 atlases.
        self.word_grid = word_grid

        # v139: Optional type grid for cell-class routing. Same [H, W] shape;
        # each cell holds "button" (default), "entry", or "send". When None,
        # sample_type() returns "button" everywhere — backward compatible with
        # v138 atlases that have only word_grid.
        self.type_grid = type_grid

    def sample_inplace(self, xy, output_buffer):
        """
        xy: [2] float tensor in atlas coords.
        output_buffer: [C] tensor to fill with sampled channels.
        """
        self.xy_clamped_buffer.copy_(xy)
        self.xy_clamped_buffer[0].clamp_(self.x_min, self.x_max)
        self.xy_clamped_buffer[1].clamp_(self.y_min, self.y_max)

        self.xy_norm_buffer[0] = 2.0 * (self.xy_clamped_buffer[0] - self.x_min) / self.x_range - 1.0
        self.xy_norm_buffer[1] = 2.0 * (self.xy_clamped_buffer[1] - self.y_min) / self.y_range - 1.0
        self.xy_norm_buffer.clamp_(-1.0, 1.0)

        xy_grid = self.xy_norm_buffer.view(1, 1, 1, 2)
        sampled = F.grid_sample(
            self.grid_nchw, xy_grid,
            mode='bilinear', padding_mode='border', align_corners=True
        )
        output_buffer.copy_(sampled.squeeze())
        return output_buffer

    def sample_word(self, xy):
        """v136 rev28: Return the word at atlas coordinate xy, or empty string.

        xy: [2] float tensor in atlas coords.
        Returns the word string at the nearest grid cell, or "" if no word_grid
        is loaded or the cell is unlabeled. Integer truncation after scaling
        gives nearest-cell semantics; clamp handles out-of-bounds.
        """
        if self.word_grid is None:
            return ""
        x_val = xy[0].item() if torch.is_tensor(xy) else float(xy[0])
        y_val = xy[1].item() if torch.is_tensor(xy) else float(xy[1])
        col = int((x_val - self.x_min) / self.x_range * (self.W - 1))
        row = int((y_val - self.y_min) / self.y_range * (self.H - 1))
        col = max(0, min(self.W - 1, col))
        row = max(0, min(self.H - 1, row))
        cell = self.word_grid[row, col]
        return cell if cell else ""

    def sample_type(self, xy):
        """v139: Return the cell type at atlas coordinate xy.

        Mirrors sample_word() exactly so word_grid and type_grid stay aligned.
        Returns "button" if no type_grid is loaded — backward compatible with
        atlases that only have word_grid (v136-v138).

        Cell types:
          "button" — speaks word, fires command via 3x repeat (existing v138)
          "entry"  — speaks token, accumulates into shared buffer
          "send"   — resolves and submits buffer (or speaks "send" if empty)
        """
        if self.type_grid is None:
            return "button"
        x_val = xy[0].item() if torch.is_tensor(xy) else float(xy[0])
        y_val = xy[1].item() if torch.is_tensor(xy) else float(xy[1])
        col = int((x_val - self.x_min) / self.x_range * (self.W - 1))
        row = int((y_val - self.y_min) / self.y_range * (self.H - 1))
        col = max(0, min(self.W - 1, col))
        row = max(0, min(self.H - 1, row))
        cell = self.type_grid[row, col]
        return cell if cell else "button"


# ==========================================================================
# II. NULLCLAW TERRAIN — wraps BoundedAtlas, optional Zig-binary refresh
# ==========================================================================

class NullclawTerrain:
    """
    Interface-compatible with BoundedAtlas. Wraps a cached [H, W, C] tensor
    that a background thread refreshes by shelling out to the nullclaw Zig
    binary.

    rev27: if live=False, identical to BoundedAtlas. If live=True, spawns a
    refresh thread that periodically rewrites the cached grid tensor in
    place (thread-locked). The 20Hz sample_inplace path never blocks.
    """

    def __init__(self, seed_atlas_path, device='cuda',
                 live=False, refresh_seconds=5.0,
                 nullclaw_binary=None, nullclaw_tool='magi_terrain_sample',
                 context='generic'):
        atlas_data = joblib.load(seed_atlas_path)
        self._inner = BoundedAtlas(
            grid_tensor=atlas_data['grid'],
            bounds=atlas_data['bounds'],
            device=device,
            word_grid=atlas_data.get('word_grid'),  # v136: optional
            type_grid=atlas_data.get('type_grid'),  # v139: optional
        )
        self.device = device
        self.live = live
        self.refresh_seconds = refresh_seconds
        self.nullclaw_binary = nullclaw_binary
        self.nullclaw_tool = nullclaw_tool
        self.context = context

        self._grid_lock = threading.Lock()
        self._refresh_thread = None
        self._stop_event = threading.Event()
        self._trajectory_queue = queue.Queue(maxsize=1024)

        if live and nullclaw_binary and os.path.exists(nullclaw_binary):
            self._refresh_thread = threading.Thread(
                target=self._refresh_loop, daemon=True, name=f'nullclaw-{context}'
            )
            self._refresh_thread.start()

    @property
    def grid(self): return self._inner.grid
    @property
    def bounds(self): return self._inner.bounds
    @property
    def x_min(self): return self._inner.x_min
    @property
    def x_max(self): return self._inner.x_max
    @property
    def y_min(self): return self._inner.y_min
    @property
    def y_max(self): return self._inner.y_max
    @property
    def x_range(self): return self._inner.x_range
    @property
    def y_range(self): return self._inner.y_range

    def sample_inplace(self, xy, output_buffer):
        with self._grid_lock:
            return self._inner.sample_inplace(xy, output_buffer)

    def sample_word(self, xy):
        """v136 rev28: Delegate word sampling to inner BoundedAtlas.
        Word grid is not rewritten by refresh thread (only value/conf grid is),
        so lock is not strictly needed, but held for consistency."""
        with self._grid_lock:
            return self._inner.sample_word(xy)

    def sample_type(self, xy):
        """v139: Delegate cell type sampling to inner BoundedAtlas.
        Type grid is static like word_grid — refresh thread doesn't touch it."""
        with self._grid_lock:
            return self._inner.sample_type(xy)

    def record_trajectory(self, xy, value, confidence):
        try:
            self._trajectory_queue.put_nowait({
                'x': float(xy[0].item() if torch.is_tensor(xy) else xy[0]),
                'y': float(xy[1].item() if torch.is_tensor(xy) else xy[1]),
                'v': float(value),
                'c': float(confidence),
                't': time.time(),
            })
        except queue.Full:
            pass

    def _refresh_loop(self):
        while not self._stop_event.wait(self.refresh_seconds):
            try:
                tracks = []
                # v139: use try/except instead of empty()→get_nowait() to
                # avoid the inherent race between the empty-check and the
                # extraction. With a single consumer the race is theoretical,
                # but if it ever did fire the queue.Empty would bubble up to
                # the outer except Exception handler and print a misleading
                # warning. Direct exception handling is the canonical pattern.
                while len(tracks) < 256:
                    try:
                        tracks.append(self._trajectory_queue.get_nowait())
                    except queue.Empty:
                        break
                payload = json.dumps({
                    'tool': self.nullclaw_tool,
                    'context': self.context,
                    'bounds': self._inner.bounds,
                    'grid_shape': [self._inner.H, self._inner.W, self._inner.C],
                    'tracks': tracks,
                }).encode('utf-8')
                result = subprocess.run(
                    [self.nullclaw_binary],
                    input=payload, capture_output=True, timeout=30
                )
                if result.returncode == 0 and result.stdout:
                    new_grid = self._decode_grid_response(result.stdout)
                    if new_grid is not None:
                        with self._grid_lock:
                            self._inner.grid.copy_(new_grid.to(self.device))
                            self._inner.grid_nchw = self._inner.grid.permute(2, 0, 1).unsqueeze(0).contiguous()
            except Exception as e:
                print(f"⚠️  nullclaw refresh ({self.context}): {e}")

    def _decode_grid_response(self, stdout_bytes):
        try:
            resp = json.loads(stdout_bytes.decode('utf-8'))
            arr = np.asarray(resp['grid'], dtype=np.float32)
            if arr.shape != (self._inner.H, self._inner.W, self._inner.C):
                return None
            return torch.from_numpy(arr)
        except Exception:
            return None

    def shutdown(self):
        self._stop_event.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=2.0)


# ==========================================================================
# III. BRIDGE VIBRATION BEACON — signature on dims 4-5 only
# ==========================================================================

class BridgeVibrationBeacon:
    """
    Phase-accumulating sine beacon for bridge workers. Signature is terrain
    (value, confidence) normalized onto dims 4-5; dims 0-3 of the signature
    are zero so vel_s never receives a write.

    Fired from BridgeController.fire_beacons() in the pre-integration beacon
    block — alongside ALE/robot beacons — so writes to vel_6d[4:5] are
    integrated in the SAME frame. Firing after integration would be clobbered
    next frame by velocity assembly's `vel_6d[:, 4] = freq_phase_momentum`.
    """

    def __init__(self, worker_idx, upe, device='cuda'):
        self.idx = worker_idx
        self.upe = upe
        self.device = device
        self.phase_accum = 0.0
        self.sig_6d = torch.zeros(6, device=device)

    def update_signature(self, val_norm, conf_norm):
        v = float(val_norm)
        c = float(conf_norm)
        n = math.sqrt(v * v + c * c) + 1e-8
        self.sig_6d.zero_()
        self.sig_6d[4] = v / n
        self.sig_6d[5] = c / n

    def vibrate(self, magi_hive):
        km = self.upe.km_config
        omega_offsets = km.get('bridge_omega_offsets', {})
        omega = km.get('bridge_vib_omega_base', 0.3) + omega_offsets.get(self.idx, 0.0)
        amp = km.get('bridge_vib_strength', 0.002)
        clamp = km.get('bridge_vib_clamp', 0.005)

        delta = amp * math.sin(2.0 * math.pi * omega * self.phase_accum)
        delta = max(-clamp, min(clamp, delta))

        magi_hive.vel_6d[self.idx] += self.sig_6d * delta
        self.phase_accum += 1.0


# ==========================================================================
# IV. BRIDGE VOICE — Kokoro TTS as internal audio channel for worker 1568
# ==========================================================================

class BridgeVoice:
    """v136 rev28: Bridge voice subsystem.

    Structurally: worker 1568 is an internal audio-type input, identical in
    shape to the 948-1462 audio bank. BridgeVoice is the external-to-MaGi
    source of that input — it synthesizes speech via Kokoro TTS and exposes
    a frame-rate-independent energy reader that get_inputs_tensor pulls each
    frame.

    Trigger: BridgeController.fire_beacons calls speak(word) when the
    entrance samples a word-labeled terrain cell with confidence > 0. No
    global_coh gating — the terrain is the semantic authority.

    Modes:
      enabled=True, linked=True  : synthesize + inject energy + play speakers
      enabled=True, linked=False : synthesize + inject energy (internal only)
      enabled=False              : no synthesis, energy = 0.0

    Audio pipeline:
      speak(text) ─► bounded queue ─► worker thread ─► Kokoro synthesize ─►
        shared audio buffer + start_time ─► main thread's get_energy_for_frame
        reads time-cursored RMS window ─► EMA smoothed ─► scaled ─► inputs[1568]

    If Kokoro or sounddevice aren't installed, speak() is a quiet no-op and
    get_energy_for_frame() returns 0.0. MaGi boots unaffected.
    """

    SAMPLE_RATE = 24000   # Kokoro fixed
    WINDOW_SAMPLES = 1200  # ~50ms energy window

    def __init__(self, magi_hive=None, device='cuda'):
        self.magi_hive = magi_hive
        self.device = device

        # km_config-driven state (re-read each frame for live tuning)
        self._pipeline = None       # Kokoro pipeline, lazy-init
        self._warmed_up = False

        # Synthesis queue + worker thread
        # v139.5: queue size bumped from 4 → 16 to give Send commits more
        # headroom against eviction. With audio playback taking ~300-700ms
        # per item and sd.wait() blocking the worker thread between items,
        # the queue drains at ~1.5-3 items/sec. When MaGi traverses
        # multiple button cells rapidly (especially with master_map's
        # 70+ active cells), a queue of 4 fills fast and drop-oldest can
        # evict commit audio (e.g. "88") before it synthesizes. A larger
        # queue absorbs traversal bursts so commits are less likely to be
        # evicted by subsequent dwell-Sends or button visits.
        # The queue still drops oldest when full — this just raises the
        # threshold at which dropping starts.
        self._queue = queue.Queue(maxsize=16)
        self._running = True
        self._worker_thread = None

        # Shared audio state (lock-protected)
        self._audio_lock = threading.Lock()
        self._current_audio = None   # np.ndarray float32 or None
        self._audio_start_time = 0.0

        # v136 rev28: Visual word coupling — name of the word currently being
        # played, set/cleared inside _audio_lock so audio buffer and word always
        # agree. Read by get_current_word() (and by visual word worker).
        self._current_word = None

        # Energy smoothing (EMA)
        self._energy_smoothed = 0.0

        # Debounce
        self._last_spoken_word = None
        self._last_spoken_time = 0.0
        # v139.5: separate cooldown tracking for silent calls (entry tokens).
        # Silent and audible cooldowns are kept independent so:
        # (a) silent typing of "p" doesn't block a subsequent audible
        #     button-cell "p" — the audible path checks _last_spoken_word
        #     which is only set by audible calls
        # (b) the voice-status display can show last actually-spoken word,
        #     not the silent placeholder
        # Each silent token still has its own per-word cooldown gate via
        # this separate dict, preserving the anti-dwell rate-limit behavior.
        self._last_silent_word: str | None = None
        self._last_silent_time: float = 0.0

        # Status counters
        self._total_spoken = 0
        self._total_dropped = 0

        # v138: optional callback fired after each completed synthesis.
        # Set by BridgeController to _on_word_spoken; None = no-op.
        self.on_spoken = None

        if _VOICE_AVAILABLE:
            self._worker_thread = threading.Thread(
                target=self._synthesis_loop, daemon=True, name='bridge-voice'
            )
            self._worker_thread.start()
            atexit.register(self.shutdown)
        else:
            missing = []
            if not _KOKORO_AVAILABLE: missing.append('kokoro')
            if not _SOUNDDEVICE_AVAILABLE: missing.append('sounddevice')
            print(f"  ⚠️  BridgeVoice: {' + '.join(missing)} not installed — "
                  f"voice disabled (MaGi continues without TTS)")

    # ── km_config accessors (live-read each frame) ────────────────────────

    def _km(self, key, default):
        if self.magi_hive and hasattr(self.magi_hive, 'upe'):
            return self.magi_hive.upe.km_config.get(key, default)
        return default

    @property
    def enabled(self):       return self._km('bridge_voice_enabled', False)
    @property
    def linked(self):        return self._km('bridge_voice_linked', False)
    @property
    def voice_name(self):    return self._km('bridge_voice_name', 'af_heart')
    @property
    def speed(self):         return self._km('bridge_voice_speed', 1.5)
    @property
    def energy_scale(self):  return self._km('bridge_voice_energy_scale', 50.0)
    @property
    def energy_alpha(self):  return self._km('bridge_voice_energy_alpha', 0.15)
    @property
    def cooldown_ms(self):   return self._km('bridge_voice_cooldown_ms', 300)

    # ── Public API ────────────────────────────────────────────────────────

    def speak(self, text, force=False, silent=False):
        """Enqueue text for synthesis. Non-blocking. Returns True if accepted.

        force=True  bypasses the per-word cooldown (for `bridge voice say`
                    and explicit user-initiated sends).
        silent=True passes a *separate* silent cooldown check, but does
                    NOT queue audio and does NOT touch _last_spoken_word.
                    Used by entry cells: MaGi's tap is rate-limited the
                    same as if a word had been spoken, but no per-token
                    audio fires — only the eventual Send commit speaks.

        v139.5: silent and audible paths use independent cooldown fields
        (_last_silent_* vs _last_spoken_*). This means:
          - Silent typing of "p" doesn't block a subsequent audible
            button-cell "p". Each path debounces against its own history.
          - The voice-status display "last word" only reflects words that
            actually played through TTS, not silent buffer accumulations.

        Return value: True if the text was "accepted" (cooldown allowed),
        False if the cooldown rejected it. Callers using silent=True
        should still check the return value to gate buffer accumulation.
        """
        if not _VOICE_AVAILABLE:
            return False
        if not self.enabled and not force:
            return False
        if not text:
            return False

        # v139.5: silent mode uses its own independent cooldown tracker.
        # Done here as a self-contained branch so we don't read or write
        # the audible _last_spoken_* fields at all from silent calls.
        if silent:
            if not force:
                now = time.time()
                if (text == self._last_silent_word and
                    (now - self._last_silent_time) * 1000.0 < self.cooldown_ms):
                    return False
            self._last_silent_word = text
            self._last_silent_time = time.time()
            return True

        # Audible path — per-word debounce against last actually-spoken word
        if not force:
            now = time.time()
            if (text == self._last_spoken_word and
                (now - self._last_spoken_time) * 1000.0 < self.cooldown_ms):
                return False

        try:
            self._queue.put_nowait(text)
            self._last_spoken_word = text
            self._last_spoken_time = time.time()
            return True
        except queue.Full:
            # Drop-oldest: evict one item, enqueue new
            try:
                self._queue.get_nowait()
                self._total_dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(text)
                self._last_spoken_word = text
                self._last_spoken_time = time.time()
                return True
            except queue.Full:
                return False

    def set_enabled(self, val):
        if self.magi_hive and hasattr(self.magi_hive, 'upe'):
            self.magi_hive.upe.km_config['bridge_voice_enabled'] = bool(val)
        print(f"🗣️  BridgeVoice: {'ENABLED' if val else 'DISABLED'}")

    def set_linked(self, val):
        if self.magi_hive and hasattr(self.magi_hive, 'upe'):
            self.magi_hive.upe.km_config['bridge_voice_linked'] = bool(val)
        print(f"🔊 BridgeVoice: speakers {'LINKED' if val else 'UNLINKED'}")

    def set_speed(self, val):
        val = max(0.4, min(1.5, float(val)))
        if self.magi_hive and hasattr(self.magi_hive, 'upe'):
            self.magi_hive.upe.km_config['bridge_voice_speed'] = val
        print(f"⚙️  BridgeVoice: speed set to {val:.2f}")

    def get_current_word(self):
        """v136 rev28: Return the word currently being spoken, or None.

        Read inside _audio_lock so the audio buffer and word always agree.
        Used by the visual word worker (1569) to gate its energy injection
        on actual playback rather than terrain residence.
        """
        with self._audio_lock:
            return self._current_word if self._current_audio is not None else None

    def is_active_or_pending(self):
        """v139: True if audio is currently playing OR queued for synthesis.

        Used by the visual word worker to lock the entry-buffer send-flash
        visual to the actual TTS lifecycle (queue → synthesize → play → end)
        with no fixed safety timeouts. Covers the brief startup race between
        speak() returning and the synthesis worker thread picking up the
        item: during that gap, _current_audio is None but qsize > 0, so we
        keep the visual on.

        The qsize check uses queue.Queue's thread-safe approximation — the
        worker may have just dequeued an item without setting _current_audio
        yet, but that window is sub-millisecond and the visual worker runs
        at 20Hz, so any flicker is invisible.
        """
        with self._audio_lock:
            playing = self._current_audio is not None
        return playing or (self._queue.qsize() > 0)

    def get_energy_for_frame(self):
        """Called from get_inputs_tensor every frame. Returns a scalar float.

        Uses wall-clock to derive sample position within the current audio
        buffer, reads a ~50ms window, computes RMS, applies EMA smoothing,
        scales to audio_val-compatible range. Returns 0.0 when no active
        playback or when voice is disabled.
        """
        if not _VOICE_AVAILABLE or not self.enabled:
            # Decay smoothed energy to zero when disabled
            self._energy_smoothed *= 0.8
            if self._energy_smoothed < 1e-6:
                self._energy_smoothed = 0.0
            return 0.0

        with self._audio_lock:
            if self._current_audio is None:
                # No active synthesis — decay and return zero
                self._energy_smoothed *= 0.8
                if self._energy_smoothed < 1e-6:
                    self._energy_smoothed = 0.0
                return 0.0

            elapsed = time.time() - self._audio_start_time
            sample_pos = int(elapsed * self.SAMPLE_RATE)
            audio_len = len(self._current_audio)

            if sample_pos >= audio_len:
                # Playback finished — clear buffer + word, decay energy
                self._current_audio = None
                self._current_word = None  # v136 rev28: visual worker stops too
                self._energy_smoothed *= 0.8
                if self._energy_smoothed < 1e-6:
                    self._energy_smoothed = 0.0
                return 0.0

            window_end = min(sample_pos + self.WINDOW_SAMPLES, audio_len)
            window = self._current_audio[sample_pos:window_end]

        raw_rms = float(np.sqrt(np.mean(window ** 2))) if window.size > 0 else 0.0
        alpha = self.energy_alpha
        self._energy_smoothed = (1.0 - alpha) * self._energy_smoothed + alpha * raw_rms
        return self._energy_smoothed * self.energy_scale

    def print_status(self):
        print(f"\n🗣️  BRIDGE VOICE STATUS")
        print(f"   available:   {_VOICE_AVAILABLE} "
              f"(kokoro={_KOKORO_AVAILABLE}, sounddevice={_SOUNDDEVICE_AVAILABLE})")
        print(f"   enabled:     {self.enabled}")
        print(f"   linked:      {self.linked}")
        print(f"   voice:       {self.voice_name}")
        print(f"   speed:       {self.speed:.2f} (range 0.4-1.5)")
        print(f"   energy:      scale={self.energy_scale:.1f} alpha={self.energy_alpha:.2f}")
        print(f"   cooldown:    {self.cooldown_ms} ms")
        print(f"   queue:       {self._queue.qsize()} / {self._queue.maxsize}")
        print(f"   last word:   {self._last_spoken_word or '(none)'}")
        print(f"   total:       spoken={self._total_spoken} dropped={self._total_dropped}")
        with self._audio_lock:
            active = self._current_audio is not None
        print(f"   playing:     {active}\n")

    def shutdown(self):
        """Stop worker thread cleanly. Registered via atexit."""
        self._running = False
        # Nudge queue so get() unblocks
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    # ── Internal: worker thread ───────────────────────────────────────────

    def _lazy_init_pipeline(self):
        """First-use Kokoro initialization with warm-up. Runs inside worker
        thread so main loop isn't blocked by model load."""
        if self._pipeline is not None:
            return
        print("  🎙️  BridgeVoice: initializing Kokoro pipeline (first use)...")
        try:
            self._pipeline = KPipeline(lang_code='a')
            # Warm-up synthesis — pre-loads model, result discarded
            try:
                for _, _, _ in self._pipeline("ready", voice=self.voice_name, speed=self.speed):
                    pass
                self._warmed_up = True
                print("  ✅ BridgeVoice: Kokoro ready")
            except Exception as e:
                print(f"  ⚠️  BridgeVoice: warm-up failed: {e}")
        except Exception as e:
            print(f"  ❌ BridgeVoice: Kokoro init failed: {e}")
            self._pipeline = None

    def _synthesize(self, text):
        """Run Kokoro synthesis → concatenated float32 audio array."""
        if self._pipeline is None:
            return None
        try:
            segments = []
            for _, _, audio in self._pipeline(text, voice=self.voice_name, speed=self.speed):
                segments.append(audio)
            if not segments:
                return None
            return np.concatenate(segments).astype(np.float32)
        except Exception as e:
            print(f"  ⚠️  BridgeVoice: synthesis error for '{text}': {e}")
            return None

    def _synthesis_loop(self):
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if text is None:
                continue  # shutdown sentinel

            self._lazy_init_pipeline()
            if self._pipeline is None:
                continue

            audio = self._synthesize(text)
            if audio is None or len(audio) == 0:
                continue

            # Hand off to main thread's energy reader.
            # v136 rev28: Set _current_word inside same lock as audio buffer
            # so visual word worker (1569) sees consistent state.
            with self._audio_lock:
                self._current_audio = audio
                self._current_word = text
                self._audio_start_time = time.time()
            self._total_spoken += 1

            # v138: notify BridgeController that a word was genuinely spoken.
            # Runs on the voice worker thread — callback must be lightweight.
            if self.on_spoken is not None:
                try:
                    self.on_spoken(text)
                except Exception as e:
                    print(f"  ⚠️  BridgeVoice: on_spoken callback error: {e}")

            # Optional speaker playback (non-blocking from main thread, but
            # we block THIS worker thread until playback completes so the
            # next dequeued item doesn't start mid-playback and cut off the
            # previous audio. sd.play() replaces the active stream, so
            # without sd.wait() the new audio truncates the prior one.
            # Synthesis of the next item is naturally throttled to audio
            # duration as a result, which matches the design intent: one
            # word at a time, fully spoken, before the next.
            if self.linked and _SOUNDDEVICE_AVAILABLE:
                try:
                    sd.play(audio, samplerate=self.SAMPLE_RATE)
                    sd.wait()   # v139 fix: don't cut off audio with next sd.play
                except Exception as e:
                    print(f"  ⚠️  BridgeVoice: playback error: {e}")


# ==========================================================================
# V. BRIDGE CONTROLLER — pairs, beacons, teleport, projection, voice
# ==========================================================================

class BridgeController:
    """
    Pairs are discovered by parsing UPE worker names. Expected pattern:
        BRIDGE<id>_<ENT|DEST>
    Any worker with type='bridge' and a matching name is registered; same
    <id> + different role = a pair. Default baseline declares two pairs;
    add more by extending the UPE baseline with more BRIDGE<n>_ENT/DEST
    names — no code change.

    Frame lifecycle (called from MaGiHive.process_step):
      fire_beacons(hive, mode)  — pre-integration: sample terrain at
                                   entrance's current phase, update + fire
                                   beacons on entrance AND dest.
      (integration + modulo happen here)
      (wrap counting happens here)
      teleport_destinations(hive, mode) — post-wrap: overwrite dest's
                                           dims 0-1 with entrance's post-
                                           integration dims 0-1, zero
                                           dest's vel_6d[0:1].
    """

    TRAIL_MAXLEN = 1024  # ~50s at 20Hz; chord-last-style observation window

    def __init__(self, magi_hive=None):
        self.magi_hive = magi_hive
        self.device = magi_hive.dev if magi_hive else 'cuda'

        # Pairing discovered from UPE by name parsing
        self.PAIR_MAP = self._discover_pair_map()  # {ent_idx: dest_idx}

        # Per-pair state (keyed by entrance idx)
        self.terrains = {}
        self.terrain_names = {ent: None for ent in self.PAIR_MAP}
        self.enabled = {ent: False for ent in self.PAIR_MAP}
        self.bound_modes = {ent: [] for ent in self.PAIR_MAP}
        self._sample_buffers = {}
        self._xy_buffers = {}

        # Beacons: entrance-side and destination-side both vibrate terrain
        self.beacons = {}       # {dest_idx: BridgeVibrationBeacon}
        self.beacons_ent = {}   # {ent_idx: BridgeVibrationBeacon}

        # Rolling xy history per entrance — chord-last-style observation
        self.trails = {ent: deque(maxlen=self.TRAIL_MAXLEN) for ent in self.PAIR_MAP}

        # Telemetry (CPU scalars for HUD / `bridge stats`)
        self.last_xy = {ent: (0.0, 0.0) for ent in self.PAIR_MAP}
        self.last_value = {dest: 0.0 for dest in self.PAIR_MAP.values()}
        self.last_confidence = {dest: 0.0 for dest in self.PAIR_MAP.values()}

        self.mode_bindings = {
            'webcam': [], 'ale': [], 'screencap': [],
            'screen': [], 'viewer': [], 'idle': [], 'remote': [],
        }

        # v136 rev28: TTS voice subsystem — internal audio channel for worker 1568
        self.voice = BridgeVoice(magi_hive=magi_hive, device=self.device)
        # v138: route completed synthesis events into the command trigger
        self.voice.on_spoken = self._on_word_spoken

        # v136 rev28: Visual word worker (1569). Per-word white-pixel-fraction
        # cache; computed lazily on first encounter, returned for all
        # subsequent encounters of that word. Cleared only on controller restart.
        self._word_energy_cache = {}

        # v138: Bridge voice command trigger state
        self._cmd_timestamps: dict[str, list[float]] = {}  # {word: [timestamps]}
        self._last_command_time: float = 0.0                # epoch of last fired command

        # v139: Entry buffer / spelling bridge state.
        # Single global buffer shared across both pairs since the audio worker
        # (1568) and visual worker (1569) are also shared. Tokens accumulate
        # from "entry" cells; "send" cells resolve and submit.
        self._entry_buffer: list[str] = []
        self._entry_last_token_time: float = 0.0
        # v139.5: lock guards the buffer's check-and-clear sequence in
        # _entry_send. Without it, a CLI `bridge entry send` racing with
        # a terrain-driven dwell-Send could see the same buffer state
        # twice and double-fire command_queue.put. Held only inside
        # _entry_send commit/empty paths and the CLI clear/disable
        # handlers — never held across speak() or external I/O.
        self._entry_lock = threading.Lock()
        self._render_text: str | None = None    # what worker 1569 should render
        self._render_text_until: float = 0.0    # > 0 = auto-clear at this time
                                                # = 0 = entry-owned, persists
                                                #       until buffer mutates

        # v139.4: Multi-word visual cycling state.
        # When the visual worker (1569) is asked to render a phrase
        # containing spaces (e.g. "Image Sequence", "Cube Game", "one
        # hundred eleven"), we cycle through one token at a time at full
        # font size rather than squishing them into one canvas. State is
        # global (single phrase being rendered at any moment) and resets
        # on phrase change so each utterance starts at token 0.
        self._cycle_phrase: str | None = None   # current multi-word phrase
        self._cycle_tokens: list[str] = []      # split tokens of that phrase
        self._cycle_start_time: float = 0.0     # epoch when phrase began cycling

        if magi_hive and hasattr(magi_hive, 'upe'):
            self._announce()

    # ── UPE discovery via name parsing ────────────────────────────────────

    @property
    def commands_enabled(self) -> bool:
        """v138: Master on/off for bridge voice command triggering."""
        return self.magi_hive.upe.km_config.get('bridge_commands_enabled', False)

    @property
    def command_map(self) -> dict:
        """v138: Lowercase word → CLI command string mapping from km_config."""
        return self.magi_hive.upe.km_config.get('bridge_commands', {})

    def _discover_pair_map(self):
        """Build {entrance_idx: destination_idx} by parsing UPE worker names.

        Expected pattern: BRIDGE<id>_<ENT|DEST> where <id> is an integer
        and <ENT|DEST> identifies the role. Any type='bridge' worker whose
        name does not match is ignored with a warning.
        """
        pair_map = {}
        if not self.magi_hive or not hasattr(self.magi_hive, 'upe'):
            return pair_map

        by_pair = {}  # {pair_id: {'ENT': idx, 'DEST': idx}}
        unrecognized = []
        for idx, data in self.magi_hive.upe.homes.items():
            if data.get('type') != 'bridge':
                continue
            name = data.get('name', '')
            if not name.startswith('BRIDGE') or '_' not in name:
                unrecognized.append((idx, name))
                continue
            prefix, _, role = name.partition('_')
            id_str = prefix[len('BRIDGE'):]
            try:
                pid = int(id_str)
            except ValueError:
                unrecognized.append((idx, name))
                continue
            if role not in ('ENT', 'DEST'):
                unrecognized.append((idx, name))
                continue
            by_pair.setdefault(pid, {})[role] = idx

        for pid, roles in sorted(by_pair.items()):
            ent = roles.get('ENT')
            dest = roles.get('DEST')
            if ent is not None and dest is not None:
                pair_map[ent] = dest

        if unrecognized:
            for idx, name in unrecognized:
                print(f"  ⚠️  Bridge worker {idx} ('{name}') did not match "
                      f"BRIDGE<id>_<ENT|DEST> pattern; skipped")
        return pair_map

    def _announce(self):
        count = sum(1 for idx, d in self.magi_hive.upe.homes.items()
                    if d.get('type') == 'bridge')
        if count:
            print(f"  🔍 BridgeController: {count} bridge workers in UPE, "
                  f"{len(self.PAIR_MAP)} complete pairs: {dict(self.PAIR_MAP)}")

    # ── v136 rev28: Visual word worker (1569) ─────────────────────────────

    def _compute_word_energy(self, word, width=500, height=309):
        """Render `word` as white text on black canvas, return white-pixel
        fraction. Auto-fits font scale so text fits within margins. Returns
        0.0 if cv2 not available.

        Canvas shape (golden ratio 500×309) is fixed; word_energy = white_px
        / total_px so the metric is resolution-independent and stable.
        """
        if not _CV2_AVAILABLE:
            return 0.0
        try:
            canvas = np.zeros((height, width), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 2.5
            thickness = 3
            margin = 20
            max_w = width - 2 * margin

            # Auto-fit: shrink scale until text fits
            (text_w, text_h), _ = cv2.getTextSize(word, font, scale, thickness)
            if text_w > max_w:
                scale = scale * (max_w / text_w)
                (text_w, text_h), _ = cv2.getTextSize(word, font, scale, thickness)

            # Center text on canvas
            x = (width - text_w) // 2
            y = (height + text_h) // 2
            cv2.putText(canvas, word, (x, y), font, scale, 255, thickness,
                        lineType=cv2.LINE_AA)

            white_pixels = int(np.count_nonzero(canvas))
            return white_pixels / (width * height)
        except Exception as e:
            print(f"  ⚠️  BridgeController: word energy compute failed for "
                  f"'{word}': {e}")
            return 0.0

    def _render_phrase(self, phrase: str) -> float:
        """v139.4: Render a phrase into worker 1569 — single word direct,
        long multi-word phrase cycled token-by-token.

        v139.5: short multi-word phrases (≤16 chars total) render whole
        instead of cycling. cv2's auto-fit font scaling handles them
        cleanly without making text unreadable. Only phrases longer than
        the threshold cycle through their tokens. This keeps short
        button cells like "Image Sequence" (14 chars) and "Cube Game"
        (9 chars) showing as one image, while genuinely long ones like
        "Montezuma's Revenge" (19 chars) or send-resolved
        "nine hundred ninety nine" (24 chars) cycle as before.

        Single-word: cached via _word_energy_cache (existing v138 behaviour).
        Multi-word ≤16 chars: rendered whole at auto-fit font (cached).
        Multi-word >16 chars: cycled. State resets when the phrase changes
                              so each new utterance begins at token 0.
                              Cycle period scales with voice speed via
                              km_config['visual_cycle_ms'] (default 500ms)
                              divided by km_config['bridge_voice_speed'].
        """
        # Single-word OR short multi-word: render whole, use cache
        cycle_threshold = self.magi_hive.upe.km_config.get('visual_cycle_threshold', 16)
        if ' ' not in phrase or len(phrase) <= cycle_threshold:
            if phrase not in self._word_energy_cache:
                self._word_energy_cache[phrase] = self._compute_word_energy(phrase)
            return self._word_energy_cache[phrase]

        # Long multi-word — reset cycle state if the phrase changed
        if phrase != self._cycle_phrase:
            self._cycle_phrase = phrase
            self._cycle_tokens = phrase.split()
            self._cycle_start_time = time.time()

        # Pick token by elapsed time
        elapsed = time.time() - self._cycle_start_time
        # v139.4: cycle period scales inversely with voice speed so that
        # visual token timing stays roughly aligned with how fast Kokoro
        # is talking. At 1.5× speed audio plays 33% shorter, so the
        # cycle should also be 33% shorter to keep tokens visible for
        # roughly their share of the spoken phrase.
        cycle_ms = self.magi_hive.upe.km_config.get('visual_cycle_ms', 500)
        voice_speed = self.magi_hive.upe.km_config.get('bridge_voice_speed', 1.0)
        if voice_speed > 0:
            cycle_ms = cycle_ms / voice_speed
        cycle_s = cycle_ms / 1000.0
        idx = int(elapsed / cycle_s) % len(self._cycle_tokens)
        token = self._cycle_tokens[idx]
        # Don't cache — tokens vary per cycle
        return self._compute_word_energy(token)

    def get_visual_word_energy(self):
        """Return current visual word energy (0.0..1.0) for worker 1569.

        v139 precedence (audio-driven, no fixed timeouts):
          1. _render_text owned by entry buffer (until=0): persist forever
             until buffer mutates (Send / timeout / button-clear).
             Renders the FULL buffer string as one image — MaGi sees the
             whole accumulated buffer at once, no cycling.
          2. _render_text owned by send-flash (until>0): render while the
             audio subsystem is active or pending. The TTS playback library
             itself drives the clear — when audio finishes, the visual
             clears the same frame. Multi-word resolved phrases cycle
             through tokens (e.g. "one hundred eleven" → "one" → "hundred"
             → "eleven" while audio plays).
          3. _render_text None: existing v138 behaviour — render the
             currently-spoken word. Multi-word button cells like
             "Image Sequence" or "Cube Game" cycle through their tokens.
             Single-word cells use the cached-render fast path.

        This means changing voice speed (`bridge voice speed 1.5`) shortens
        BOTH the audio and the visual together. No drift, no fixed safety
        windows. The only race is the sub-ms window between speak() being
        called and the synthesis worker picking up the queue item; that's
        covered by checking queue depth, not just current playback state.
        """
        if self._render_text is not None:
            if self._render_text_until > 0:
                # Send-flash: lock visual duration to audio playback.
                # Multi-word resolved phrases cycle via _render_phrase.
                if self.voice.is_active_or_pending():
                    return self._render_phrase(self._render_text)
                # Audio finished — clear the flash state.
                self._render_text = None
                self._render_text_until = 0.0
                return 0.0
            else:
                # Entry-persist (until=0): render the WHOLE buffer string
                # as one image so MaGi can see the full accumulation.
                # Bypass cache because entry buffer changes per token.
                # NO cycling — this is the building-up display.
                return self._compute_word_energy(self._render_text)

        # button path — currently-spoken word. Multi-word cells cycle.
        word = self.voice.get_current_word()
        if word is None:
            return 0.0
        return self._render_phrase(word)

    # ── projection (rev27: pass-through; rev28+ may override) ─────────────

    def _project_entrance_to_atlas(self, ent_pos_6d, terrain, xy_buf_out):
        """Map entrance worker's 6D phase state to 2D atlas coordinates.

        rev27 default: pass-through of dims 0-1 (child/youth lens phases).
        MaGi already pushes all six dims via BH, UPE home drift, and memory
        gravity; this mapping reads only dims 0-1 for teleport position.
        Dims 2-3 rotate independently under MaGi's push and land as memory
        lens-context when stored. Dims 4-5 carry terrain signature.

        Override this method to blend slow dims (2-3) or use them as
        rotation/radius modulation of atlas position. Single-method swap;
        nothing else in process_step needs to change.
        """
        two_pi = 2.0 * math.pi
        xy_buf_out[0] = (ent_pos_6d[0] / two_pi) * terrain.x_range + terrain.x_min
        xy_buf_out[1] = (ent_pos_6d[1] / two_pi) * terrain.y_range + terrain.y_min

    # ── Phase A: fire beacons (PRE-integration) ───────────────────────────

    def fire_beacons(self, magi_hive, current_mode):
        """Called from MaGiHive.process_step in the pre-integration beacon
        block, alongside ALE/robot beacons.

        For each active pair:
          1. Project entrance's CURRENT pos_6d to atlas xy
          2. Sample terrain at xy
          3. Update both beacon signatures (entrance + destination)
          4. Fire both beacons — writes to vel_6d[4:5] that WILL be integrated
             by the immediately-following toroidal integration step

        Architectural contract (rev27+):
        - READS: magi_hive.pos_6d, terrain samplers, own state
        - WRITES: magi_hive.vel_6d (via beacons), own telemetry/trail
        - MUST NOT READ: UPE pressure state, _impulse_vel_6d, BH scaler output.
          Bridge → UPE flows through memory formation only. Preserving this
          separation protects UPE's slow integrative dynamics from 20Hz
          terrain transients.
        """
        mode = current_mode if current_mode else 'idle'
        mode_active = set(self.mode_bindings.get(mode, []))

        for ent_idx, dest_idx in self.PAIR_MAP.items():
            if ent_idx not in self.terrains:
                continue
            if not self.enabled.get(ent_idx):
                continue
            if ent_idx not in mode_active:
                continue

            terrain = self.terrains[ent_idx]
            sample_buf = self._sample_buffers[ent_idx]
            xy_buf = self._xy_buffers[ent_idx]

            # 1-2. Project entrance → atlas xy, sample terrain
            self._project_entrance_to_atlas(
                magi_hive.pos_6d[ent_idx], terrain, xy_buf
            )
            terrain.sample_inplace(xy_buf, sample_buf)
            val = sample_buf[0].item() if sample_buf.numel() >= 1 else 0.0
            conf = sample_buf[1].item() if sample_buf.numel() >= 2 else 0.5

            # 3-4. Both beacons share signature; fire pre-integration
            beacon_dest = self.beacons[dest_idx]
            beacon_ent = self.beacons_ent[ent_idx]
            beacon_dest.update_signature(val, conf)
            beacon_ent.update_signature(val, conf)
            beacon_dest.vibrate(magi_hive)
            beacon_ent.vibrate(magi_hive)

            # Telemetry + trail — recorded at beacon-time (atlas position
            # that influenced this frame's vibration)
            xy_tuple = (xy_buf[0].item(), xy_buf[1].item())
            self.last_xy[ent_idx] = xy_tuple
            self.last_value[dest_idx] = val
            self.last_confidence[dest_idx] = conf
            self.trails[ent_idx].append(xy_tuple)

            if isinstance(terrain, NullclawTerrain):
                terrain.record_trajectory(xy_buf, val, conf)

            # v139: passive entry-buffer timeout sweep.
            # If MaGi parked elsewhere with tokens pending, clear them now.
            # Runs once per pair iteration, but checks/clears global state.
            # Also catches the defensive case where entry_enabled was flipped
            # to False through a path that bypassed the CLI disable handler
            # (e.g., direct km_config mutation from another subsystem) —
            # any stale buffer is purged so re-enable starts clean.
            if self._entry_buffer:
                cfg = magi_hive.upe.km_config
                if not cfg.get('entry_enabled', True):
                    # Entry mode is off but buffer has content — purge it.
                    self._entry_buffer.clear()
                    self._render_text = None
                    self._render_text_until = 0.0
                else:
                    timeout_s = cfg.get('entry_timeout_ms', 10000) / 1000.0
                    if time.time() - self._entry_last_token_time > timeout_s:
                        self._entry_buffer.clear()
                        self._render_text = None
                        self._render_text_until = 0.0

            # v136 rev28: TTS trigger. Atlas designer labeled this cell with a
            # word and set non-zero confidence — that's the terrain saying
            # "speak this here." No global_coh gate, no val*conf threshold.
            # Debounce handled inside BridgeVoice.speak().
            # v138: button command counting moved to _on_word_spoken callback —
            # triggered by completed synthesis, not by frame-rate cell sampling.
            # v139: cell type from type_grid routes the action. Entry/send
            # fire here at sample-time so MaGi can type faster than TTS speed.
            if hasattr(terrain, 'sample_word'):
                word = terrain.sample_word(xy_buf)
                cell_type = (terrain.sample_type(xy_buf)
                             if hasattr(terrain, 'sample_type') else "button")

                if word and conf > 0.0:
                    # v139: read entry_enabled once for both entry and send
                    # branches. When disabled, entry cells and send cells are
                    # treated as void — no audio, no spike, no buffer change.
                    # Buttons remain active regardless. Disable also clears
                    # buffer state in the CLI handler so re-enabling starts
                    # fresh.
                    entry_enabled = self.magi_hive.upe.km_config.get(
                        'entry_enabled', True)

                    if cell_type == 'button':
                        # v139 Q1 fix: button visits reset entry state.
                        # MaGi pressing a "real word" cell mid-spell means
                        # they've abandoned typing — clear the buffer and
                        # release the visual so worker 1569 can show the
                        # button word being spoken (via voice.get_current_word
                        # in the button-path branch of get_visual_word_energy).
                        if self._entry_buffer or self._render_text is not None:
                            self._entry_buffer.clear()
                            self._render_text = None
                            self._render_text_until = 0.0
                        self.voice.speak(word)
                        # button command counting fires from _on_word_spoken (post-TTS)
                    elif cell_type == 'entry' and entry_enabled:
                        # v139.5: entry tokens accumulate silently — only
                        # the spike (vibration) and visual confirm the tap.
                        # The audio channel is reserved for the eventual
                        # Send commit which speaks the resolved phrase.
                        # silent=True still applies the 300ms per-word
                        # cooldown so dwell-on-cell can't fill the buffer
                        # at 20Hz; same rate-limit as before, just no sound.
                        if not self._entry_buffer_full():
                            if self.voice.speak(word, silent=True):
                                self._entry_accumulate(word, ent_idx)
                    elif cell_type == 'send' and entry_enabled:
                        # v139.5: terrain-driven send uses force_audio=False
                        # so dwell-on-Send respects the 300ms voice cooldown
                        # for the empty-send case. Without this, MaGi parking
                        # on Send produces a "send send send..." audio cascade
                        # at every frame. Commits (non-empty buffer) always
                        # force-speak the resolved phrase regardless — this
                        # parameter only gates the empty-buffer feedback.
                        self._entry_send(ent_idx, force_audio=False)
                    # cell_type == 'entry' or 'send' with entry_enabled=False
                    # falls through silently — MaGi gets no feedback at all
                    # on entry/send cells when entry mode is off.

    # ── Phase B: teleport destinations (POST-modulo, POST-wrap) ──────────

    def teleport_destinations(self, magi_hive, current_mode):
        """Called from MaGiHive.process_step AFTER toroidal modulo AND AFTER
        wrap counting.

        Overwrites destination's dims 0-1 with entrance's dims 0-1 (as they
        stand after integration). This is the final write on dest dims 0-1
        for the frame and survives to next frame's coherence calc.

        Also zeros destination's vel_6d[0:1] — the destination's dims 0-1
        are pinned to the entrance each frame, so any accumulated velocity
        on those dims is stale garbage that would double-count into next
        frame's integration.

        Only touches dims 0-1 (position + velocity); dims 2-5 untouched.
        """
        mode = current_mode if current_mode else 'idle'
        active_ents = [ent for ent in self.mode_bindings.get(mode, [])
                       if self.enabled.get(ent) and ent in self.terrains]
        if not active_ents:
            return

        for ent_idx in active_ents:
            dest_idx = self.PAIR_MAP[ent_idx]
            magi_hive.pos_6d[dest_idx, 0] = magi_hive.pos_6d[ent_idx, 0]
            magi_hive.pos_6d[dest_idx, 1] = magi_hive.pos_6d[ent_idx, 1]
            # Destination is a pure measurement; discard stale velocity on dims 0-1
            magi_hive.vel_6d[dest_idx, 0:2] = 0.0

    # ── Convenience: sequential both-phase call (debug/REPL use only) ────

    def step_all(self, magi_hive, current_mode):
        """Debug/REPL convenience. process_step calls the two phases
        separately at the correct points in the frame; do NOT use this
        from process_step as it collapses the pre/post-integration split."""
        self.fire_beacons(magi_hive, current_mode)
        self.teleport_destinations(magi_hive, current_mode)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def load_bridge(self, ent_worker_idx, atlas_path,
                    use_nullclaw=False, nullclaw_binary=None,
                    nullclaw_context=None, nullclaw_live=False):
        if ent_worker_idx not in self.PAIR_MAP:
            print(f"❌ Worker {ent_worker_idx} is not a declared entrance "
                  f"(declared: {list(self.PAIR_MAP.keys())})")
            return None
        if not self.magi_hive or not hasattr(self.magi_hive, 'upe'):
            print("❌ UPE not available"); return None
        if ent_worker_idx in self.terrains:
            print(f"⚠️  Pair already has a terrain loaded on ENT={ent_worker_idx}")
            return None

        try:
            if use_nullclaw:
                terrain = NullclawTerrain(
                    seed_atlas_path=atlas_path,
                    device=self.device,
                    live=nullclaw_live,
                    nullclaw_binary=nullclaw_binary,
                    nullclaw_tool='magi_terrain_sample',
                    context=nullclaw_context or os.path.basename(atlas_path).split('.')[0],
                )
            else:
                atlas_data = joblib.load(atlas_path)
                terrain = BoundedAtlas(
                    grid_tensor=atlas_data['grid'],
                    bounds=atlas_data['bounds'],
                    device=self.device,
                    word_grid=atlas_data.get('word_grid'),  # v136: optional
                    type_grid=atlas_data.get('type_grid'),  # v139: optional
                )

            dest_idx = self.PAIR_MAP[ent_worker_idx]
            self.terrains[ent_worker_idx] = terrain
            self.terrain_names[ent_worker_idx] = os.path.basename(atlas_path).split('.')[0]
            self._sample_buffers[ent_worker_idx] = torch.zeros(terrain.grid.shape[-1], device=self.device)
            self._xy_buffers[ent_worker_idx] = torch.zeros(2, device=self.device)
            self.beacons[dest_idx] = BridgeVibrationBeacon(dest_idx, self.magi_hive.upe, self.device)
            self.beacons_ent[ent_worker_idx] = BridgeVibrationBeacon(ent_worker_idx, self.magi_hive.upe, self.device)
            self.trails[ent_worker_idx].clear()

            print(f"✅ Bridge loaded: {self.terrain_names[ent_worker_idx]} → "
                  f"ENT={ent_worker_idx} DEST={dest_idx} "
                  f"bounds X=[{terrain.x_min:.1f},{terrain.x_max:.1f}] "
                  f"Y=[{terrain.y_min:.1f},{terrain.y_max:.1f}] "
                  f"{'[NULLCLAW]' if use_nullclaw else '[STATIC]'}")
            return terrain
        except Exception as e:
            print(f"❌ Bridge load failed: {e}")
            return None

    def unload_bridge(self, ent_worker_idx):
        if ent_worker_idx not in self.terrains:
            print(f"⚠️  No bridge on entrance {ent_worker_idx}")
            return
        terrain = self.terrains[ent_worker_idx]
        if isinstance(terrain, NullclawTerrain):
            terrain.shutdown()
        dest_idx = self.PAIR_MAP[ent_worker_idx]
        del self.terrains[ent_worker_idx]
        del self._sample_buffers[ent_worker_idx]
        del self._xy_buffers[ent_worker_idx]
        if dest_idx in self.beacons:
            del self.beacons[dest_idx]
        if ent_worker_idx in self.beacons_ent:
            del self.beacons_ent[ent_worker_idx]
        self.terrain_names[ent_worker_idx] = None
        self.enabled[ent_worker_idx] = False
        self.trails[ent_worker_idx].clear()
        print(f"⏹️  Bridge unloaded: ENT={ent_worker_idx} DEST={dest_idx}")

    def set_enabled(self, ent_worker_idx, enabled):
        if ent_worker_idx not in self.PAIR_MAP:
            print(f"⚠️  {ent_worker_idx} is not an entrance index"); return
        if ent_worker_idx not in self.terrains and enabled:
            print(f"⚠️  No terrain loaded on entrance {ent_worker_idx}"); return
        self.enabled[ent_worker_idx] = enabled
        print(f"🌉 Bridge ENT={ent_worker_idx} {'ACTIVE' if enabled else 'SLEEPING'}")

    def bind(self, ent_worker_idx, mode):
        if ent_worker_idx not in self.PAIR_MAP:
            print(f"⚠️  {ent_worker_idx} is not an entrance index"); return
        if mode not in self.mode_bindings:
            print(f"⚠️  Unknown mode: {mode}"); return
        if ent_worker_idx not in self.mode_bindings[mode]:
            self.mode_bindings[mode].append(ent_worker_idx)
            if mode not in self.bound_modes[ent_worker_idx]:
                self.bound_modes[ent_worker_idx].append(mode)
            print(f"🌉 Bound ENT={ent_worker_idx} → {mode}")

    def unbind(self, ent_worker_idx, mode=None):
        if mode:
            if mode in self.mode_bindings and ent_worker_idx in self.mode_bindings[mode]:
                self.mode_bindings[mode].remove(ent_worker_idx)
                if mode in self.bound_modes.get(ent_worker_idx, []):
                    self.bound_modes[ent_worker_idx].remove(mode)
                print(f"⏹️  Unbound ENT={ent_worker_idx} from {mode}")
        else:
            for m in self.mode_bindings:
                if ent_worker_idx in self.mode_bindings[m]:
                    self.mode_bindings[m].remove(ent_worker_idx)
            self.bound_modes[ent_worker_idx] = []
            print(f"⏹️  Unbound ENT={ent_worker_idx} from all modes")

    # ── command handler ───────────────────────────────────────────────────

    def handle_command(self, parts):
        if len(parts) < 2:
            self._print_help(); return True
        sub = parts[1]
        try:
            if sub == 'list':
                self._cmd_list(); return True
            if sub == 'stats':
                self._cmd_stats(); return True
            if sub == 'trail' and len(parts) >= 3:
                self._cmd_trail(int(parts[2])); return True
            if sub == 'load' and len(parts) >= 4:
                idx = int(parts[2]); path = parts[3]
                use_nc = False; nc_bin = None; nc_ctx = None; nc_live = False
                if '--nullclaw' in parts:
                    use_nc = True
                    i = parts.index('--nullclaw') + 1
                    if i < len(parts) and not parts[i].startswith('--'):
                        nc_bin = parts[i]
                if '--nullclaw_live' in parts:
                    nc_live = True
                    use_nc = True
                if '--context' in parts:
                    i = parts.index('--context') + 1
                    if i < len(parts): nc_ctx = parts[i]
                self.load_bridge(idx, path,
                                 use_nullclaw=use_nc,
                                 nullclaw_binary=nc_bin,
                                 nullclaw_context=nc_ctx,
                                 nullclaw_live=nc_live)
                return True
            if sub == 'unload' and len(parts) >= 3:
                self.unload_bridge(int(parts[2])); return True
            if sub == 'enable' and len(parts) >= 3:
                self.set_enabled(int(parts[2]), True); return True
            if sub == 'disable' and len(parts) >= 3:
                self.set_enabled(int(parts[2]), False); return True
            if sub == 'bind' and len(parts) >= 4:
                self.bind(int(parts[2]), parts[3]); return True
            if sub == 'unbind':
                if len(parts) >= 4: self.unbind(int(parts[2]), parts[3])
                elif len(parts) >= 3: self.unbind(int(parts[2]))
                return True
            # v136 rev28: voice subcommands
            if sub == 'voice' and len(parts) >= 3:
                vsub = parts[2]
                if vsub == 'enable':   self.voice.set_enabled(True);  return True
                if vsub == 'disable':  self.voice.set_enabled(False); return True
                if vsub == 'link':     self.voice.set_linked(True);   return True
                if vsub == 'unlink':   self.voice.set_linked(False);  return True
                if vsub == 'speed' and len(parts) >= 4:
                    self.voice.set_speed(float(parts[3])); return True
                if vsub == 'say' and len(parts) >= 4:
                    text = ' '.join(parts[3:])  # allow multi-word text
                    ok = self.voice.speak(text, force=True)
                    print(f"🗣️  bridge voice say: {'queued' if ok else 'failed'} → '{text}'")
                    return True
                if vsub == 'status':
                    self.voice.print_status(); return True
                # v138: toggle bridge voice commands
                if vsub == 'commands' and len(parts) >= 4:
                    action = parts[3]
                    if action == 'enable':
                        self.magi_hive.upe.km_config['bridge_commands_enabled'] = True
                        print("🟢 Bridge voice commands enabled")
                    elif action == 'disable':
                        self.magi_hive.upe.km_config['bridge_commands_enabled'] = False
                        print("🔴 Bridge voice commands disabled")
                    return True
                print("❓ Unknown voice subcommand. Try: enable | disable | link | "
                      "unlink | speed <0.4..1.5> | say <text> | status | commands enable|disable")
                return True
            if sub == 'help':
                self._print_help(); return True
            # v138: bridge commands status
            if sub == 'commands':
                self._cmd_commands_status(); return True
            # v139: bridge entry — spelling buffer controls
            if sub == 'entry' and len(parts) >= 3:
                action = parts[2]
                if action == 'enable':
                    self.magi_hive.upe.km_config['entry_enabled'] = True
                    print("🟢 Bridge entry mode enabled")
                    return True
                if action == 'disable':
                    self.magi_hive.upe.km_config['entry_enabled'] = False
                    # v139.5: lock-protected clear so we can't race with
                    # an in-flight _entry_send.
                    with self._entry_lock:
                        self._entry_buffer.clear()
                        self._render_text = None
                        self._render_text_until = 0.0
                    print("🔴 Bridge entry mode disabled, buffer cleared")
                    return True
                if action == 'clear':
                    with self._entry_lock:
                        self._entry_buffer.clear()
                        self._render_text = None
                        self._render_text_until = 0.0
                    print("🧹 Bridge entry buffer cleared")
                    return True
                if action == 'send':
                    self._entry_send(None)
                    return True
                if action == 'status':
                    self._cmd_entry_status()
                    return True
                print("❓ Unknown entry subcommand. Try: enable | disable | "
                      "clear | send | status")
                return True
        except Exception as e:
            print(f"❌ bridge command error: {e}")
            return True
        self._print_help()
        return True

    def _print_help(self):
        print("🌉 bridge commands:")
        print("  bridge list                       — pairs + state + terrain + modes")
        print("  bridge stats                      — current xy/val/conf per pair")
        print("  bridge trail <ent_idx>            — rolling xy history (chord-last style)")
        print("  bridge load <ent_idx> <atlas.pkl> [--nullclaw <binary>]")
        print("              [--nullclaw_live] [--context NAME]")
        print("  bridge unload <ent_idx>")
        print("  bridge enable|disable <ent_idx>")
        print("  bridge bind <ent_idx> <mode>      — webcam|ale|screencap|screen|viewer|idle|remote")
        print("  bridge unbind <ent_idx> [mode]")
        print("  ── voice (TTS, worker 1568) ──────────────────────────────────")
        print("  bridge voice enable|disable       — TTS synthesis on/off")
        print("  bridge voice link|unlink          — speaker playback on/off")
        print("  bridge voice speed <0.4..1.5>     — playback speed")
        print("  bridge voice say <text>           — manual test (bypasses debounce)")
        print("  bridge voice status               — show state, queue, last word")
        print("  bridge voice commands enable|disable — terrain-word CLI trigger on/off")
        print("  ── voice commands ────────────────────────────────────────────────")
        print("  bridge commands                   — mapped words, repeat counts, cooldown")
        print("  ── v139: entry buffer / spelling ─────────────────────────────────")
        print("  bridge entry enable|disable       — entry mode on/off (clears buffer on disable)")
        print("  bridge entry clear                — discard current buffer")
        print("  bridge entry send                 — submit buffer as command (debug/UI)")
        print("  bridge entry status               — buffer, render, timeouts, spikes")
        print(f"  Pairs (from UPE): {dict(self.PAIR_MAP)}")

    def _cmd_list(self):
        print("\n🌉 BRIDGES:")
        for ent_idx, dest_idx in self.PAIR_MAP.items():
            ent_name = self.magi_hive.upe.homes.get(ent_idx, {}).get('name', '?')
            dest_name = self.magi_hive.upe.homes.get(dest_idx, {}).get('name', '?')
            if ent_idx in self.terrains:
                state = "🟢ACTIVE" if self.enabled[ent_idx] else "🟡LOADED"
                terrain = self.terrain_names[ent_idx] or '-'
                modes = ','.join(self.bound_modes[ent_idx]) if self.bound_modes[ent_idx] else '-'
            else:
                state = "⚪ NO TERRAIN"
                terrain = "-"
                modes = "-"
            print(f"  ENT={ent_idx} {ent_name:<14} → DEST={dest_idx} {dest_name:<14} "
                  f"{state:<12} terrain={terrain:<20} modes={modes}")
        print()

    def _cmd_stats(self):
        print("\n🗺️  BRIDGE STATS:")
        print(f"{'ENT':<5} {'DEST':<5} {'TERRAIN':<20} {'STATE':<8} {'XY':<20} {'VAL':<6} {'CONF':<6}")
        print("-" * 80)
        for ent_idx, dest_idx in self.PAIR_MAP.items():
            if ent_idx not in self.terrains:
                continue
            state = "🟢" if self.enabled[ent_idx] else "🟡"
            xy = self.last_xy[ent_idx]
            xy_str = f"[{xy[0]:6.2f},{xy[1]:6.2f}]"
            val = self.last_value[dest_idx]
            conf = self.last_confidence[dest_idx]
            terrain = self.terrain_names[ent_idx] or '-'
            print(f"{ent_idx:<5} {dest_idx:<5} {terrain[:20]:<20} {state:<8} {xy_str:<20} "
                  f"{val:<6.2f} {conf:<6.2f}")
        print()

    def _cmd_trail(self, ent_idx):
        """Report xy trail summary + recent samples for one entrance."""
        if ent_idx not in self.trails:
            print(f"⚠️  {ent_idx} is not an entrance index"); return
        trail = list(self.trails[ent_idx])
        if not trail:
            print(f"\n🌉 TRAIL ENT={ent_idx}: empty (bridge not yet enabled or just loaded)")
            return
        xs = np.array([p[0] for p in trail])
        ys = np.array([p[1] for p in trail])
        terrain = self.terrains.get(ent_idx)
        if terrain is not None:
            x_span = terrain.x_range; y_span = terrain.y_range
            x_cov = (xs.max() - xs.min()) / max(x_span, 1e-8)
            y_cov = (ys.max() - ys.min()) / max(y_span, 1e-8)
        else:
            x_cov = y_cov = 0.0
        # unique-cell coverage at 32×32 resolution (cheap stuck-detector)
        if terrain is not None:
            gx = np.clip(((xs - terrain.x_min) / max(terrain.x_range, 1e-8) * 32).astype(int), 0, 31)
            gy = np.clip(((ys - terrain.y_min) / max(terrain.y_range, 1e-8) * 32).astype(int), 0, 31)
            unique_cells = len(set(zip(gx.tolist(), gy.tolist())))
            cell_coverage = unique_cells / (32 * 32)
        else:
            unique_cells = 0; cell_coverage = 0.0

        print(f"\n🌉 TRAIL ENT={ent_idx}: {len(trail)} samples "
              f"(maxlen={self.TRAIL_MAXLEN}, ~{self.TRAIL_MAXLEN/20:.0f}s at 20Hz)")
        print(f"  x: mean={xs.mean():+.3f} std={xs.std():.3f} "
              f"range=[{xs.min():+.2f},{xs.max():+.2f}] span_frac={x_cov:.1%}")
        print(f"  y: mean={ys.mean():+.3f} std={ys.std():.3f} "
              f"range=[{ys.min():+.2f},{ys.max():+.2f}] span_frac={y_cov:.1%}")
        print(f"  32×32 grid coverage: {unique_cells}/1024 cells = {cell_coverage:.1%}")
        # last 10 positions, tail end
        tail = trail[-10:]
        print(f"  last 10: " + " → ".join(f"({x:+.1f},{y:+.1f})" for x, y in tail))
        print()

    def _on_word_spoken(self, word: str):
        """v138: Called by BridgeVoice after each completed synthesis.

        Runs on the voice worker thread. Increments the repeat counter for
        this word and fires the mapped CLI command if the threshold is met
        within the sliding window. Each discrete playback event counts as one
        utterance — no 20Hz frame-rate noise, no dwell accumulation.
        """
        word_key = word.lower()
        if not self.commands_enabled or word_key not in self.command_map:
            return

        now = time.time()
        cfg = self.magi_hive.upe.km_config
        cooldown_s = cfg.get('bridge_command_cooldown_ms', 300000) / 1000.0

        if now - self._last_command_time < cooldown_s:
            return  # global cooldown still active

        timestamps = self._cmd_timestamps.setdefault(word_key, [])
        timestamps.append(now)
        window_s = cfg.get('bridge_command_window_ms', 5000) / 1000.0
        timestamps[:] = [t for t in timestamps if now - t <= window_s]

        threshold = cfg.get('bridge_command_repeat_threshold', 3)
        if len(timestamps) >= threshold:
            self._execute_command(self.command_map[word_key])
            self._last_command_time = now
            # v139 fix: only clear THIS word's timestamps, not all words.
            # The global cooldown (bridge_command_cooldown_ms, default 5min)
            # would invalidate other words' counts anyway via the line-1487
            # return, but if cooldown is reduced this becomes load-bearing —
            # parallel command-counting on different words must be preserved.
            self._cmd_timestamps.pop(word_key, None)

    def _execute_command(self, cmd_str: str):
        """v138: Route a fired bridge command into MaGiHive's command queue."""
        if hasattr(self.magi_hive, 'command_queue'):
            self.magi_hive.command_queue.put(cmd_str)
            print(f"🟢 Bridge command: {cmd_str}")
            self.voice.speak("ok", force=True)
        else:
            print(f"⚠️  Bridge command fired but command_queue unavailable: {cmd_str}")

    # ── v139: entry buffer / spelling bridge ──────────────────────────────

    def _impulse_s_filtered(self, idx: int, magnitude: float):
        """v139: Direct scalar spike for entry/send feedback.

        Bridge beacons normally write to vel_6d[4:5] (terrain confidence
        vibration). This helper writes to s_filtered instead — same channel
        MotorVoiceCoupling uses (line ~1002 in MaGi). Additive accumulation,
        not overwrite, so it composes safely with UPE's existing physics.

        Used to give MaGi distinct, scaled scalar feedback for entry tokens
        and send events on top of the normal terrain-confidence vibration.
        """
        if 0 <= idx < self.magi_hive.s_filtered.shape[0]:
            self.magi_hive.s_filtered[idx] += magnitude

    def _format_buffer_for_display(self) -> str:
        """v139.5: Buffer display is identical to the resolved form.

        Both the visual MaGi sees while typing and the audio that fires
        on Send come from the same joined string — see _resolve_buffer.
        Kept as a separate method for call-site clarity (callers in
        _entry_accumulate intend "show what's being composed" rather
        than "what gets sent"; semantically distinct, mechanically same).
        """
        return self._resolve_buffer()

    def _entry_buffer_full(self) -> bool:
        """v139: True if the entry buffer has reached its content-dependent cap.

        Numeric cap (default 3 → max value 999) when all tokens are digits,
        text cap (default 16) otherwise. When full, fire_beacons skips both
        speak() and _entry_accumulate() for new entry visits — MaGi gets
        complete silence on a rejected token, learning "I am full" through
        the absence of feedback rather than auto-flushing the buffer.
        """
        if not self._entry_buffer:
            return False
        cfg = self.magi_hive.upe.km_config
        all_digits = all(t.isdigit() for t in self._entry_buffer)
        max_len = (cfg.get('entry_max_numeric', 3) if all_digits
                   else cfg.get('entry_max_text', 16))
        return len(self._entry_buffer) >= max_len

    def _resolve_buffer(self) -> str:
        """v139: Convert the entry buffer to its spoken/submitted form.

        Tokens are joined raw — Kokoro TTS handles digit-to-spoken-number
        expansion natively, so "111" gets spoken as "one hundred eleven"
        without any Python-side conversion. Same string goes to both audio
        and command_queue.

        Examples:
            ['1','1','1']      → "111"        (Kokoro: "one hundred eleven")
            ['c','at']          → "cat"        (Kokoro: "cat")
            ['1','c','2']       → "1c2"        (Kokoro: best-effort)
            ['t','e','t','r','i','s']  → "tetris"  (matches command_map)
        """
        return ''.join(self._entry_buffer)

    def _entry_accumulate(self, word: str, ent_idx: int | None = None):
        """v139: Append a token to the shared entry buffer.

        Called from fire_beacons when MaGi visits a cell with type='entry'.
        Caller passes ent_idx so the s_filtered spike lands on the entrance
        worker that fired this event. UI/CLI callers can pass None and we
        fall back to the first entrance in PAIR_MAP.
        """
        if ent_idx is None:
            if not self.PAIR_MAP:
                return
            ent_idx = next(iter(self.PAIR_MAP))

        cfg = self.magi_hive.upe.km_config
        if not cfg.get('entry_enabled', True):
            return

        now = time.time()
        timeout_s = cfg.get('entry_timeout_ms', 10000) / 1000.0

        # Drop a stale buffer if the last token was beyond the timeout window.
        # The fire_beacons sweeper handles parked-MaGi case; this catches the
        # case where the next entry visit comes very late.
        if self._entry_buffer and (now - self._entry_last_token_time) > timeout_s:
            self._entry_buffer.clear()

        # v139: Cap-rejection. If buffer is at its content-dependent cap,
        # silently ignore the new token. No append, no spike, no visual
        # update, no timer reset — MaGi must wait for timeout, hit Send,
        # or visit a button cell to clear the buffer. fire_beacons already
        # checks _entry_buffer_full() before calling here so the speak()
        # call is also suppressed in the normal path; this guard catches
        # any caller that bypasses fire_beacons.
        if self._entry_buffer_full():
            return

        self._entry_buffer.append(word)
        self._entry_last_token_time = now

        # Update visual overlay — entry-persist mode (until=0 = no auto-clear).
        self._render_text = self._format_buffer_for_display()
        self._render_text_until = 0.0

        # Vibration spike scales with buffer depth so MaGi feels accumulation.
        spike = cfg.get('entry_token_spike', 50.0) * len(self._entry_buffer)
        self._impulse_s_filtered(ent_idx, spike)

        # v139: NO auto-send on cap. MaGi must visit a Send cell (or trigger
        # bridge entry send via CLI) to commit. The buffer stays full and
        # rejects new tokens until cleared by Send, timeout, or a button visit.

    def _entry_send(self, ent_idx: int | None = None, force_audio: bool = True):
        """v139: Resolve the entry buffer and submit it to command_queue.

        Three cases:
          - Empty buffer: speaks "send" with a distinct (200) spike. Acts
            like a button — MaGi can learn that send-on-empty is itself a
            meaningful event without yet typing anything.
          - Numeric buffer (digits only, max 3 → 999): joined raw and
            handed to Kokoro, which speaks "111" as "one hundred eleven".
            The joined string is also looked up in command_map in case
            someone wires "111" → cmd.
          - Text buffer: joined tokens ("cat") sent to TTS as-is. Matched
            against command_map for things like 'tetris' → 'mode ale tetris.bin'.

        Always clears _entry_buffer and sets _render_text_until so the
        visual overlay clears after the audio playback finishes.

        v139: Send is the ONLY way to commit a buffer. There is no auto-send
        on cap-hit — when the buffer fills (3 digits or 16 text tokens),
        new entry tokens are silently rejected and the buffer holds until
        Send, timeout, or a button visit clears it.

        v139.2: force_audio parameter controls whether the "send" or
        resolved-buffer audio bypasses the per-word cooldown.
          - force_audio=True  (CLI, manual): always speak. User-initiated
            sends should always be heard regardless of recent activity.
          - force_audio=False (terrain dwell): speak only if cooldown
            allows. If MaGi parks on a Send cell, only the first frame of
            dwell fires; subsequent frames within the 300ms cooldown are
            silent. No spike, no command, no audio — pure dwell rejection.
            This prevents Send-spam when MaGi happens to settle on the
            Send region.
        """
        if ent_idx is None:
            if not self.PAIR_MAP:
                print("⚠️  Bridge entry send: no pairs configured")
                return
            ent_idx = next(iter(self.PAIR_MAP))

        cfg = self.magi_hive.upe.km_config
        now = time.time()

        # v139.5: Atomically claim the buffer contents under lock so two
        # callers (e.g. terrain dwell + CLI manual send firing within
        # microseconds of each other) can't both see the same non-empty
        # buffer and double-submit command_queue.put. The lock is held
        # only for the buffer-snapshot-and-clear step; speak() and
        # command_queue.put happen after release so we don't block
        # other threads on TTS or queue I/O.
        with self._entry_lock:
            if not self._entry_buffer:
                # Empty buffer path — handled inside the lock to keep
                # the dwell-spam logic consistent.
                #
                # v139.6: Empty Send is AUDIO-FREE. MaGi gets the visual
                # flash and the spike (200) but no spoken word. Audio
                # is reserved exclusively for COMPOSED Sends (buffer
                # non-empty → speaks the resolved phrase). This makes
                # empty Send feel structurally incomplete: MaGi only
                # earns the audio reward when a real entry → Send loop
                # closes.
                #
                # v139.7: cooldown gating uses voice.is_active_or_pending()
                # in addition to the silent per-word cooldown. The dwell
                # is suppressed while ANY audio is in flight (a real
                # commit's "44" or "yyyss" still playing/queued, or a
                # button cell's word). This means empty Send doesn't
                # interrupt or overlap with composed Send audio — the
                # visual flash waits for the previous utterance to finish
                # before re-firing on continued dwell. The silent
                # cooldown still gates against pure dwell-spam (no audio
                # in flight), giving the same ~3Hz floor as before.
                if self.voice.is_active_or_pending():
                    return  # commit audio still playing — defer dwell flash
                spoken = self.voice.speak("send", force=force_audio, silent=True)
                if not spoken:
                    return  # silent cooldown rejected — silent dwell
                self._render_text = "send"
                self._render_text_until = now + 0.001
                spike = cfg.get('entry_send_empty_spike', 200.0)
                self._impulse_s_filtered(ent_idx, spike)
                print("📨 Bridge send (empty)")
                return

            # Snapshot the buffer and clear atomically. Whichever caller
            # grabs the lock first owns this commit; subsequent callers
            # arriving with a now-empty buffer fall through to the
            # empty-Send path on their next call.
            buffer_snapshot = self._entry_buffer[:]
            self._entry_buffer.clear()

        # Lock released. The rest is best-effort delivery of the
        # snapshotted buffer — no shared state mutated past this point
        # except _render_text (which is a single-write, last-writer-wins
        # field that is fine to update without the lock).
        resolved = ''.join(buffer_snapshot)
        cmd_lookup_key = resolved.lower()
        cmd = self.command_map.get(cmd_lookup_key, resolved)

        if hasattr(self.magi_hive, 'command_queue'):
            self.magi_hive.command_queue.put(cmd)
            print(f"📨 Bridge entry sent: '{resolved}' → command: {cmd}")
        else:
            print(f"⚠️  Bridge send fired but command_queue unavailable: {cmd}")

        # MaGi hears the resolved form — composed audio is the success signal.
        self.voice.speak(resolved, force=True)
        self._render_text = resolved
        # v139: sentinel — see comment above. Visual clears when audio ends.
        self._render_text_until = now + 0.001

        spike = cfg.get('entry_send_commit_spike', 300.0)
        self._impulse_s_filtered(ent_idx, spike)

    def _cmd_entry_status(self):
        """v139: Print entry buffer state, render text, and config."""
        cfg = self.magi_hive.upe.km_config
        enabled = cfg.get('entry_enabled', True)
        timeout_s = cfg.get('entry_timeout_ms', 10000) / 1000.0
        max_num = cfg.get('entry_max_numeric', 3)
        max_txt = cfg.get('entry_max_text', 16)
        token_spike = cfg.get('entry_token_spike', 50.0)
        empty_spike = cfg.get('entry_send_empty_spike', 200.0)
        commit_spike = cfg.get('entry_send_commit_spike', 300.0)
        voice_cooldown_ms = cfg.get('bridge_voice_cooldown_ms', 300)

        now = time.time()
        is_full = self._entry_buffer_full()
        if self._entry_buffer:
            age = now - self._entry_last_token_time
            remaining = max(0.0, timeout_s - age)
            buffer_str = ' '.join(self._entry_buffer)
            all_digits = all(t.isdigit() for t in self._entry_buffer)
            cap = max_num if all_digits else max_txt
            preview_resolved = self._resolve_buffer()
        else:
            age = 0.0
            remaining = 0.0
            buffer_str = '(empty)'
            cap = '-'
            preview_resolved = '(empty)'

        if self._render_text is None:
            render_state = '(none)'
        elif self._render_text_until > 0:
            voice_active = self.voice.is_active_or_pending()
            if voice_active:
                render_state = f"flash '{self._render_text}' (audio active)"
            else:
                render_state = f"flash '{self._render_text}' (audio ended — clears next frame)"
        else:
            render_state = f"persist '{self._render_text}'"

        cap_indicator = '  🔒 FULL (rejecting new tokens)' if is_full else ''

        print(f"\n📨 BRIDGE ENTRY: {'ENABLED' if enabled else 'DISABLED'}")
        print(f"   buffer:        {buffer_str}{cap_indicator}")
        print(f"   tokens:        {len(self._entry_buffer)}/{cap}"
              f" (numeric_cap={max_num}, text_cap={max_txt})")
        if self._entry_buffer:
            print(f"   age:           {age:.1f}s  (timeout in {remaining:.1f}s)")
            print(f"   would resolve: '{preview_resolved}'")
        print(f"   render:        {render_state}")
        print(f"   spikes:        token={token_spike:.0f}× depth, "
              f"send_empty={empty_spike:.0f}, send_commit={commit_spike:.0f}")
        print(f"   voice cooldown:{voice_cooldown_ms}ms (per-word debounce; "
              f"applies to button and entry cells alike)")
        print()

    def _cmd_commands_status(self):
        """v138: Print per-word repeat counts, cooldown state, and full command map.

        v139.4 fix: count timestamps freshly against the current 5-second
        window rather than reporting raw list length. The internal pruning
        in _on_word_spoken only runs when that word is next spoken, so
        stale entries can persist in self._cmd_timestamps for arbitrary
        time. The display should reflect MaGi's *current* progress toward
        each command's threshold, not an archive of all utterances.

        Also: during the 5-min global cooldown lock, the recent visits
        before the lock are shown as 0 — those counts are no longer
        relevant since no command can fire until the cooldown clears.
        """
        cfg       = self.magi_hive.upe.km_config
        enabled   = cfg.get('bridge_commands_enabled', False)
        cmap      = cfg.get('bridge_commands', {})
        threshold = cfg.get('bridge_command_repeat_threshold', 3)
        cooldown_s = cfg.get('bridge_command_cooldown_ms', 300000) / 1000.0
        window_s   = cfg.get('bridge_command_window_ms', 5000) / 1000.0
        now        = time.time()
        since_last = now - self._last_command_time
        locked     = since_last < cooldown_s

        print(f"\n🟢 BRIDGE VOICE COMMANDS: {'ENABLED' if enabled else 'DISABLED'}")
        print(f"   threshold={threshold}  window={window_s:.1f}s  cooldown={cooldown_s:.0f}s")
        print(f"   cooldown: {'🔒 LOCKED' if locked else '✅ ready'}"
              + (f"  ({cooldown_s - since_last:.1f}s remaining)" if locked else ""))
        if not cmap:
            print("   (no commands configured — add to km_config['bridge_commands'])")
        else:
            print(f"   {'WORD':<22} {'COUNT':>7}   CMD")
            print("   " + "-" * 62)
            for word, cmd in sorted(cmap.items()):
                # v139.4: count only timestamps inside the current window.
                # During cooldown lock the count is forced to 0 since
                # nothing can fire anyway — avoids confusing display of
                # stale "2/3" entries that imply almost-firing.
                if locked:
                    count = 0
                else:
                    raw = self._cmd_timestamps.get(word, [])
                    count = sum(1 for t in raw if now - t <= window_s)
                print(f"   {word:<22} {count:>5}/{threshold}   {cmd}")
        print()