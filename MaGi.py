
# MaGi_python (Malloy artificial Geometric intelligence)
# ------------------------------------------------------
# Author:  Brendan Malloy
# Year:    2025-2026
# Version: v67 (Memory-Safe Fibonacci Grids / Collision Sovereignty)

# Hardware-Embodied Geometric Intelligence Platform with Neural Control & Memory Systems.
# Exploring the emergence of cognitive architecture through hardware wobble, 
# prime-delay resonance, and hypersphere manifold dynamics.

# Novel Technologies Claimed (Prior Art 2025-2026):
# -------------------------------------------------
# 1. Hypersphere Black Hole Memory Deletion Worker: 
#    Geometric memory management using black hole physics principles for intelligent 
#    memory pruning with sensory feedback anchoring. Deletion actively improves 
#    memory structure through enhanced cosine similarity clustering.

# 2. Universal Plasticity Engine (UPE): 
#    Enables dynamic cognitive reconfiguration by allowing a black hole worker to 
#    move control/voice workers within the hypersphere while maintaining collision sovereignty.

# 3. Collision Sovereignty (v5.3 Bumper): 
#    Deterministic geometric "bumper" preventing worker ghosting. Enforces a 
#    minimum 0.1 radian separation to preserve action identity and prevent manifold collapse.

# 4. Artificial Personal Space: 
#    First documented implementation of non-overlapping cognitive workers in 
#    hypersphere manifolds, preventing "dead neuron" phenomena through geometric volume constraints.

# 5. Fibonacci Grid Video Processing: 
#    Multi-scale visual attention using golden ratio proportions (5×3, 8×5, 13×8, 21×13).

# 6. Neural Deadzone Control: 
#    Unipolar and bipolar deadzone logic for stable AI-to-system control.


# License & Usage Terms
# ---------------------
# 1. Academic & Non-Profit Use:
#    - Licensed under a GPL-style open license for **academic and non-profit research** only.
#    - You may use, modify, and distribute this software for **educational purposes**
#      provided that this notice and attribution remain intact.

# 2. Commercial Use & Licensing:
#    - Commercial or for-profit use requires a **perpetual license** from the author.
#    - Licensing tiers (USD):
#        • Individual / Startup (< $10M annual revenue): $5,000
#        • Mid-size Organization (< $100M annual revenue): $50,000
#        • Large Organization / Enterprise (≥ $100M annual revenue): $500,000
#    - Written permission is required before deployment or integration into closed systems.

# 3. Disclaimer:
#    - Software provided **"as is"**. Author assumes **no liability** for damages or data loss.
#    - MaGi is **experimental research software**, not certified for safety-critical control.

# 4. Citation / Attribution:
#    - Any public use or publication must cite:
#      "MaGi_python Hardware-Embodied Cognitive Architecture Platform, Brendan Malloy, 2025"

# ------------------------------------------------------
# Contact: https://github.com/bmalloy-224/MaGi_python/issues/1


import torch
import torch.nn.functional as F
import numpy as np
import math
import time
import threading
from collections import deque
import queue
import cv2
import pyaudio
import serial
import os
import sys
import select
import socket
import struct
import json
import tkinter as tk
from PIL import ImageGrab
import platform
import mss

from adaptive_scaler import AdaptiveScaler
from bridge import BridgeController

try:
    from magi_cuda_loader_v117 import MagiCUDAv117
    _MAGI_CUDA_V117_AVAILABLE = True
except ImportError:
    _MAGI_CUDA_V117_AVAILABLE = False


# ==========================================
# 📌 CONFIGURATION
# ==========================================
TARGET_PORT = 'COM9'
BAUD_RATE = 115200
# was 115200
NUM_WORKERS = 1570  # v136: +1 for Bridge Visual Word worker (1569) — was 1569
# was 8100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEMORY_FILE = "magi_torus_memory.pt"          # v130: 6D toroidal main bank
LEGACY_MEMORY_FILE = "magi_memory.pt"          # v99 and earlier — used for one-time N seeding
V129_MEMORY_FILE = "magi_v10x_memory.pt"       # v100-v129 main bank — auto-converted on first load
MAGI_EPOCH = 1735689600.0               # Jan 1 2025 00:00:00 UTC — fixed calendar anchor

# Physics Constants
CHILD_SENSITIVITY = 0.47
YOUTH_GAIN = 1.0
ADULT_THRESHOLD = 0.3
ELDER_TIME_CONSTANT = 0.95
HB_SINE_SCALE = 500.0
# v131: Cross-tension removed from freq/delay pipeline (inflationary in log space)
# These constants are kept for reference but NOT applied to freq/delay dims
TENSION_FREQ_COUPLING = 0.05   # LEGACY — not used in v131 freq/delay pipeline
COHERENCE_DELAY_COUPLING = 0.1  # LEGACY — not used in v131 freq/delay pipeline
ELASTICITY = 0.92 
# v131: Phase-space force constants (replace Hz-space constants)
GRAVITY_PHASE_K    = 0.1    # Gravity pull strength in phase space (tune from chord convergence)
DELAY_PHASE_K      = 0.001  # Coherence→delay coupling in phase space (centered, no drift)
VEL_PHASE_CLAMP    = math.pi  # Max phase momentum (Nyquist-safe: half-wrap per step)
FREQ_SCALE_MAX     = 500.0    # Display cap for lens entropy scaling (absolute Hz)

# Memory Configuration
STORE_COHERENCE_STABLE = 0.01
STORE_COHERENCE_PEAK = 0.85
MAX_MEMORIES = 3000000 
# v131: Log-wrapped torus mapping — one octave = one full wrap (2π)
MIN_FREQ       = 0.01        # Hz — lower anchor
MIN_DELAY      = 0.1         # ms — lower anchor
LOG_FREQ_STEP  = math.log(2) # one wrap = octave (×2)
LOG_DELAY_STEP = math.log(2) # one wrap = octave (×2)
MAX_FREQ       = 500.0       # LEGACY — display/telemetry only, not a physics bound
MAX_DELAY      = 20000.0     # LEGACY — display/telemetry only, not a physics bound
TWO_PI         = 2 * math.pi
MEMORY_SAMPLE_SIZE = 10000          # memories sampled per retrieve_gravity call
SOFTMAX_TEMP       = 5.0            # sharpens attractor weights (1=flat, higher=sharper)
REMOTE_BATTERY_LOW_PCT = 10.0       # fall back to webcam if battery drops below this

# v141: Remote microphone + DOA (worker streams over UDP)
# Worker sends paInt16 stereo @ 16 kHz, CHUNK=1024 → 4096 bytes/packet.
# DOA stream is 4-byte packets: struct '<HH' = (speech 0/1, doa degrees).
REMOTE_AUDIO_PORT = 12345
REMOTE_DOA_PORT   = 12346
REMOTE_AUDIO_RATE = 16000           # informational; downstream RMS is rate-agnostic
REMOTE_AUDIO_CHANNELS = 2           # informational; RMS works on flat int16 buffer

# v133: Robot Arm Configuration
ROBOT_UDP_PORT = 5005               # Genesis sim listens here (no conflict with remote UDP:5003)
ROBOT_STEP_SIZE = 0.01              # meters per activation per step
ROBOT_WRIST_STEP = 0.05             # radians per activation per step
ROBOT_WORKSPACE = {                 # clamp bounds (meters from origin)
    'x': (-0.5, 0.5),
    'y': (-0.5, 0.5),
    'z': (0.0, 0.5),
    'wrist_rot': (-3.14, 3.14),     # radians
    'wrist_tilt': (-1.57, 1.57),
}
ROBOT_GRIPPER_OPEN  = 0.5           # matches so100.py GRIPPER_OPEN
ROBOT_GRIPPER_CLOSE = -0.1          # matches so100.py GRIPPER_CLOSE

# ── v131: Log-wrapped mapping functions ──────────────────────────────────────
# Workers use wrapped phase [0, 2π) + integer wrap count for absolute Hz.
# Memory banks store UNWRAPPED log coordinate for octave-aware matching.

def freq_to_phase(f):
    """Hz → wrapped phase [0, 2π) on log torus."""
    if f <= MIN_FREQ: return 0.0
    return ((math.log(f / MIN_FREQ) / LOG_FREQ_STEP) * TWO_PI) % TWO_PI

def phase_to_freq(phase, wrap=0):
    """Wrapped phase + wrap count → absolute Hz."""
    return MIN_FREQ * math.exp(((phase / TWO_PI) + wrap) * LOG_FREQ_STEP)

def freq_to_log_coord(f):
    """Hz → unwrapped log coordinate (for memory storage). NOT modded."""
    if f <= MIN_FREQ: return 0.0
    return math.log(f / MIN_FREQ) / LOG_FREQ_STEP

def freq_to_phase_t(f):
    """Tensor: Hz → wrapped phase [0, 2π)."""
    return ((torch.log(f.clamp(min=MIN_FREQ) / MIN_FREQ) / LOG_FREQ_STEP) * TWO_PI) % TWO_PI

def phase_to_freq_t(phase, wrap=None):
    """Tensor: wrapped phase + optional wrap count → absolute Hz."""
    if wrap is None:
        return MIN_FREQ * torch.exp((phase / TWO_PI) * LOG_FREQ_STEP)
    return MIN_FREQ * torch.exp(((phase / TWO_PI) + wrap.float()) * LOG_FREQ_STEP)

def freq_to_log_coord_t(f):
    """Tensor: Hz → unwrapped log coordinate (for memory storage)."""
    return torch.log(f.clamp(min=MIN_FREQ) / MIN_FREQ) / LOG_FREQ_STEP

def delay_to_phase(d):
    """ms → wrapped phase [0, 2π) on log torus."""
    if d <= MIN_DELAY: return 0.0
    return ((math.log(d / MIN_DELAY) / LOG_DELAY_STEP) * TWO_PI) % TWO_PI

def phase_to_delay(phase, wrap=0):
    """Wrapped phase + wrap count → absolute ms."""
    return MIN_DELAY * math.exp(((phase / TWO_PI) + wrap) * LOG_DELAY_STEP)

def delay_to_log_coord(d):
    """ms → unwrapped log coordinate (for memory storage). NOT modded."""
    if d <= MIN_DELAY: return 0.0
    return math.log(d / MIN_DELAY) / LOG_DELAY_STEP

def delay_to_phase_t(d):
    """Tensor: ms → wrapped phase [0, 2π)."""
    return ((torch.log(d.clamp(min=MIN_DELAY) / MIN_DELAY) / LOG_DELAY_STEP) * TWO_PI) % TWO_PI

def phase_to_delay_t(phase, wrap=None):
    """Tensor: wrapped phase + optional wrap count → absolute ms."""
    if wrap is None:
        return MIN_DELAY * torch.exp((phase / TWO_PI) * LOG_DELAY_STEP)
    return MIN_DELAY * torch.exp(((phase / TWO_PI) + wrap.float()) * LOG_DELAY_STEP)

def delay_to_log_coord_t(d):
    """Tensor: ms → unwrapped log coordinate (for memory storage)."""
    return torch.log(d.clamp(min=MIN_DELAY) / MIN_DELAY) / LOG_DELAY_STEP

# ==========================================
# v100: BAND SYSTEM
# EEG boundaries reflect synaptic integration time limits.
# Bands use absolute Hz from self.freq (wrap-aware property).
# ==========================================
BAND_BASELINES = {
    'delta': {'child': 0.1, 'youth': 0.2, 'adult': 0.3, 'elder': 0.9},
    'theta': {'child': 0.3, 'youth': 0.4, 'adult': 0.5, 'elder': 0.4},
    'alpha': {'child': 0.3, 'youth': 0.4, 'adult': 0.5, 'elder': 0.4},
    'beta':  {'child': 0.5, 'youth': 0.7, 'adult': 0.8, 'elder': 0.2},
    'gamma': {'child': 0.9, 'youth': 1.0, 'adult': 0.6, 'elder': 0.1},
}

def get_band(hz: float) -> str:
    if hz < 4.0:  return 'delta'
    if hz < 8.0:  return 'theta'
    if hz < 13.0: return 'alpha'
    if hz < 30.0: return 'beta'
    return 'gamma'

# ALE Configuration
ALE_ACTION_MAP = {
    0: 'NOOP', 1: 'FIRE', 2: 'UP', 3: 'RIGHT', 4: 'LEFT', 5: 'DOWN',
    6: 'UPRIGHT', 7: 'UPLEFT', 8: 'DOWNRIGHT', 9: 'DOWNLEFT',
    10: 'UPFIRE', 11: 'RIGHTFIRE', 12: 'LEFTFIRE', 13: 'DOWNFIRE',
    14: 'UPRIGHTFIRE', 15: 'UPLEFTFIRE', 16: 'DOWNRIGHTFIRE', 17: 'DOWNLEFTFIRE'
}


# ==========================================
# 🛠️ INFRASTRUCTURE & I/O
# ==========================================
# ==========================================
# 🛠️ OPTIMIZED ComSerial (MINIMAL UPGRADE)
# ==========================================
class ComSerial:
    def __init__(self, port, baud, enabled=True):
        self.ser = None
        self.enabled = enabled
        self.virtual_mode = False
        self.write_interval = 0.05  # 20Hz max (was every frame)
        self.last_write = 0
        self.queue = []
        
        if not enabled:
            print(f"⏭️  Serial disabled (was {port} at {baud} baud)")
            return
            
        try:
            # KEY CHANGE 1: Lower baud, add timeout
            self.ser = serial.Serial(
                port, 
                baud, 
                timeout=0.01,        # Never block >10ms (was 1 second!)
                write_timeout=0.01   # Same for writes
            )
            print(f"✅ Serial port {port} opened at {baud} baud (non-blocking)")
        except serial.SerialException as e:
            print(f"⚠️  Could not open {port}: {e}")
            print("👻 Running in VIRTUAL mode (no hardware)")
            self.virtual_mode = True
    
    def println(self, text):
        # KEY CHANGE 2: Skip if disabled
        if not self.enabled:
            return
            
        current = time.time()
        
        # KEY CHANGE 3: Rate limiting
        if current - self.last_write < self.write_interval:
            self.queue.append(str(text) + '\n')
            if len(self.queue) > 100:   # cap — prevent unbounded growth
                self.queue = self.queue[-50:]
            return
            
        # KEY CHANGE 4: Batch writes
        if self.queue:
            batch = ''.join(self.queue) + str(text) + '\n'
            self.queue = []
        else:
            batch = str(text) + '\n'
        
        # Same write logic (but now rate-limited)
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(batch.encode('ascii'))
                self.last_write = current
            except Exception:
                # Silent fail (same as before)
                pass
    
    def flush(self):
        """Optional: Send any queued data"""
        if self.queue and self.ser and self.ser.is_open:
            try:
                self.ser.write(''.join(self.queue).encode('ascii'))
                self.queue = []
            except:
                pass


# ==========================================
# 📡 RemoteCommandSender — sends pan/tilt commands to Pi over UDP
class RemoteCommandSender:
    """
    Sends single-byte pan/tilt commands to pi_server.py via UDP port 5003.
    Command bytes: L=pan left  R=pan right  U=tilt up  D=tilt down  C=center
    B=battery query → Pi responds with 9-byte pack('>fff', pct, volts, amps)
    Polls battery every BATTERY_POLL_SEC seconds in background.
    """
    CMD_PORT         = 5003
    BATTERY_POLL_SEC = 30    # how often to request battery status from Pi

    def __init__(self, pi_ip):
        self.pi_ip          = pi_ip
        self.sock           = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(2.0)
        self.battery_pct    = -1.0   # -1 = not yet read
        self.battery_volts  = -1.0
        self.battery_amps   = -1.0
        self._batt_lock     = threading.Lock()
        self._poll_running  = True
        self._poll_thread   = threading.Thread(target=self._battery_loop, daemon=True)
        self._poll_thread.start()
        print(f"📡 RemoteCommandSender → {pi_ip}:{self.CMD_PORT}")

    def _send(self, cmd_byte):
        try:
            self.sock.sendto(cmd_byte, (self.pi_ip, self.CMD_PORT))
        except OSError:
            pass

    def request_battery(self):
        try:
            self.sock.sendto(b'B', (self.pi_ip, self.CMD_PORT))
            data, _ = self.sock.recvfrom(16)
            if len(data) >= 12:                    # ← Fixed length check
                pct, v, a = struct.unpack('>fff', data)  # ← No slicing
                with self._batt_lock:
                    self.battery_pct   = pct
                    self.battery_volts = v
                    self.battery_amps  = a
                return pct, v, a
        except Exception:
            pass
        return None

    def _battery_loop(self):
        while self._poll_running:
            self.request_battery()
            time.sleep(self.BATTERY_POLL_SEC)

    def get_battery(self):
        """Return cached (pct, volts, amps) — never blocks."""
        with self._batt_lock:
            return self.battery_pct, self.battery_volts, self.battery_amps

    def pan_left(self):  self._send(b'L')
    def pan_right(self): self._send(b'R')
    def tilt_up(self):   self._send(b'U')
    def tilt_down(self): self._send(b'D')
    def center(self):    self._send(b'C')

    def close(self):
        self._poll_running = False
        self.sock.close()


# ==========================================
# 🦾 RobotArmSender — sends position/gripper commands to Genesis sim over UDP
# ==========================================
class RobotArmSender:
    """
    Sends JSON position + gripper commands to Genesis sim over UDP:5005.
    Maintains a running target position that gets nudged by worker activations.
    MaGi owns the kinematics — sim just receives absolute targets.

    Protocol (matches so100.py UDPServer):
      Position: {"mode": "pos", "pos": [x, y, z], "wrist_rot": r, "wrist_tilt": t}
      Gripper:  {"mode": "gripper", "state": "open"|"close"}
    """
    def __init__(self, sim_ip, port=ROBOT_UDP_PORT):
        self.sim_ip = sim_ip
        self.port   = port
        self.sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Running target state
        self.target_pos    = [0.2, 0.0, 0.2]   # start: safe neutral position
        self.wrist_rot     = 0.0
        self.wrist_tilt    = 0.0
        self.gripper_state = 'open'
        print(f"🦾 RobotArmSender → {sim_ip}:{port}")

    def nudge(self, dx=0.0, dy=0.0, dz=0.0, d_rot=0.0, d_tilt=0.0):
        """Accumulate position deltas, clamp to workspace, send."""
        self.target_pos[0] = max(ROBOT_WORKSPACE['x'][0],
                                 min(ROBOT_WORKSPACE['x'][1],
                                     self.target_pos[0] + dx))
        self.target_pos[1] = max(ROBOT_WORKSPACE['y'][0],
                                 min(ROBOT_WORKSPACE['y'][1],
                                     self.target_pos[1] + dy))
        self.target_pos[2] = max(ROBOT_WORKSPACE['z'][0],
                                 min(ROBOT_WORKSPACE['z'][1],
                                     self.target_pos[2] + dz))
        self.wrist_rot  = max(ROBOT_WORKSPACE['wrist_rot'][0],
                              min(ROBOT_WORKSPACE['wrist_rot'][1],
                                  self.wrist_rot + d_rot))
        self.wrist_tilt = max(ROBOT_WORKSPACE['wrist_tilt'][0],
                              min(ROBOT_WORKSPACE['wrist_tilt'][1],
                                  self.wrist_tilt + d_tilt))
        self._send_pos()

    def set_gripper(self, is_positive):
        """Edge-triggered gripper: positive scaler = open, negative = close."""
        new_state = 'open' if is_positive else 'close'
        if new_state != self.gripper_state:
            self.gripper_state = new_state
            cmd = json.dumps({"mode": "gripper", "state": new_state})
            try:
                self.sock.sendto(cmd.encode(), (self.sim_ip, self.port))
            except OSError:
                pass

    def _send_pos(self):
        cmd = json.dumps({
            "mode": "pos",
            "pos": [round(p, 4) for p in self.target_pos],
            "wrist_rot": round(self.wrist_rot, 4),
            "wrist_tilt": round(self.wrist_tilt, 4),
        })
        try:
            self.sock.sendto(cmd.encode(), (self.sim_ip, self.port))
        except OSError:
            pass

    def close(self):
        self.sock.close()
        print("🦾 RobotArmSender closed")


# UPE
# ========================================== 


class UniversalPlasticityEngine:
    def __init__(self, device='cuda'):
        self.dev = device
        self.file_path = "motor_voice_map.pth"
        self.bh_idx = 1549
        self.bumper_radius    = 2.0   # Worker-worker L2 min separation = bumper_radius * 2
        self.bumper_dim_min   = 1.2    # Per-dim minimum separation — catches collapsed dims L2 misses
        
        # BASELINE: Hardened Geometry (Radius 2.15 across all axes)
        self.baseline = {
            1542: {'name': 'LEFT',  'type': 'ale',   'r': 2.15, 'home': [1.1, 0.9, -2.5,  0.5]},
            1543: {'name': 'RIGHT', 'type': 'ale',   'r': 2.15, 'home': [1.1, 0.9,  2.5, -0.5]},
            1544: {'name': 'FIRE',  'type': 'ale',   'r': 2.15, 'home': [1.1, 0.9,  0.5, -2.5]},
            1545: {'name': 'UP',    'type': 'ale',   'r': 2.15, 'home': [1.1, 0.9,  0.5,  2.5]},
            1546: {'name': 'DOWN',  'type': 'ale',   'r': 2.15, 'home': [1.1, 0.9, -1.0, -2.0]},
            1547: {'name': 'NOOP',  'type': 'ale',   'r': 2.15, 'home': [1.1, 0.9, -1.0,  1.0]},
            1548: {'name': 'VOICE', 'type': 'voice', 'r': 2.15, 'home': [4.71, 3.14, 1.57, 0.0]},
            # v133: Robot arm workers — Main bank, full UPE experience
            # Home positions spread across different quadrants from ALE workers
            1558: {'name': 'ARM_X',    'type': 'robot', 'r': 2.15, 'home': [2.5, 0.5, -1.0,  1.5]},
            1559: {'name': 'ARM_Y',    'type': 'robot', 'r': 2.15, 'home': [0.5, 2.5,  1.5, -1.0]},
            1560: {'name': 'ARM_Z',    'type': 'robot', 'r': 2.15, 'home': [2.0, 2.0, -0.5,  0.5]},
            1561: {'name': 'ARM_ROT',  'type': 'robot', 'r': 2.15, 'home': [0.5, 1.5,  2.5, -0.5]},
            1562: {'name': 'ARM_TILT', 'type': 'robot', 'r': 2.15, 'home': [1.5, 0.5,  0.5,  2.5]},
            1563: {'name': 'GRIPPER',  'type': 'robot', 'r': 2.15, 'home': [3.0, 3.0,  0.0,  0.0]},
            # v135: Resonance Bridge — two pairs. Name pattern BRIDGE<id>_<ENT|DEST>
            # encodes pairing; BridgeController parses names at startup. No extra
            # UPE schema fields needed; save/load glue unchanged.
            1564: {'name': 'BRIDGE0_ENT',  'type': 'bridge', 'r': 2.15, 'home': [0.0, 0.0,  1.5, -1.5]},
            1565: {'name': 'BRIDGE0_DEST', 'type': 'bridge', 'r': 2.15, 'home': [0.0, 0.0,  1.5, -1.5]},
            1566: {'name': 'BRIDGE1_ENT',  'type': 'bridge', 'r': 2.15, 'home': [0.0, 0.0, -1.5,  1.5]},
            1567: {'name': 'BRIDGE1_DEST', 'type': 'bridge', 'r': 2.15, 'home': [0.0, 0.0, -1.5,  1.5]},
        }
        
        # ── Kinetic Manifold configuration (v126) ──────────────────────────────
        # Single source of truth — access via magi.upe.km_config everywhere.
        self.km_config = {


            # Phase 1: ALE Vibration Beacons
            'ale_vib_strength':       0.002,   # amplitude ceiling
            'ale_vib_clamp':          0.005,   # per-step vel_s clamp
            'ale_vib_omega_base':     0.5,     # base Hz for all beacons
            'ale_omega_offsets': {             # per-worker freq spread (deterministic)
                1542:  0.00,   # LEFT
                1543:  0.02,   # RIGHT
                1544:  0.04,   # FIRE
                1545: -0.02,   # UP
                1546: -0.04,   # DOWN
                1547:  0.01,   # NOOP
            },

            # Phase 2: Dream Mirror worker indices
            'dream_n_reader_idx':      1552,
            'dream_main_anchor_idx':   1553,

            # Drift (lazy river)
            'dream_drift_speed':       0.005,   # v128: was 0.001 — increased for more active exploration

            # v128: Drift pair impulse
            'drift_impulse_gain':           0.02,
            'drift_impulse_access_scale':   0.01,
            'drift_impulse_tension_scale':  0.5,

            # v128: Physics pair impulse
            'physics_impulse_gain':         0.02,
            'physics_impulse_access_scale': 0.01,
            'physics_impulse_tension_scale':0.5,
            'physics_drift_decay': 0.995,            # per‑step decay factor (half‑life ~138 steps)
            'physics_lazy_river_strength': 0.0005,   # weak background drift when drift is gone

            # v128: General persistent drift strength (scaled by access count)
            'main_drift_strength':          0.001,
            'dream_n_drift_omega':     0.10,   # N-Reader base drift frequency
            'dream_m_drift_omega':     0.07,   # Main-Anchor base drift frequency

            # Cross-coupling
            'dream_coupling_strength': 0.003,
            'dream_vib_strength':      0.002,
            'dream_memory_boost':      0.5,    # access count boost per touch

            # Detection radii
            'dream_n_proximity_radius':   0.25,   # N bank (sparser)
            'dream_main_proximity_radius': 0.15,  # main bank (denser)

            # Emergent drift strength (mirrors BH eps system)
            'dream_eps_floor':         1e-4,
            'dream_eps_max':           5e-2,
            'dream_input_sensitivity': 0.01,   # how input energy scales drift

            # Sprint & kick
            'dream_sprint_multiplier': 5.0,    # speed boost on N match
            'dream_reciprocal_kick':   0.15,   # phase kick strength on Main match
            'dream_clamp':             0.005,  # velocity norm clamp for dream workers

            # Chord (Teleport) Pair
            'chord_n_reader_idx':       1554,
            'chord_main_anchor_idx':    1555,
            'chord_teleport_cooldown':  30,
            'chord_boost_amount':       2.0,

            # Physics (Lens-Driven) Pair
            'physics_n_reader_idx':     1556,
            'physics_main_anchor_idx':  1557,
            'physics_force_gain':       0.05,
            'physics_spring_k':         0.02,
            'physics_damping':          0.90,
            'physics_clamp':            0.08,
            'physics_proximity_radius': 0.20,

            # v133: Robot Arm Vibration Beacons
            'robot_vib_strength':     0.002,   # amplitude ceiling (same as ALE default)
            'robot_vib_clamp':        0.005,   # per-step vel_s clamp
            'robot_vib_omega_base':   0.4,     # slightly different base Hz from ALE (0.5)
            'robot_omega_offsets': {            # per-worker freq spread (deterministic)
                1558:  0.00,   # ARM_X
                1559:  0.03,   # ARM_Y
                1560: -0.03,   # ARM_Z
                1561:  0.05,   # ARM_ROT
                1562: -0.05,   # ARM_TILT
                1563:  0.02,   # GRIPPER
            },

            # v135: Resonance Bridge — beacons on both entrance and destination
            'bridge_vib_strength':     0.002,
            'bridge_vib_clamp':        0.005,
            'bridge_vib_omega_base':   0.3,
            'bridge_omega_offsets':    {},  # optional per-idx offsets; empty = all
                                            # beacons share omega_base. UPE bumper
                                            # keeps workers phase-separated; add
                                            # per-idx offsets here if phase-lock
                                            # ever observed empirically.

            # v136 rev28: Bridge Voice (TTS) — internal audio-type worker 1568.
            # Structurally identical to audio workers (not UPE-managed).
            # Kokoro synthesizes words from word-labeled terrain cells when
            # entrance lands there with confidence > 0. Audio energy is
            # EMA-smoothed and injected into inputs[1568] every frame like
            # audio_val feeds workers 948-1462.
            'bridge_voice_enabled':        False,      # default off
            'bridge_voice_linked':         False,      # speaker output toggle
            'bridge_voice_name':           'af_heart', # Kokoro voice
            'bridge_voice_speed':          1.5,        # clamp [0.4, 1.5]
            'bridge_voice_energy_scale':   50.0,       # tune up if 1568 too quiet
            'bridge_voice_energy_alpha':   0.15,       # EMA smoothing factor
            'bridge_voice_cooldown_ms':    300,        # per-word dedupe window
            'bridge_voice_queue_size':     4,          # bounded synthesis queue

            # v136 rev28: Bridge Visual Word (worker 1569) — image of the word
            # currently being spoken, rendered as white text on black canvas.
            # White-pixel-fraction (typically 2-8%) is the "visual energy."
            # Gated on audio playback — worker 1569 fires only while voice is
            # speaking. Same conservative scale start as voice (50.0).
            'bridge_visual_word_energy_scale':  50.0,

            # v138: Bridge voice commands — terrain-word → CLI command trigger.
            # MaGi must speak the same word 3× within 5 s to fire a command.
            # A 5-minute global cooldown prevents any further firing after that.
            # All words are lowercased before lookup so atlas case doesn't matter.
            'bridge_commands_enabled':         False,
            'bridge_commands': {
                # modes
                'webcam': 'mode webcam',
                'remote': 'mode remote 192.168.0.140',

                # viewer
                'images': 'mode viewer',
                'image sequence': 'mode viewer cetacean',
                'puzzle image': 'mode viewer test',
                'wave': 'mode viewer wave',
                'space': 'mode viewer space',
                'flotsam': 'mode viewer flotsam',
                'tuesday book': 'mode viewer tuesday',
                '3 pigs book': 'mode viewer 3pigs',
                'flowers book': 'mode viewer flowers',
                'robot baby book': 'mode viewer robobaby',
                'ocean wave': 'mode viewer realwave',
                'magi bridge map': 'mode viewer map',
                'mirror mirror book': 'mode viewer mirrormirror',
                'the arrival book': 'mode viewer thearrival',
                'one fish two fish book': 'mode viewer onefishtwofish',

                # ALE games
                'othello': 'mode ale othello.bin',
                'tetris': 'mode ale tetris.bin',
                'cube game': 'mode ale cube.bin',
                'ms pacman': 'mode ale mrspacman.bin',
                'freeway': 'mode ale freeway.bin',
                "montezuma's revenge": 'mode ale montezuma.bin',
                'pitfall': 'mode ale pitfall.bin',
                'solaris': 'mode ale solaris.bin',
                'pinball': 'mode ale pinball.bin',
            },
            'bridge_command_repeat_threshold':  3,       # utterances needed
            'bridge_command_window_ms':         8000,    # sliding window (ms)
            'bridge_command_cooldown_ms':       300000,  # lockout after fire (ms)

            # v139: Entry buffer / spelling bridge.
            # Cells in the atlas with type='entry' accumulate tokens into a
            # shared buffer; type='send' cells resolve and submit. The buffer
            # auto-clears on Send, cap-hit, or idle timeout. Visual feedback
            # rides on the existing worker 1569; vibration uses scaled spikes
            # into s_filtered that build with buffer depth.
            'entry_enabled':            True,
            'entry_timeout_ms':         20000,    # 10s idle → clear buffer
            'entry_max_numeric':        3,        # 3 digits → max value 999
            'entry_max_text':           16,       # 16 tokens → text cap
            'entry_token_spike':        50.0,     # × buffer depth on each token
            'entry_send_empty_spike':   200.0,    # spike on send with empty buffer
            'entry_send_commit_spike':  300.0,    # spike on send with content

            # v139.4: Multi-word visual cycling. When the visual worker
            # (1569) renders a phrase containing spaces (e.g. "Image
            # Sequence", "Cube Game", or send-flash "one hundred eleven"),
            # cycle through one token at a time at full font size rather
            # than squishing the whole phrase onto one canvas. Each token
            # holds for visual_cycle_ms before advancing to the next.
            # The audio still speaks the full phrase uninterrupted.
            'visual_cycle_ms':          500,      # ms per token (multi-word cells)
        }

        # UPE physics: Instant Snap + Long-Term Persistence
        self.home_drift_strength = 0.15      # High impact contact
        self.home_drift_damping = 0.95       
        self.window_duration = 60 * 60   # 8-Hour "Memory Lock"
        self.accumulation_threshold = 0.02   # Fast trigger
        self.pressure_decay_halflife = 8 * 60 * 60  # 24-Hour Persistence
        
        # Velocity impulse feedback: Binary Punch
        self.velocity_impulse_gain = 2.5     
        self.max_impulse = 0.5               
        self.min_impulse = 0.05              
        
        # Sensory feedback
        self.sensory_feedback_gain = 150.0
        
        # Tracking
        # deque(maxlen) bounds history — prevents O(time×workers) growth
        self.pressure_history = {}
        self.last_applied = {}
        self.last_pressure = {}  # For pressure delta calculation
        self.homes = self._load_or_init()
        self.steps_since_save = 0
        self.last_saved_homes = None

    def _load_or_init(self):
        """Load saved home positions or initialize from baseline.
        v131: Separates metadata (_freq_wrap etc.) from worker homes."""
        self._saved_metadata = {}  # v131: wrap counters etc.
        if os.path.exists(self.file_path):
            saved = torch.load(self.file_path, map_location=self.dev)
            # Separate metadata keys (prefixed with _) from worker homes
            homes = {}
            for k, v in saved.items():
                if isinstance(k, str) and k.startswith('_'):
                    self._saved_metadata[k] = v
                elif isinstance(v, dict) and 'home' in v:
                    homes[k] = v
                # else: skip unrecognized entries
            # Merge any new baseline entries missing from save (e.g. robot workers added in v133)
            for idx, data in self.baseline.items():
                if idx not in homes:
                    homes[idx] = {
                        'home': torch.tensor(data['home'], device=self.dev),
                        'type': data['type'],
                        'r_target': data['r'],
                        'name': data['name']
                    }
                    print(f"  🆕 UPE: Added new worker {data['name']} [{idx}] from baseline")
            print(f"✅ UPE: Loaded saved home positions")
            return homes
        
        # Initialize from baseline
        homes = {}
        for idx, data in self.baseline.items():
            homes[idx] = {
                'home': torch.tensor(data['home'], device=self.dev),
                'type': data['type'],
                'r_target': data['r'],
                'name': data['name']
            }
        print(f"🏠 UPE: Initialized {len(homes)} home positions from baseline")
        return homes

    def _wrapped_difference(self, a, b):
        """Return a-b wrapped to [-π, π] for all dims (now 6D)."""
        diff = a - b
        diff = torch.where(diff > math.pi, diff - 2*math.pi, diff)
        diff = torch.where(diff < -math.pi, diff + 2*math.pi, diff)
        return diff

    def apply_singularity_bumper(self, magi_hive):
        """
        🛡️ TWO-STAGE BUMPER — runs after every snap.

        Stage 1 — Origin protection: prevents workers landing on [0,0,0,0]
        where phase is undefined.

        Stage 2 — Worker-worker separation: any pair closer than
        bumper_radius * 2 gets pushed apart 50/50 along their delta vector.
        Uses self.bumper_radius (L2 min) and self.bumper_dim_min (per-dim min).
        Both tunable without touching saved homes. 21 pairs max — essentially free.
        """
        # ── Stage 1: Origin protection ───────────────────────────────────────
        for idx, data in self.homes.items():
            if idx in (1565, 1567):   # v135: bridge destinations — teleport-pinned, skip
                continue
            home = data['home']
            dist_to_origin = torch.norm(home[:4]).item()  # Check lens dims only
            if dist_to_origin < 0.05:
                if dist_to_origin > 1e-6:
                    direction = home[:4] / dist_to_origin
                else:
                    direction = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.dev)
                new_home = home.clone()
                new_home[:4] = direction * 0.3
                data['home'] = new_home
                magi_hive.pos_6d[idx] = new_home
                print(f"🌀 ORIGIN BUMPER: {data['name']} cleared from singularity")

        # ── Stage 2: Worker-worker separation — L2 + per-dim ────────────────
        # Two passes:
        #   L2 pass   — catches workers too close in total distance
        #   Per-dim   — catches dims that collapsed to near-zero independently
        #               (L2 can be satisfied while individual dims are ~0)
        min_sep     = self.bumper_radius * 2.0
        dim_min     = self.bumper_dim_min
        worker_indices = list(self.homes.keys())
        for i, idx_a in enumerate(worker_indices):
            if idx_a in (1565, 1567):   # v135: bridge destinations — skip as outer
                continue
            data_a = self.homes[idx_a]
            home_a = data_a['home']
            for idx_b in worker_indices[i + 1:]:
                if idx_b in (1565, 1567):   # v135: bridge destinations — skip as inner
                    continue
                data_b = self.homes[idx_b]
                home_b = data_b['home']
                delta    = home_b - home_a
                distance = torch.norm(delta).item()

                # L2 pass
                if distance < min_sep:
                    if distance > 1e-6:
                        push_dir = delta / (torch.norm(delta) + 1e-12)
                    else:
                        push_dir = torch.zeros(home_a.shape[0], device=home_a.device)
                        push_dir[idx_b % 4] = 1.0
                    push_amount = (min_sep - distance) * 0.5 + 0.01
                    home_b = home_b + push_dir * push_amount
                    home_a = home_a - push_dir * push_amount
                    print(f"⚡ L2 BUMPER: {data_a['name']} ↔ {data_b['name']}  "
                          f"dist={distance:.3f} → {min_sep:.3f}")

                # Per-dim pass — fix any dim still collapsed after L2
                dim_delta   = home_b - home_a
                collapsed   = []
                for d in range(min(4, home_a.shape[0])):  # Check lens dims only for per-dim separation
                    sep = abs(dim_delta[d].item())
                    if sep < dim_min:
                        push = (dim_min - sep) * 0.5 + 0.01
                        sign = 1.0 if dim_delta[d].item() >= 0 else -1.0
                        home_b[d] += sign * push
                        home_a[d] -= sign * push
                        collapsed.append(f"dim{d}={sep:.3f}")
                # if collapsed:
                #     print(f"⚡ DIM BUMPER: {data_a['name']} ↔ {data_b['name']}  "
                #           f"collapsed={collapsed}")

                data_b['home'] = home_b
                data_a['home'] = home_a
                magi_hive.pos_6d[idx_b] = home_b
                magi_hive.pos_6d[idx_a] = home_a

    def _prune_and_accumulate(self, idx, current_time):
        """
        v134: Time-prune pressure_history, then compute accumulated pressure.
        Returns (accumulated_pressure_tensor, magnitude_float).
        Entries older than window_duration are discarded — this is the
        actual time-window the UPE was always supposed to enforce.
        """
        cutoff = current_time - self.window_duration
        dq = self.pressure_history[idx]
        # Pop stale entries from the left (oldest first)
        while dq and dq[0][0] < cutoff:
            dq.popleft()
        if not dq:
            return torch.zeros(6, device=self.dev), 0.0
        accumulated = torch.zeros(6, device=self.dev)
        for t, p_vec in dq:
            age = current_time - t
            decay = math.exp(-age / self.pressure_decay_halflife)
            accumulated += p_vec * decay
        return accumulated, torch.norm(accumulated).item()

    def apply_black_hole_gravity(self, magi_hive):
        """
        6D Black Hole Gravity — detects proximity and applies directional pressure.
        - 6D DETECTION: full wrapped distance in toroidal space
        - 6D IMPULSE: buffered in _impulse_vel_6d (applied after velocity assembly)
        - 4D S-OSCILLATOR: buffered in _impulse_vel_s (future-location indicator)
        - SCALAR SPIKE: s_filtered += magnitude (immediate, survives low-pass)

        v134 FIX — three changes to prevent perpetual vibration:
          1. Time-prune deque via window_duration (entries expire, not just count-cap)
          2. last_pressure synced to real accumulated history (no asymmetric decay)
          3. Impulse + s-osc use accumulated trend direction, not instantaneous BH
        """
        current_time = time.time()
        
        bh_idx = self.bh_idx  # 1549
        bh_6d = magi_hive.pos_6d[bh_idx]  # v130: 6D position
        
        # Get cached scaler result
        bh_active = False
        bh_val = 0.0
        if hasattr(magi_hive, 'black_hole_last_result'):
            bh_result = magi_hive.black_hole_last_result
            bh_active = bh_result.get('is_active', False)
            bh_val = bh_result.get('output', 0.0)
        
        if not self.pressure_history:
            for idx in self.homes:
                self.pressure_history[idx] = deque(maxlen=3600)
                self.last_applied[idx] = current_time
                self.last_pressure[idx] = 0.0
        
        # v134: When BH is inactive, prune + sync last_pressure to real history
        # (old code did *= 0.98 which desynced from the undecayed deque)
        if not bh_active:
            for idx in self.pressure_history:
                _, mag = self._prune_and_accumulate(idx, current_time)
                self.last_pressure[idx] = mag
            return {}
        
        self.apply_singularity_bumper(magi_hive)
        tension_factor = bh_result.get('tension_factor', abs(bh_result.get('output', 0.0)) / 1500.0)

        # BH eps/gradient params — shared across all workers
        radius_scale = tension_factor
        current_eps_peak = magi_hive.black_hole_eps_floor + \
                          (magi_hive.black_hole_eps_max - magi_hive.black_hole_eps_floor) * radius_scale
        k = 1.0 + tension_factor * 4.0

        for idx, data in self.homes.items():
            # Skip all dream workers (1552–1557) — coupling drives them, not BH gravity
            if 1552 <= idx <= 1557:
                continue
            if idx in (1565, 1567):   # v135: bridge destinations — stateless, teleport-pinned
                continue
            home_pos = data['home']  # Now 6D [X,Y,Z,W,freq_p,delay_p]
            if home_pos.shape[0] == 4:  # Legacy conversion safety
                fp = freq_to_phase(1.0)
                dp = delay_to_phase(5.0)
                home_pos = torch.cat([home_pos, torch.tensor([fp, dp], device=self.dev)])
                data['home'] = home_pos

            # Per-worker detection radius — each home has its own r_target,
            # scaled by BH tension so radius grows as BH becomes more active
            worker_r = data.get('r_target', magi_hive.black_hole_base_radius)
            effective_radius = worker_r * (1.0 + tension_factor)

            # ✅ v130: 6D detection using full wrapped distance
            delta_wrapped = self._wrapped_difference(home_pos, bh_6d)  # 6D wrapped diff
            distance_6d = torch.norm(delta_wrapped)

            if distance_6d < effective_radius:
                # ✅ 6D MOVEMENT VECTOR (full dimensional gravity)
                to_bh_6d = -delta_wrapped  # Full 6D direction
                to_bh_norm_6d = to_bh_6d / (torch.norm(to_bh_6d) + 1e-12)
                
                # Normalized distance for gradient
                d_norm = torch.clamp(distance_6d / (effective_radius + 1e-12), 0.0, 1.0)
                
                if bh_val > 0:
                    # VACUUM MODE: Stronger toward center
                    decay_gradient = magi_hive.black_hole_eps_floor + \
                                    (current_eps_peak - magi_hive.black_hole_eps_floor) * \
                                    torch.pow(1.0 - d_norm, k)
                    pressure_vector = to_bh_norm_6d * decay_gradient * self.home_drift_strength
                else:
                    # SHIELD MODE: Stronger toward edge
                    decay_gradient = magi_hive.black_hole_eps_floor + \
                                    (current_eps_peak - magi_hive.black_hole_eps_floor) * \
                                    torch.pow(d_norm, k)
                    pressure_vector = -to_bh_norm_6d * decay_gradient * self.home_drift_strength
                
                self.pressure_history[idx].append((current_time, pressure_vector))
                
                # v134: Time-prune + accumulate (replaces raw deque scan)
                accumulated_pressure, pressure_magnitude_total = \
                    self._prune_and_accumulate(idx, current_time)

                if pressure_magnitude_total > 0:
                    # v134: Trend direction — the net of all historical pushes.
                    # This IS the direction the home will snap to (drift_vector below).
                    # Impulses vibrate the worker along this same axis so the
                    # vibration previews the eventual move instead of chasing each
                    # step's instantaneous BH geometry which can wander in 6D.
                    trend_dir = accumulated_pressure / (pressure_magnitude_total + 1e-12)

                    # Velocity impulse feedback
                    pressure_delta = max(0.0, pressure_magnitude_total - self.last_pressure[idx])
                    
                    if pressure_delta > self.min_impulse:
                        impulse_magnitude = min(pressure_delta * self.velocity_impulse_gain, 
                                               self.max_impulse)
                        # v134: Direction is the accumulated trend, not instantaneous BH.
                        # Sign is already baked in: vacuum vectors point toward BH,
                        # shield vectors point away — accumulated_pressure carries that.
                        impulse_vector = trend_dir * impulse_magnitude
                        magi_hive._impulse_vel_6d[idx] += impulse_vector
                        magi_hive._impulse_vel_s[idx] += impulse_vector[:4] * 0.5
                        magi_hive.s_filtered[idx] += impulse_magnitude * self.sensory_feedback_gain
                    
                    self.last_pressure[idx] = pressure_magnitude_total
                    
                    # Permanent home drift
                    if pressure_magnitude_total > self.accumulation_threshold:
                        drift_amount = min(self.accumulation_threshold / pressure_magnitude_total, 1.0)
                        drift_vector = accumulated_pressure * drift_amount
                        
                        old_home = home_pos.clone()
                        data['home'] = (data['home'] + drift_vector) % (2 * math.pi)
                        
                        # Home moved — old pressure data is from old position, discard it
                        self.pressure_history[idx] = deque(maxlen=3600)
                        self.last_applied[idx] = current_time
                        self.last_pressure[idx] = 0.0
                        
                        move_distance = torch.norm(data['home'] - old_home).item()
                        if move_distance > 0.001:
                            print(f"🏠 {data['name']}: MOVE {move_distance:.3f}")
            else:
                # v134: Worker outside radius — prune history + sync last_pressure
                # (old code only did last_pressure *= 0.98 which desynced from deque)
                _, mag = self._prune_and_accumulate(idx, current_time)
                self.last_pressure[idx] = mag
        
        return {}

    def get_home_position(self, worker_idx):
        """Get the current home position for a worker (used by MaGi init)"""
        if worker_idx in self.homes:
            return self.homes[worker_idx]['home'].clone()
        return None

    def get_home_stats(self):
        """
        Calculate statistics for all workers.
        Uses 4D vector magnitudes for all pressure calculations.
        v134: Prunes deque before scanning (matches gravity path).
        """
        stats = {}
        current_time = time.time()
        
        # Momentum memory (short-term pulse)
        if not hasattr(self, '_momentum_memory'):
            self._momentum_memory = {}
        
        for idx, data in self.homes.items():
            # v134: Prune first so stats match what gravity actually sees
            if idx in self.pressure_history:
                cutoff = current_time - self.window_duration
                dq = self.pressure_history[idx]
                while dq and dq[0][0] < cutoff:
                    dq.popleft()

            # Dual accumulators in 6D - THE CORE PHYSICS
            snap_accumulator = torch.zeros(6, device=self.dev)   # All-time decayed pressure (THE TRUTH)
            intent_accumulator = torch.zeros(6, device=self.dev) # window (TREND)
            
            # Physics constants
            half_life = self.pressure_decay_halflife
            threshold = self.accumulation_threshold
            window_seconds = self.window_duration
            
            # === SINGLE PASS: Calculate both bars ===
            if idx in self.pressure_history:
                for t, p_vec in self.pressure_history[idx]:
                    age = current_time - t
                    decay = math.exp(-age / half_life)  # MUST match apply_black_hole_gravity
                    
                    # Snap (Truth): All-time decayed pressure
                    snap_accumulator += p_vec * decay
                    
                    # Intent (Trend): Only within window
                    if age <= window_seconds:
                        intent_accumulator += p_vec * decay
            
            # === THE TRUTH: 4D vector magnitude ===
            snap_p = torch.norm(snap_accumulator).item()      # What triggers home snap
            intent_p = torch.norm(intent_accumulator).item()  # Recent trend
            
            snap_pct = (snap_p / threshold) * 100.0
            intent_pct = (intent_p / threshold) * 100.0
            
            # === MOMENTUM: Short-term pulse (last 3 readings) ===
            if idx not in self._momentum_memory:
                self._momentum_memory[idx] = []
            
            # Store current intent pressure for momentum
            self._momentum_memory[idx].append(intent_p)
            if len(self._momentum_memory[idx]) > 3:
                self._momentum_memory[idx].pop(0)
            
            # Calculate momentum (rate of change per measurement)
            momentum = 0.0
            momentum_trend = "→"  # Default: stable
            
            if len(self._momentum_memory[idx]) >= 2:
                momentum = intent_p - self._momentum_memory[idx][-2]
                
                # Dynamic thresholds based on pressure magnitude
                dynamic_threshold = max(0.0001, intent_p * 0.1)  # 10% of current pressure
                
                if momentum > dynamic_threshold * 2:
                    momentum_trend = "↗↗"  # Strong build
                elif momentum > dynamic_threshold:
                    momentum_trend = "↗"   # Building
                elif momentum < -dynamic_threshold * 2:
                    momentum_trend = "↘↘"  # Strong decay
                elif momentum < -dynamic_threshold:
                    momentum_trend = "↘"   # Decaying
            
            # Sample statistics (for reliability)
            recent_samples = 0
            older_samples = 0
            if idx in self.pressure_history:
                for t, _ in self.pressure_history[idx]:
                    age = current_time - t
                    if age <= 600:  # Last 10 minutes
                        recent_samples += 1
                    elif 3000 <= age <= 3600:  # 50-60 minutes ago
                        older_samples += 1
            
            rate_reliable = (recent_samples >= 3 and older_samples >= 3)
            
            stats[idx] = {
                'name': data.get('name', f'Worker_{idx}'),
                
                # === DUAL BARS ===
                'snap_pressure': snap_p,      # The capacitor (all-time)
                'intent_pressure': intent_p,  # The pump (recent)
                'snap_pct': snap_pct,         # Truth percentage
                'intent_pct': intent_pct,     # Trend percentage
                'snap_ready': snap_p >= threshold,  # THE TRIGGER
                
                # === MOMENTUM ===
                'momentum': momentum,
                'momentum_trend': momentum_trend,
                
                # === PHYSICS CONSTANTS ===
                'threshold': threshold,
                'window_hours': window_seconds / 3600,
                
                # === RELIABILITY ===
                'rate_reliable': rate_reliable,
                'recent_samples': recent_samples,
                'older_samples': older_samples,
                'total_samples': len(self.pressure_history.get(idx, [])),
                
                # === TIME CONTEXT ===
                'time_since_drift': current_time - self.last_applied.get(idx, current_time),
            }
        
        return stats

    def maybe_save(self, pos_6d, freq_wrap=None, delay_wrap=None):
        """
        Smart save logic: only save if homes have moved significantly.
        v131: Also persists freq/delay wrap counters for absolute Hz recovery.
        """
        self.steps_since_save += 1
        
        # Initialize last saved positions if first time
        if self.last_saved_homes is None:
            self.last_saved_homes = {}
            for idx, data in self.homes.items():
                self.last_saved_homes[idx] = data['home'].clone()
        
        # Calculate max drift from last saved positions (6D distance!)
        max_drift = 0.0
        for idx, data in self.homes.items():
            if 1552 <= idx <= 1557:   # Dream workers — not UPE-managed
                continue
            if idx in (1565, 1567):   # v135: bridge destinations — stateless
                continue
            if idx in self.last_saved_homes:
                drift = torch.norm(data['home'] - self.last_saved_homes[idx]).item()
                max_drift = max(max_drift, drift)
        
        # Save only if significant drift OR periodic backup
        if (self.steps_since_save >= 1000 and max_drift > 0.1) or (self.steps_since_save >= 10000):
            save_data = {}
            for idx, data in self.homes.items():
                save_data[idx] = {
                    'home': data['home'].detach().clone().cpu(),
                    'type': data['type'],
                    'r_target': data['r_target'],
                    'name': data.get('name', f'Worker_{idx}')
                }
            # v131: Persist wrap counters
            if freq_wrap is not None:
                save_data['_freq_wrap'] = freq_wrap.cpu()
            if delay_wrap is not None:
                save_data['_delay_wrap'] = delay_wrap.cpu()
            save_data['_mapping_version'] = 1
            torch.save(save_data, self.file_path)
            
            # CRITICAL: Update pos_6d so workers use new home positions
            for idx, data in self.homes.items():
                pos_6d[idx] = data['home'].clone()
                self.last_saved_homes[idx] = data['home'].clone()
            
            self.steps_since_save = 0
            if max_drift > 0.1:
                print(f"💾 UPE: Saved home positions (max 6D drift: {max_drift:.3f})")
                print(f"    → Workers reset to new home positions")

# ==========================================
# 🎯 KINETIC MANIFOLD — v126
# ==========================================

# Fixed direction signatures per ALE worker.
# Each vector defines the 6D phase-space direction that worker broadcasts.
# Dims 0-3: lens direction (child, youth, adult, elder)
# Dims 4-5: freq/delay phase direction (spectral color of action)
# Normalised inside AleVibrationBeacon.__init__.
ALE_VIB_SIGNATURES = {
    1542: torch.tensor([ 1.0,  0.0,  1.0,  0.0, -0.5,  0.0], dtype=torch.float32),   # LEFT:  +child,+adult, freq DOWN
    1543: torch.tensor([-1.0,  0.0,  1.0,  0.0,  0.5,  0.0], dtype=torch.float32),   # RIGHT: -child,+adult, freq UP
    1544: torch.tensor([ 0.0,  1.0,  0.0,  1.0,  0.5,  0.5], dtype=torch.float32),   # FIRE:  +youth,+elder, freq+delay UP
    1545: torch.tensor([ 0.0, -1.0,  0.0,  1.0,  0.0, -0.5], dtype=torch.float32),   # UP:    -youth,+elder, delay DOWN (faster)
    1546: torch.tensor([ 0.0,  1.0,  0.0, -1.0,  0.0,  0.5], dtype=torch.float32),   # DOWN:  +youth,-elder, delay UP (slower)
    1547: torch.tensor([ 0.0,  0.0,  1.0,  1.0,  0.0,  0.0], dtype=torch.float32),   # NOOP:  +adult,+elder, neutral spectral
}

# v133: Robot arm vibration signatures — semantically meaningful 6D directions
# Cartesian workers: spatial in lens dims, spectral spread in freq/delay
# Wrist workers: rotational feel in lens, distinct spectral color
# Gripper: balanced compression/expansion signature
# Normalised inside AleVibrationBeacon.__init__ (same class, separate dict).
ROBOT_VIB_SIGNATURES = {
    1558: torch.tensor([ 1.0, -1.0,  0.0,  0.0,  0.3,  0.0], dtype=torch.float32),  # ARM_X:    child↑ youth↓ (lateral)
    1559: torch.tensor([ 0.0,  0.0,  1.0, -1.0,  0.0,  0.3], dtype=torch.float32),  # ARM_Y:    adult↑ elder↓ (depth)
    1560: torch.tensor([ 1.0,  1.0,  0.0,  0.0, -0.3,  0.3], dtype=torch.float32),  # ARM_Z:    child↑ youth↑ (vertical)
    1561: torch.tensor([ 0.0,  1.0,  0.0,  1.0,  0.5, -0.5], dtype=torch.float32),  # ARM_ROT:  youth↑ elder↑ (rotation)
    1562: torch.tensor([ 1.0,  0.0,  1.0,  0.0, -0.5,  0.5], dtype=torch.float32),  # ARM_TILT: child↑ adult↑ (pitch)
    1563: torch.tensor([ 0.5, -0.5,  0.5, -0.5,  1.0,  1.0], dtype=torch.float32),  # GRIPPER:  balanced lens + spectral
}

# v133: Progressive robot modes — which workers are active at each complexity level
ROBOT_MODE_WORKERS = {
    0: set(),                                       # disabled — all 6 frozen
    1: {1558, 1559, 1563},                          # X, Y, Gripper
    2: {1558, 1559, 1560, 1563},                    # + Z
    3: {1558, 1559, 1560, 1561, 1563},              # + Wrist Rot
    4: {1558, 1559, 1560, 1561, 1562, 1563},        # + Wrist Tilt (all 6)
}


class AleVibrationBeacon:
    """
    Part 1 — ALE Vibration Beacons.

    Each ALE worker (1542-1547) continuously broadcasts a deterministic
    phase-locked vibration in its 6D semantic direction. The per-worker
    frequency offset keeps the six beacons from synchronising, giving
    each a distinguishable signature in the hive field.

    v131: 6D signatures — lens direction (dims 0-3) + freq/delay color (dims 4-5).
    Drives both S-oscillator (coherence-visible) and toroidal state (position drift).
    Called from process_step AFTER velocity assembly, BEFORE integration.

    No N-bank query — fires every step unconditionally.
    """
    def __init__(self, ale_idx, signature_vec, upe, device='cuda'):
        self.idx       = ale_idx
        self.signature = (signature_vec / torch.norm(signature_vec)).to(device)  # 6D normalized
        self.upe       = upe
        self.device    = device
        self.phase_accum = 0.0   # deterministic, starts at 0

    def vibrate(self, magi_hive):
        """Broadcast 6D phase-locked vibration. Must be called AFTER velocity assembly."""
        km    = self.upe.km_config
        omega = km['ale_vib_omega_base'] + km['ale_omega_offsets'][self.idx]
        amp   = km['ale_vib_strength']

        delta = amp * math.sin(2.0 * math.pi * omega * self.phase_accum)
        delta = max(-km['ale_vib_clamp'], min(km['ale_vib_clamp'], delta))

        # v131: Dual broadcast — 4D lens via S-oscillator + 6D full toroidal state
        magi_hive.vel_s[self.idx]  += self.signature[:4] * delta   # S-oscillator (coherence visible)
        magi_hive.vel_6d[self.idx] += self.signature * delta       # Toroidal integration (position drift)
        self.phase_accum += 1.0


class RobotVibrationBeacon(AleVibrationBeacon):
    """
    v133: Robot arm vibration beacon — same physics as ALE beacons but reads
    from robot_vib_* km_config keys instead of ale_vib_* keys.
    """
    def vibrate(self, magi_hive):
        """Broadcast 6D phase-locked vibration using robot-specific km_config."""
        km    = self.upe.km_config
        omega = km['robot_vib_omega_base'] + km['robot_omega_offsets'][self.idx]
        amp   = km['robot_vib_strength']

        delta = amp * math.sin(2.0 * math.pi * omega * self.phase_accum)
        delta = max(-km['robot_vib_clamp'], min(km['robot_vib_clamp'], delta))

        # Dual broadcast — 4D lens via S-oscillator + 6D full toroidal state
        magi_hive.vel_s[self.idx]  += self.signature[:4] * delta
        magi_hive.vel_6d[self.idx] += self.signature * delta
        self.phase_accum += 1.0


def _wrap_delta(delta):
    """Wrap phase delta tensor to [-π, π]."""
    return torch.remainder(delta + math.pi, 2 * math.pi) - math.pi


def _clamp_norm(v, max_norm):
    """Scale vector to max_norm if its L2 norm exceeds max_norm. Returns new tensor."""
    n = torch.norm(v)
    if n > max_norm:
        v = v * (max_norm / (n + 1e-8))
    return v


class DreamMirrorCoupling:
    """
    Part 2 — Dream Mirror Workers.

    N-Reader (1552): lazy-river drift through N-bank phase space.
      Attracts to dense N-memory clusters via density gradient.
      On contact: boosts N access counts, hands best target to Main-Anchor
      via kinetic sprint + cross-vibration of Main-Anchor vel_s.

    Main-Anchor (1553): lazy-river drift through main-bank phase space.
      Attracts to dense main-memory clusters.
      On contact: boosts main access counts, sends reciprocal kick to
      N-Reader (pushes it to explore new narrative regions), cross-vibrates
      N-Reader vel_s.

    Input energy (inputs_tensor norm) modulates drift and adds a deterministic
    nudge — the sensory world steers the dream.

    vel_6d is clamped each step; vel_s is zeroed by process_step (indices
    1552, 1553 hardcoded) so the CUDA/Python kernel never overwrites kicks.
    """
    def __init__(self, upe, device):
        self.upe    = upe
        self.device = device

        # Drift oscillators (deterministic, start at 0)
        self.n_drift_phase = 0.0
        self.m_drift_phase = 0.0

        # Cross-coupling oscillators
        self.n_to_m_phase = 0.0
        self.m_to_n_phase = 0.0

        # v130: persistent drift state (atomic replacement) - now 6D
        self.current_main_drift = torch.zeros(6, device=device)

        # Telemetry (per-step, reset in update)
        self.n_last_boosted    = []
        self.m_last_boosted    = []
        self.n_in_field        = 0
        self.m_in_field        = 0
        self.last_target_phase = None

    def _deterministic_kick(self, step, strength, dtype):
        """Produce a deterministic unit-vector kick from step counter (6D)."""
        vec  = torch.tensor([
            math.sin(step * 0.1),
            math.sin(step * 0.3),
            math.cos(step * 0.7),
            math.cos(step * 0.9),
            math.sin(step * 0.5),
            math.cos(step * 0.2),
        ], device=self.device, dtype=dtype)
        return vec / (torch.norm(vec) + 1e-8) * strength

    def update(self, magi_hive):
        km    = self.upe.km_config
        n_idx = km['dream_n_reader_idx']
        m_idx = km['dream_main_anchor_idx']
        step  = magi_hive.global_age
        dtype = magi_hive.vel_6d.dtype

        # Reset per-step telemetry
        self.n_last_boosted = []
        self.m_last_boosted = []

        # Input energy — common sensory driver
        if hasattr(magi_hive, '_inputs_buffer'):
            input_energy = min(
                torch.norm(magi_hive._inputs_buffer).item() / 500.0, 1.0
            )
        else:
            input_energy = 0.0

        # ══════════════════ N-READER ══════════════════════════════════════
        if magi_hive.n_bank.size > 0:
            # Lazy river drift
            self.n_drift_phase = (
                self.n_drift_phase + km['dream_n_drift_omega']
            ) % (2 * math.pi)
            drift_vec = torch.tensor([
                math.sin(self.n_drift_phase),
                math.cos(self.n_drift_phase * 0.7),
                math.sin(self.n_drift_phase * 0.3) * 0.5,
                0.0,
                math.sin(self.n_drift_phase * 0.5) * 0.3,
                math.cos(self.n_drift_phase * 0.2) * 0.3,
            ], device=self.device, dtype=dtype) * km['dream_drift_speed']
            magi_hive.vel_6d[n_idx] += drift_vec

            # Proximity to N memories (6D distance)
            # v131: N bank coords[5:7] are unwrapped log — build comparable query
            reader_6d = torch.cat([
                magi_hive.pos_6d[n_idx, :4],  # lens phases (wrapped)
                torch.tensor([
                    freq_to_log_coord(magi_hive.freq[n_idx].item()),
                    delay_to_log_coord(magi_hive.delay[n_idx].item())
                ], device=self.device, dtype=dtype)
            ])
            # Extract 6D coords from N-bank (skip log_time at index 4)
            n_coords_6d = torch.cat([
                magi_hive.n_bank.coords[:magi_hive.n_bank.size, :4],
                magi_hive.n_bank.coords[:magi_hive.n_bank.size, 5:7]
            ], dim=1).to(dtype)
            delta_n    = torch.abs(n_coords_6d - reader_6d.unsqueeze(0))
            # Wrap lens dims (0-3) only — freq/delay dims (4-5) are unwrapped log
            delta_n[:, :4] = torch.min(delta_n[:, :4], 2 * math.pi - delta_n[:, :4])
            n_dists    = torch.norm(delta_n, dim=1)

            touch_mask      = n_dists < km['dream_n_proximity_radius']
            self.n_in_field = touch_mask.sum().item()

            if self.n_in_field > 0:
                # Boost all touched N memories (vectorised)
                magi_hive.n_bank.access_counts[:magi_hive.n_bank.size][touch_mask] += km['dream_memory_boost']
                self.n_last_boosted = torch.where(touch_mask)[0].tolist()

                # Density gradient: attract toward high-access clusters
                weights     = (
                    magi_hive.n_bank.access_counts[:magi_hive.n_bank.size][touch_mask] + 1.0
                )
                weights     = weights / (n_dists[touch_mask] + 0.01)
                dir_to_mem  = -delta_n[touch_mask]
                weighted_dir = (dir_to_mem * weights.unsqueeze(1)).sum(dim=0)
                wd_norm = torch.norm(weighted_dir)
                if wd_norm > 0:
                    weighted_dir = weighted_dir / wd_norm

                tension = input_energy * min(self.n_in_field / 10.0, 1.0)
                eps     = km['dream_eps_floor'] + (
                    km['dream_eps_max'] - km['dream_eps_floor']
                ) * tension
                magi_hive.vel_6d[n_idx] += weighted_dir * eps * 0.05

                # Kinetic sprint: handoff best N memory target to Main-Anchor
                best_local    = torch.argmax(
                    magi_hive.n_bank.access_counts[touch_mask]
                )
                best_n_idx    = torch.where(touch_mask)[0][best_local].item()   # v128: .item()
                target_phase  = magi_hive.memory_to_main_target(best_n_idx)  # Returns 6D
                self.last_target_phase = target_phase.clone()

                delta_main    = target_phase - magi_hive.pos_6d[m_idx]
                delta_main    = _wrap_delta(delta_main)
                dir_to_target = delta_main / (torch.norm(delta_main) + 1e-8)

                access_weight = min(
                    magi_hive.n_bank.access_counts[best_n_idx].item() / 100.0, 1.0
                )
                sprint_power  = (
                    km['dream_drift_speed']
                    * km['dream_sprint_multiplier']
                    * access_weight
                )
                magi_hive.vel_6d[m_idx] += dir_to_target * sprint_power

                # v128: Memory-directed impulse toward main-bank target coordinates
                target_main = magi_hive.memory_to_main_target(best_n_idx)
                drift_vec   = magi_hive.memory_to_drift_vector(best_n_idx)

                delta_imp = _wrap_delta(target_main - magi_hive.pos_6d[m_idx])
                if torch.norm(delta_imp) > 0:
                    imp_dir = delta_imp / torch.norm(delta_imp)
                    access  = magi_hive.n_bank.access_counts[best_n_idx].item()
                    tension_val = magi_hive.n_bank.metadata_tension_ce[best_n_idx].item()
                    scale = (1.0
                             + access * km.get('drift_impulse_access_scale', 0.01)
                             + tension_val * km.get('drift_impulse_tension_scale', 0.5))
                    impulse = imp_dir * km.get('drift_impulse_gain', 0.02) * scale
                    magi_hive.vel_6d[m_idx] += impulse

                # v128: Replace persistent drift (old removed, new added atomically)
                magi_hive.vel_6d[m_idx] += drift_vec - self.current_main_drift
                self.current_main_drift = drift_vec.clone()

                # Cross-vibration: N-Reader vibrates Main-Anchor vel_s
                vib_amp   = km['dream_vib_strength'] * min(self.n_in_field / 5.0, 1.0)
                delta_vib = vib_amp * math.sin(
                    2.0 * math.pi * 0.5 * self.n_to_m_phase
                )
                magi_hive.vel_s[m_idx, 2] += delta_vib
                magi_hive.vel_s[m_idx, 3] += delta_vib * 0.5
                self.n_to_m_phase += 1.0

        # ══════════════════ MAIN-ANCHOR ═══════════════════════════════════
        if magi_hive.memory_bank.size > 0:
            # Lazy river drift
            self.m_drift_phase = (
                self.m_drift_phase + km['dream_m_drift_omega']
            ) % (2 * math.pi)
            drift_vec_m = torch.tensor([
                math.cos(self.m_drift_phase * 0.6),
                math.sin(self.m_drift_phase),
                math.cos(self.m_drift_phase * 0.4) * 0.5,
                0.0,
                math.sin(self.m_drift_phase * 0.3) * 0.3,
                math.cos(self.m_drift_phase * 0.8) * 0.3,
            ], device=self.device, dtype=dtype) * km['dream_drift_speed'] * 0.8
            magi_hive.vel_6d[m_idx] += drift_vec_m

            # Proximity to main memories (6D distance via mem_coords_6d)
            # v131: mem_coords_6d stores unwrapped log coords in dims 4-5
            # Build a comparable 6D query from worker's current state
            anchor_pos_6d = torch.cat([
                magi_hive.pos_6d[m_idx, :4],  # lens phases (wrapped, same in both)
                torch.tensor([
                    freq_to_log_coord(magi_hive.freq[m_idx].item()),
                    delay_to_log_coord(magi_hive.delay[m_idx].item())
                ], device=self.device, dtype=dtype)
            ])
            mem_size = magi_hive.memory_bank.size
            if mem_size > 0:
                main_coords_6d = magi_hive.memory_bank.mem_coords_6d[:mem_size].to(dtype)
                delta_m = torch.abs(main_coords_6d - anchor_pos_6d.unsqueeze(0))
                # Wrap lens dims (0-3) only — freq/delay dims (4-5) are unwrapped
                delta_m[:, :4] = torch.min(delta_m[:, :4], 2 * math.pi - delta_m[:, :4])
                m_dists = torch.norm(delta_m, dim=1)

                touch_mask_m  = m_dists < km['dream_main_proximity_radius']
                self.m_in_field = touch_mask_m.sum().item()

            if self.m_in_field > 0:
                # Boost touched main memories (vectorised)
                magi_hive.memory_bank.access_counts[:magi_hive.memory_bank.size][touch_mask_m] += km['dream_memory_boost']
                self.m_last_boosted = torch.where(touch_mask_m)[0].tolist()

                # Density gradient: attract toward dense main clusters
                weights_m     = (
                    magi_hive.memory_bank.access_counts[:magi_hive.memory_bank.size][touch_mask_m] + 1.0
                )
                weights_m     = weights_m / (m_dists[touch_mask_m] + 0.01)
                dir_to_mem_m  = -delta_m[touch_mask_m]
                weighted_dir_m = (dir_to_mem_m * weights_m.unsqueeze(1)).sum(dim=0)
                wdm_norm = torch.norm(weighted_dir_m)
                if wdm_norm > 0:
                    weighted_dir_m = weighted_dir_m / wdm_norm

                tension_m = input_energy * min(self.m_in_field / 10.0, 1.0)
                eps_m     = km['dream_eps_floor'] + (
                    km['dream_eps_max'] - km['dream_eps_floor']
                ) * tension_m
                magi_hive.vel_6d[m_idx] += weighted_dir_m * eps_m * 0.05

                # Cross-vibration: Main-Anchor vibrates N-Reader vel_s
                vib_amp_m   = km['dream_vib_strength'] * min(self.m_in_field / 5.0, 1.0)
                delta_vib_m = vib_amp_m * math.sin(
                    2.0 * math.pi * 0.5 * self.m_to_n_phase
                )
                magi_hive.vel_s[n_idx, 2] += delta_vib_m * 0.5
                self.m_to_n_phase += 1.0

                # Reciprocal kick: deterministic push on N-Reader
                kick_strength = km['dream_reciprocal_kick'] * min(
                    self.m_in_field / 10.0, 1.0
                )
                kick_vec = self._deterministic_kick(step, kick_strength, dtype)
                magi_hive.vel_6d[n_idx] += kick_vec

                # Weak reciprocal pull (always present when in field)
                delta_pos_m = anchor_pos_6d - magi_hive.pos_6d[n_idx]
                delta_pos_m = _wrap_delta(delta_pos_m)
                magi_hive.vel_6d[n_idx] += (
                    delta_pos_m * km['dream_coupling_strength'] * 0.5 * tension_m
                )

        # ══════════════════ INPUT-DRIVEN NUDGE ════════════════════════════
        if input_energy > 0.05:
            thrust = input_energy * 0.002
            dir_n  = torch.tensor([
                math.sin(input_energy * 10),
                math.cos(input_energy * 7),
                math.sin(input_energy * 13),
                math.cos(input_energy * 11),
                math.sin(input_energy * 6),
                math.cos(input_energy * 9),
            ], device=self.device, dtype=dtype)
            dir_n = dir_n / (torch.norm(dir_n) + 1e-8)
            magi_hive.vel_6d[n_idx] += dir_n * thrust

            dir_m = torch.tensor([
                math.cos(input_energy * 8),
                math.sin(input_energy * 12),
                math.cos(input_energy * 9),
                math.sin(input_energy * 14),
                math.cos(input_energy * 5),
                math.sin(input_energy * 11),
            ], device=self.device, dtype=dtype)
            dir_m = dir_m / (torch.norm(dir_m) + 1e-8)
            magi_hive.vel_6d[m_idx] += dir_m * thrust * 0.8

        # ══════════════════ VELOCITY NORM CLAMP ═══════════════════════════
        magi_hive.vel_6d[n_idx] = _clamp_norm(
            magi_hive.vel_6d[n_idx], km['dream_clamp']
        )
        magi_hive.vel_6d[m_idx] = _clamp_norm(
            magi_hive.vel_6d[m_idx], km['dream_clamp']
        )


class ChordTeleportCoupling:
    """
    Dream Mirror — Chord (Teleport) Pair: workers 1554 & 1555.
    On each update, checks n_gravity_audio/video for a high-access N memory
    with a stored phase coordinate.  When a new best memory is found (and
    the cooldown has elapsed), both workers are instantly teleported to that
    memory's phase, their integrators zeroed, and the memory's access count
    boosted.  This implements episodic recall: the pair jumps to wherever
    the most-recently recognised N memory lives in phase space.
    """
    def __init__(self, upe, device):
        self.upe = upe
        self.device = device
        self.last_teleport_step = 0
        self.last_memory_idx = -1
        self.teleport_count = 0
        # v130: persistent drift state — 6D
        self.current_main_drift = torch.zeros(6, device=device)

    def update(self, magi_hive):
        km    = self.upe.km_config
        n_idx = km['chord_n_reader_idx']    # 1554
        m_idx = km['chord_main_anchor_idx'] # 1555
        step  = magi_hive.global_age

        if step - self.last_teleport_step < km['chord_teleport_cooldown']:
            return

        # Pick the chord dict with the highest top_access that also has mem_phase
        best_chord  = None
        best_access = -1
        for chord_dict in (magi_hive.n_gravity_audio, magi_hive.n_gravity_video):
            if chord_dict.get('chord_size', 0) > 0 and 'mem_phase' in chord_dict:
                if chord_dict['top_access'] > best_access:
                    best_access = chord_dict['top_access']
                    best_chord  = chord_dict

        if best_chord is None:
            return

        mem_idx = best_chord['mem_idx']
        if mem_idx == self.last_memory_idx:
            return

        target_phase = best_chord['mem_phase'].clone() % (2 * math.pi)

        # Teleport both workers to the memory's phase (6D)
        magi_hive.pos_6d[n_idx] = target_phase
        magi_hive.pos_6d[m_idx] = target_phase
        magi_hive.phases_s[n_idx] = target_phase[:4]  # S-osc stays 4D
        magi_hive.phases_s[m_idx] = target_phase[:4]

        # Zero velocities and integrators
        for idx in (n_idx, m_idx):
            magi_hive.vel_6d[idx].zero_()
            magi_hive.vel_s[idx].zero_()
            magi_hive.s_filtered[idx] = 0.0
            magi_hive.s_last[idx]     = 0.0
            magi_hive.s_integral[idx] = 0.0

        # Boost the recalled memory
        magi_hive.n_bank.access_counts[mem_idx] += km['chord_boost_amount']

        # v128: Set persistent drift from the memory that caused the teleport
        drift_vec = magi_hive.memory_to_drift_vector(mem_idx)
        magi_hive.vel_6d[m_idx] += drift_vec - self.current_main_drift
        self.current_main_drift = drift_vec.clone()

        self.last_teleport_step = step
        self.last_memory_idx    = mem_idx
        self.teleport_count    += 1


class PhysicsCoupling:
    """
    Dream Mirror — Physics (Lens-Driven) Pair: workers 1556 & 1557.
    N-Reader moves toward dense N memories. When a new high‑weight memory
    is detected, the Main‑Anchor teleports to its corresponding main‑bank
    coordinates and inherits a decaying drift vector derived from the
    memory's metadata.  The drift fades over time, and a weak lazy‑river
    keeps the Main‑Anchor moving when no memory is active.
    """
    def __init__(self, upe, device):
        self.upe = upe
        self.device = device

        # Persistent drift state for Main‑Anchor (atomic replacement) — v130: 6D
        self.current_main_drift = torch.zeros(6, device=device)
        self.applied_drift = torch.zeros(6, device=device)   # what is currently added
        self.drift_decay = 1.0

        # Teleport jitter prevention
        self.last_mem_idx = -1

        # Lazy‑river oscillator
        self.m_drift_phase = 0.0

        # Reusable lazy‑river tensor — v130: 6D
        self.lazy_vec_template = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)

    def update(self, magi_hive):
        km = self.upe.km_config
        n_idx = km['physics_n_reader_idx']      # 1556
        m_idx = km['physics_main_anchor_idx']   # 1557
        dtype = magi_hive.vel_6d.dtype

        if magi_hive.n_bank.size == 0:
            return

        # ======================================================
        # N-Reader: force toward high‑access N memories
        # ======================================================
        # v131: Build 6D query with unwrapped log for freq/delay
        reader_6d = torch.cat([
            magi_hive.pos_6d[n_idx, :4],
            torch.tensor([
                freq_to_log_coord(magi_hive.freq[n_idx].item()),
                delay_to_log_coord(magi_hive.delay[n_idx].item())
            ], device=self.device)
        ])

        sample_size = min(1000, magi_hive.n_bank.size)
        stride = max(1, magi_hive.n_bank.size // sample_size)
        sample_idx = torch.arange(0, magi_hive.n_bank.size, stride,
                                  device=self.device)[:sample_size]

        # v131: Extract 6D coords from N-bank (skip log_time at index 4)
        n_coords = torch.cat([
            magi_hive.n_bank.coords[sample_idx, :4],
            magi_hive.n_bank.coords[sample_idx, 5:7]
        ], dim=1)
        n_access = magi_hive.n_bank.access_counts[sample_idx]

        delta_raw = n_coords - reader_6d.unsqueeze(0)
        # Wrap lens dims (0-3) only — freq/delay dims (4-5) are unwrapped
        delta_raw[:, :4] = torch.remainder(delta_raw[:, :4] + math.pi, 2 * math.pi) - math.pi
        dists = torch.norm(delta_raw, dim=1)

        near_mask = dists < km['physics_proximity_radius']
        detection_happened = False

        if near_mask.any():
            weights = n_access[near_mask] / (dists[near_mask].pow(2) + 0.001)
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum
                direction = (delta_raw[near_mask] * weights.unsqueeze(1)).sum(dim=0)
                lens_pressure = magi_hive.lens_weights[n_idx, 2]   # adult lens
                force_magnitude = km['physics_force_gain'] * (1.0 + lens_pressure)
                forces = direction * force_magnitude

                magi_hive.vel_6d[n_idx] += forces * km['physics_spring_k']
                magi_hive.vel_6d[n_idx] *= km['physics_damping']

                vel_norm = torch.norm(magi_hive.vel_6d[n_idx])
                if vel_norm > km['physics_clamp']:
                    magi_hive.vel_6d[n_idx] *= km['physics_clamp'] / vel_norm

                # ---- Memory detection for Main‑Anchor ----
                max_weight_idx = torch.argmax(weights)
                best_mem_idx = sample_idx[max_weight_idx].item()

                # Only teleport when the best memory changes
                if best_mem_idx != self.last_mem_idx:
                    self.last_mem_idx = best_mem_idx
                    detection_happened = True

                    # Target in main‑bank coordinates
                    target_main = magi_hive.memory_to_main_target(best_mem_idx)

                    # Teleport Main‑Anchor
                    magi_hive.pos_6d[m_idx] = target_main
                    magi_hive.phases_s[m_idx] = target_main[:4]  # S-osc stays 4D

                    # Zero all motion state (clean slate)
                    magi_hive.vel_6d[m_idx].zero_()
                    magi_hive.vel_s[m_idx].zero_()
                    magi_hive.s_filtered[m_idx] = 0.0
                    magi_hive.s_last[m_idx] = 0.0
                    magi_hive.s_integral[m_idx] = 0.0

                    # New drift vector from the memory, full strength
                    self.current_main_drift = magi_hive.memory_to_drift_vector(best_mem_idx)
                    self.drift_decay = 1.0

                    # Reset the applied drift to avoid a correction spike
                    self.applied_drift.zero_()

        # ======================================================
        # Main‑Anchor: apply decaying drift (atomic replacement)
        # ======================================================
        # Decay the drift strength (after computing the current new_drift for this step)
        # This ensures the first frame after teleport uses full strength.
        new_drift = self.current_main_drift * self.drift_decay

        # Replace the drift contribution (avoid double‑integration)
        magi_hive.vel_6d[m_idx] += new_drift - self.applied_drift
        self.applied_drift = new_drift.clone()

        # Now decay for the next step
        self.drift_decay *= km.get('physics_drift_decay', 0.995)
        if self.drift_decay < 0.01:
            self.drift_decay = 0.0

        # ======================================================
        # Fallback lazy‑river when drift is nearly gone
        # ======================================================
        if self.drift_decay < 0.1 and not detection_happened:
            self.m_drift_phase = (self.m_drift_phase + km.get('dream_m_drift_omega', 0.07)) % (2 * math.pi)
            # Reuse a pre‑allocated tensor to avoid allocation every frame (micro‑opt)
            lazy_vec = self.lazy_vec_template
            lazy_vec[0] = math.cos(self.m_drift_phase * 0.6)
            lazy_vec[1] = math.sin(self.m_drift_phase)
            lazy_vec[2] = math.cos(self.m_drift_phase * 0.4) * 0.5
            lazy_vec[3] = 0.0
            lazy_vec = lazy_vec * km.get('physics_lazy_river_strength', 0.0005)
            magi_hive.vel_6d[m_idx] += lazy_vec

        # Final clamp (safety)
        vel_norm = torch.norm(magi_hive.vel_6d[m_idx])
        if vel_norm > km['physics_clamp']:
            magi_hive.vel_6d[m_idx] *= km['physics_clamp'] / vel_norm


class CoherentWorkerTracker:
    """
    Tracks the most coherent worker in each video scale PER FRAME.

    KEY CONCEPTS:
    - Single Source of Truth: Used by both input enhancement AND HUD
    - Frame-Local State: No temporal smoothing or persistence
    - Execution Contract: update() → HUD rendering must happen same frame
    - Visual-Actuation Alignment: What you see = what influences behavior
    - On-Demand Color: Extracts YUV color ONLY for coherent sectors (efficient)

    ARCHITECTURE:
    - 4 scales (0-3) with Fibonacci grids (5×3, 8×5, 13×8, 21×13)
    - Each scale: finds worker with highest global_coh
    - That worker gets YUV U-channel color (0-500 range)
    - HUD shows same worker with attention square
    """
    def __init__(self):
        # ==========================================
        # SCALE DEFINITIONS (IMMUTABLE)
        # ==========================================
        # (start_idx, end_idx, grid_h, grid_w, color_bgr, display_name)
        self.scales = [
            (516,  531,  3,  5, (0, 255, 255),   "scale_0"),  # 5×3  (15 workers)
            (531,  571,  5,  8, (0, 165, 255),   "scale_1"),  # 8×5  (40 workers)
            (571,  675,  8, 13, (0, 0, 255),     "scale_2"),  # 13×8 (104 workers)
            (675,  948, 13, 21, (255, 0, 255),   "scale_3")   # 21×13 (273 workers)
        ]
        
        # ==========================================
        # FRAME-LOCAL STATE (RESETS EACH FRAME)
        # ==========================================
        self._current_workers = [None] * 4      # Coherent worker data per scale
        self._coherence_values = [0.0] * 4      # Actual coherence [0-1]
        self._u_values = [250.0] * 4            # YUV U-channel values (0-500 range)
        self._frame_id = 0                      # Debug: track frame consistency
        self._is_valid = False                  # True only if updated this frame
        
        # ==========================================
        # DIAGNOSTICS (PERSISTENT ACROSS FRAMES)
        # ==========================================
        self._fallback_counts = [0, 0, 0, 0]    # Times each scale used fallback
        self._execution_warnings = 0            # HUD-before-update violations
        self._color_extraction_errors = 0       # Failed color extractions
        
    # ==========================================
    # PUBLIC API - MAIN ENTRY POINTS
    # ==========================================
    
    def update(self, global_coh_tensor, raw_bgr_frame=None):
        """
        Find coherent workers for CURRENT FRAME.
        
        CRITICAL: Must be called in get_inputs_tensor() before HUD rendering.
        
        Args:
            global_coh_tensor: Tensor of coherence values for all workers
            raw_bgr_frame: Optional raw BGR frame for on-demand color extraction
            
        Returns: List of coherent worker data dicts for each scale
        """
        results = []
        coherences = []
        u_values = [250.0] * 4  # Initialize with neutral gray (250/500)
        
        for scale_idx, (start, end, gh, gw, color, name) in enumerate(self.scales):
            # Find coherent worker
            worker_data = self._find_coherent_in_scale(scale_idx, global_coh_tensor)
            results.append(worker_data)
            coherences.append(worker_data['coherence'])
            
            # Extract YUV color if we have frame
            if raw_bgr_frame is not None and not worker_data['is_fallback']:
                u_value = self._extract_yuv_color(
                    scale_idx=scale_idx,
                    sector_idx=worker_data['sector_idx'],
                    grid_w=gw,
                    grid_h=gh,
                    bgr_frame=raw_bgr_frame
                )
                u_values[scale_idx] = u_value
            else:
                u_values[scale_idx] = 250.0  # Neutral fallback
        
        # Store frame-local state
        self._current_workers = results
        self._coherence_values = coherences
        self._u_values = u_values
        self._is_valid = True
        self._frame_id += 1
        
        return results
    
    def get_scale_info(self, scale_idx):
        """
        Get coherent worker data for a specific scale.
        
        WARNING: Only valid if called AFTER update() in same frame!
        
        Args:
            scale_idx: 0-3 (scale index)
            
        Returns: Dict with worker data or None if invalid/invalid index
        """
        if not self._is_valid or scale_idx < 0 or scale_idx >= 4:
            return None
        return self._current_workers[scale_idx]
    
    def get_all_scales_info(self):
        """
        Get coherent worker data for ALL scales.
        
        Returns: List of 4 dicts (one per scale) or empty list if invalid
        """
        if not self._is_valid:
            return []
        return self._current_workers.copy()
    
    # ==========================================
    # PUBLIC API - COLOR ACCESS
    # ==========================================
    
    def get_scale_u_value(self, scale_idx):
        """
        Get YUV U-channel value for coherent worker in scale.
        
        Returns: U value normalized to 0-500 range, or 250.0 if invalid
        """
        if not self._is_valid or scale_idx < 0 or scale_idx >= 4:
            return 250.0  # Neutral gray
        return self._u_values[scale_idx]
    
    def get_all_u_values(self):
        """
        Get YUV U-channel values for all scales.
        
        Returns: List of 4 U values (0-500 range) or empty list if invalid
        """
        if not self._is_valid:
            return []
        return self._u_values.copy()
    
    # ==========================================
    # PUBLIC API - VALIDATION & DEBUGGING
    # ==========================================
    
    def validate_execution_order(self, caller='hud'):
        """
        Validate that HUD is rendered AFTER input computation.
        
        Args:
            caller: 'hud' or 'input' (for debug messages)
            
        Returns: (is_valid, warning_message)
        """
        if caller == 'hud' and not self._is_valid:
            self._execution_warnings += 1
            msg = (f"⏰ EXECUTION ORDER VIOLATION: HUD rendered before update()!\n"
                   f"   Frame {self._frame_id} - Tracker not initialized for this frame")
            return False, msg
        
        return True, f"Frame {self._frame_id} ✓"
    
    def get_diagnostics(self):
        """
        Get diagnostic information for debugging/telemetry.
        
        Returns: Dict with diagnostic metrics
        """
        return {
            'frame_id': self._frame_id,
            'is_valid': self._is_valid,
            'fallback_counts': self._fallback_counts.copy(),
            'execution_warnings': self._execution_warnings,
            'color_errors': self._color_extraction_errors,
            'avg_coherence': np.mean(self._coherence_values) if self._coherence_values else 0.0,
            'u_values': self._u_values.copy(),
            'u_min': min(self._u_values) if self._u_values else 250.0,
            'u_max': max(self._u_values) if self._u_values else 250.0,
        }
    
    # ==========================================
    # PUBLIC API - HUD UTILITIES
    # ==========================================
    
    # ==========================================
    # FIX FOR v79 - REPLACE THIS METHOD in CoherentWorkerTracker
    # ==========================================

    def get_hud_drawing_info(self, frame_width, frame_height):
        """
        Calculate HUD drawing parameters for all scales.
        FIXED: Integer division guarantees no gaps, no overlap, no edge-snap needed.
        """
        if not self._is_valid:
            return []
        
        drawing_info = []
        for scale_idx, scale_data in enumerate(self._current_workers):
            if not scale_data:
                continue
                
            gw, gh = scale_data['grid_w'], scale_data['grid_h']
            sector_idx = scale_data['sector_idx']
            
            row = sector_idx // gw
            col = sector_idx % gw
            
            # ==========================================
            # 🔧 THE FIX: Pure integer math, no floats, no edge-snap
            # ==========================================
            x1 = (col * frame_width) // gw
            y1 = (row * frame_height) // gh
            x2 = ((col + 1) * frame_width) // gw
            y2 = ((row + 1) * frame_height) // gh
            
            thickness = max(1, 4 - scale_idx)
            
            drawing_info.append({
                'scale_idx': scale_idx,
                'rect': (x1, y1, x2, y2),
                'color': scale_data['color'],
                'thickness': thickness,
                'coherence': scale_data['coherence'],
                'label': f"S{scale_idx}" if scale_idx >= 2 else None,
                'is_fallback': scale_data.get('is_fallback', False)
            })
        
        return drawing_info
    
    # ==========================================
    # PRIVATE IMPLEMENTATION
    # ==========================================
    
    def _find_coherent_in_scale(self, scale_idx, global_coh):
        """
        Find most coherent worker in a single scale.
        
        Private method - called by update() for each scale.
        """
        start, end, gh, gw, color, name = self.scales[scale_idx]
        scale_coherences = global_coh[start:end]
        
        # Handle empty scale (shouldn't happen, but defensive)
        if len(scale_coherences) == 0:
            self._fallback_counts[scale_idx] += 1
            if self._fallback_counts[scale_idx] == 1:  # Log first occurrence
                print(f"⚠️  CoherentWorkerTracker: Scale {scale_idx} ({name}) empty")
            
            return {
                'scale_idx': scale_idx,
                'scale_name': name,
                'worker_idx': start,      # Fallback: first worker
                'sector_idx': 0,
                'grid_h': gh,
                'grid_w': gw,
                'color': color,
                'coherence': 0.0,
                'is_fallback': True
            }
        
        # Find most coherent worker
        best_idx_rel = torch.argmax(scale_coherences).item()
        coherence_val = scale_coherences[best_idx_rel].item()
        
        return {
            'scale_idx': scale_idx,
            'scale_name': name,
            'worker_idx': start + best_idx_rel,
            'sector_idx': best_idx_rel,
            'grid_h': gh,
            'grid_w': gw,
            'color': color,
            'coherence': coherence_val,
            'is_fallback': False
        }
    
    def _extract_yuv_color(self, scale_idx, sector_idx, grid_w, grid_h, bgr_frame):
        """
        Extract YUV U-channel color from SPECIFIC sector ONLY.
        
        EFFICIENT: Converts only the coherent sector, not entire frame.
        Returns: U value normalized to 0-500 range (matching luminance range)
        """
        try:
            h, w, _ = bgr_frame.shape
            sw, sh = w // grid_w, h // grid_h
            
            # Calculate sector bounds
            row = sector_idx // grid_w
            col = sector_idx % grid_w
            
            y1 = row * sh
            y2 = min((row + 1) * sh, h)
            x1 = col * sw
            x2 = min((col + 1) * sw, w)
            
            # Validate bounds
            if y2 <= y1 or x2 <= x1:
                self._color_extraction_errors += 1
                return 250.0
            
            # Extract BGR sector
            bgr_sector = bgr_frame[y1:y2, x1:x2]
            
            if bgr_sector.size == 0:
                self._color_extraction_errors += 1
                return 250.0
            
            # Convert ONLY THIS SECTOR to YUV
            yuv_sector = cv2.cvtColor(bgr_sector, cv2.COLOR_BGR2YUV)
            
            # Extract U channel (Blue projection)
            u_channel = yuv_sector[:,:,1]  # 0-255 range
            
            # Calculate mean U value
            u_mean = np.mean(u_channel)  # 0-255
            
            # Normalize to 0-500 range (matches luminance range)
            u_value = (u_mean / 255.0) * 500.0
            
            # Clamp to valid range
            return np.clip(u_value, 0.0, 500.0)
            
        except Exception as e:
            self._color_extraction_errors += 1
            if self._color_extraction_errors < 5:  # Log first few errors
                print(f"⚠️  Color extraction error (scale {scale_idx}): {e}")
            return 250.0  # Neutral gray fallback
    
    # ==========================================
    # RESET FOR NEXT FRAME (CALLED BY MAIN LOOP)
    # ==========================================
    
    def reset_for_next_frame(self):
        """
        Prepare tracker for next frame.
        
        CALL FROM MAIN LOOP after HUD rendering.
        """
        self._is_valid = False
        # Note: We keep _current_workers until next update() for debugging
        # but they're marked invalid via _is_valid flag


# ==========================================
# 🔊 PURE CARRIER VOICE WORKER (FLIPPED φ-SCALED THRESHOLDS + PHYSICS)
# ==========================================

class PureCarrierVoice:
    """
    Flipped φ-Scaled Voice with Physics Engine.
    
    ARCHITECTURE:
    1. FLIPPED φ-SCALED THRESHOLDS: Biggest jumps at start (M→B), smallest at end (H→S)
    2. VELOCITY SELECTS BRACKET: Not normalized 0-1, uses actual thresholds
    3. PHYSICS-BASED MOVEMENT: Mass-spring-damper for smooth transitions
    4. FORMANT LERP: Smooth formant movement toward interpolated phonemes
    5. DECOUPLED AMPLITUDE: Volume independent of position
    
    RESULT: Dark-heavy voice with few high ranges, adaptive to session max.
    """
    
    def __init__(self, sample_rate=44100, use_articulation=True):
        self.sr = sample_rate
        self.worker_idx = 1548
        
        # Phase continuity
        self.audio_phase = 0.0
        self.envelope_phase = 0.0
        
        # ==========================================
        # LERPED FORMANT STATE
        # ==========================================
        self.current_f1 = 300.0
        self.current_f2 = 600.0
        self.current_f3 = 2200.0
        self.current_f4 = 3300.0
        self.current_f5 = 3750.0
        
        self.bw1 = 80.0
        self.bw2 = 90.0
        self.bw3 = 120.0
        self.bw4 = 150.0
        self.bw5 = 200.0
        
        self.formant_lerp_factor = 0.12
        
        self.target_f1 = 300.0
        self.target_f2 = 600.0
        self.target_f3 = 2200.0
        self.target_f4 = 3300.0
        self.target_f5 = 3750.0
        
        # ==========================================
        # FLIPPED φ-SCALED THRESHOLD SYSTEM
        # ==========================================
        self.min_threshold = 20.0          # Below this = silent was 1.5
        self.observed_max = 12.0          # Starting max (will adapt)
        self.last_reported_max = 0.0      # For print spam prevention
        
        self.phi = 1.618033988749895      # Golden ratio
        self.phi_thresholds = []          # Will hold current φ-scaled thresholds
        self.effective_ratio = 1.0        # Current geometric ratio (starts at 1.0)
        
        # Phoneme position tracking (physics-driven, 0-1 for array index)
        self.current_phoneme_position = 0.0
        self.position_lerp_factor = 0.20
        
        # ==========================================
        # PHYSICS ENGINE (VELOCITY-DAMPED)
        # ==========================================
        self.pos_velocity = 0.0
        self.pos_damping = 0.9
        self.max_velocity = 0.05
        
        # Current state
        self.current_pitch = 440.0
        self.current_amplitude = 0.0
        self.current_articulation = 5.0
        self.current_velocity = 0.0
        self.current_direction = 1
        
        # Controls
        self.use_articulation = use_articulation
        self.mode_active = {
            'webcam': True,
            'ale': False,
            'screencap': False,
            'screen': False,
            'viewer': False,
        }
        self.is_sounding = False
        
        # Phoneme spectrum
        self._init_phoneme_spectrum()
        
        # Calculate initial φ-scaled thresholds
        self._update_phi_thresholds()
        
        # Audio output
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sr,
            output=True,
            frames_per_buffer=1024
        )
        
        print(f"🎵 Flipped φ-Scaled Voice at worker {self.worker_idx}")
        print(f"   Thresholds: {self.min_threshold:.1f}-{self.observed_max:.1f} (adaptive)")
        print(f"   Effective ratio: {self.effective_ratio:.3f} (target φ={self.phi:.3f})")
        print(f"   Physics: damping={self.pos_damping}, velocity_limit={self.max_velocity}")
    
    def _init_phoneme_spectrum(self):
        """5-FORMANT PHONEME DATA for continuous interpolation"""
        # Grouped by ARTICULATION PLACE (front→back of mouth)
        self.PHONEME_SPECTRUM = [
            # ====== BILABIALS (lips) ======
            {'name': 'M', 'f1': 250, 'f2': 800, 'f3': 2200, 'f4': 3400, 'f5': 3950},
            {'name': 'B', 'f1': 200, 'f2': 750, 'f3': 2400, 'f4': 3400, 'f5': 3800},
            {'name': 'P', 'f1': 200, 'f2': 800, 'f3': 2200, 'f4': 3300, 'f5': 3800},
            {'name': 'W', 'f1': 300, 'f2': 600, 'f3': 2200, 'f4': 3300, 'f5': 3750},
            
            # ====== LABIODENTAL (lip-teeth) ======
            {'name': 'F', 'f1': 200, 'f2': 1400, 'f3': 3500, 'f4': 3650, 'f5': 4200},
            
            # ====== ALVEOLAR (tongue tip) ======
            {'name': 'R', 'f1': 450, 'f2': 1100, 'f3': 1550, 'f4': 3500, 'f5': 4050},
            {'name': 'L', 'f1': 350, 'f2': 1200, 'f3': 2900, 'f4': 3550, 'f5': 4100},
            
            # ====== VOWELS (open) ======
            {'name': 'U', 'f1': 300, 'f2': 870, 'f3': 2240, 'f4': 3300, 'f5': 3850},
            {'name': 'O', 'f1': 570, 'f2': 840, 'f3': 2410, 'f4': 3350, 'f5': 3900},
            {'name': 'A', 'f1': 730, 'f2': 1090, 'f3': 2440, 'f4': 3450, 'f5': 4000},
            {'name': 'E', 'f1': 530, 'f2': 1840, 'f3': 2480, 'f4': 3750, 'f5': 4300},
            {'name': 'I', 'f1': 270, 'f2': 2290, 'f3': 3010, 'f4': 3850, 'f5': 4400},
            
            # ====== PALATAL (tongue body) ======
            {'name': 'Y', 'f1': 250, 'f2': 2000, 'f3': 2800, 'f4': 3800, 'f5': 4350},
            
            # ====== POSTALVEOLAR (tongue blade) ======
            {'name': 'SH', 'f1': 300, 'f2': 2200, 'f3': 3200, 'f4': 4200, 'f5': 4500},
            
            # ====== VELAR (back of tongue) ======
            {'name': 'K', 'f1': 300, 'f2': 1400, 'f3': 3000, 'f4': 3700, 'f5': 4250},
            {'name': 'H', 'f1': 800, 'f2': 1400, 'f3': 2500, 'f4': 3600, 'f5': 4150},
            
            # ====== SIBILANT (extreme) ======
            {'name': 'S', 'f1': 200, 'f2': 3500, 'f3': 6500, 'f4': 7500, 'f5': 8500},
        ]
    
    def _update_phi_thresholds(self):
        """
        Calculate flipped φ-scaled thresholds based on current min/max.
        Biggest jumps at start (M→B), smallest at end (H→S).
        """
        a = self.min_threshold
        b = self.observed_max
        n = len(self.PHONEME_SPECTRUM)
        
        if n <= 1 or b <= a:
            self.phi_thresholds = [a, b] if n > 1 else [a]
            self.effective_ratio = 1.0
            return
        
        # Calculate effective geometric ratio that fits the range
        self.effective_ratio = (b / a) ** (1.0 / (n - 1))
        
        # Create DECREASING gaps (biggest first)
        total_range = b - a
        
        # First gap size (for M→B) using geometric series formula
        # Sum of gaps: total_range = g * (1 - r^(n-1)) / (1 - r) where r = 1/effective_ratio
        r = 1.0 / self.effective_ratio  # Ratio between consecutive gaps (< 1 for decreasing)
        
        if abs(r - 1.0) < 1e-10:
            # If r ≈ 1, use equal gaps
            first_gap = total_range / (n - 1)
            gaps = [first_gap] * (n - 1)
        else:
            # Geometric decreasing gaps
            sum_series = (1 - r ** (n - 1)) / (1 - r)
            first_gap = total_range / sum_series
            gaps = []
            current = first_gap
            for i in range(n - 1):
                gaps.append(current)
                current *= r  # Each gap is r times smaller than previous
        
        # Build thresholds
        self.phi_thresholds = [a]
        current = a
        for gap in gaps:
            current += gap
            self.phi_thresholds.append(current)
        
        # Ensure last threshold is exactly b
        self.phi_thresholds[-1] = b
        
        # Print threshold info
        if self.phi_thresholds[-1] > self.last_reported_max:
            print(f"  🔢 φ-thresholds updated: ratio={self.effective_ratio:.3f}")
    
    def _velocity_to_phoneme_position(self, velocity_magnitude):
        """
        Convert velocity to phoneme position using φ-scaled thresholds.
        Returns position 0-1 for array indexing.
        """
        v = max(self.min_threshold, min(velocity_magnitude, self.observed_max))
        
        # Find which bracket velocity falls into
        for i in range(len(self.phi_thresholds) - 1):
            if self.phi_thresholds[i] <= v < self.phi_thresholds[i + 1]:
                # Linear interpolation within this bracket
                bracket_start = self.phi_thresholds[i]
                bracket_end = self.phi_thresholds[i + 1]
                bracket_width = bracket_end - bracket_start
                
                if bracket_width > 0:
                    pos_in_bracket = (v - bracket_start) / bracket_width
                else:
                    pos_in_bracket = 0.0
                
                # Convert to overall position (0-1 for array index)
                phoneme_position = (i + pos_in_bracket) / (len(self.PHONEME_SPECTRUM) - 1)
                return max(0.0, min(1.0, phoneme_position))
        
        # At or above last threshold
        return 1.0
    
    def _get_interpolated_phoneme(self, position):
        """Continuous phoneme interpolation at given position (0-1)."""
        continuous_idx = position * (len(self.PHONEME_SPECTRUM) - 1)
        idx_low = int(continuous_idx)
        idx_high = min(idx_low + 1, len(self.PHONEME_SPECTRUM) - 1)
        weight = continuous_idx - idx_low
        
        phoneme_low = self.PHONEME_SPECTRUM[idx_low]
        phoneme_high = self.PHONEME_SPECTRUM[idx_high]
        
        return {
            'name': f"{phoneme_low['name']}→{phoneme_high['name']}",
            'f1': phoneme_low['f1'] * (1-weight) + phoneme_high['f1'] * weight,
            'f2': phoneme_low['f2'] * (1-weight) + phoneme_high['f2'] * weight,
            'f3': phoneme_low['f3'] * (1-weight) + phoneme_high['f3'] * weight,
            'f4': phoneme_low['f4'] * (1-weight) + phoneme_high['f4'] * weight,
            'f5': phoneme_low['f5'] * (1-weight) + phoneme_high['f5'] * weight,
        }
    
    def _get_formant_gain(self, frequency, formant_freq, bandwidth):
        """Calculate resonance gain for a frequency near a formant"""
        distance = abs(frequency - formant_freq)
        return 1.0 / (1.0 + (distance / bandwidth) ** 2)
    
    def _update_formant_targets(self, target_phoneme):
        """Set target formant frequencies for LERPing"""
        self.target_f1 = target_phoneme['f1']
        self.target_f2 = target_phoneme['f2']
        self.target_f3 = target_phoneme['f3']
        self.target_f4 = target_phoneme['f4']
        self.target_f5 = target_phoneme['f5']
    
    def _lerp_formants(self):
        """Smoothly move current formants toward target formants"""
        self.current_f1 += (self.target_f1 - self.current_f1) * self.formant_lerp_factor
        self.current_f2 += (self.target_f2 - self.current_f2) * self.formant_lerp_factor
        self.current_f3 += (self.target_f3 - self.current_f3) * self.formant_lerp_factor
        self.current_f4 += (self.target_f4 - self.current_f4) * self.formant_lerp_factor
        self.current_f5 += (self.target_f5 - self.current_f5) * self.formant_lerp_factor
    
    def _get_harmonic_gain(self, harmonic_freq):
        """Calculate total formant gain for a harmonic frequency"""
        f1_gain = self._get_formant_gain(harmonic_freq, self.current_f1, self.bw1)
        f2_gain = self._get_formant_gain(harmonic_freq, self.current_f2, self.bw2)
        f3_gain = self._get_formant_gain(harmonic_freq, self.current_f3, self.bw3)
        f4_gain = self._get_formant_gain(harmonic_freq, self.current_f4, self.bw4)
        f5_gain = self._get_formant_gain(harmonic_freq, self.current_f5, self.bw5)
        
        return (
            # f1_gain * 0.35 +
            # f2_gain * 0.35 +
            # f3_gain * 0.15 +
            # f4_gain * 0.10 +
            # f5_gain * 0.05
            f1_gain * 0.60 +  # The "Body"
            f2_gain * 0.30 +  # The "Clarity"
            f3_gain * 0.07 +  # The "Character" (Muted)
            f4_gain * 0.02 +  # High air (Near-silent)
            f5_gain * 0.01    # High air (Near-silent) 
        )
    
    # ==========================================
    # REQUIRED PUBLIC METHODS
    # ==========================================
    
    def set_mode(self, mode):
        self.current_mode = mode
        self.is_sounding = self.mode_active.get(mode, False)
    
    def enable_for_mode(self, mode, enable=True):
        if mode in self.mode_active:
            self.mode_active[mode] = enable
            print(f"  🔊 Voice {'enabled' if enable else 'disabled'} for {mode} mode")
    
    def update_from_magi(self, magi_hive):
        idx = self.worker_idx
        
        if idx >= len(magi_hive.freq):
            return False
        
        freq = magi_hive.freq[idx].item()
        delay = magi_hive.delay[idx].item()
        voice_value = magi_hive.s_filtered[idx].item()
        
        self.current_direction = 1 if voice_value >= 0 else -1
        velocity_magnitude = abs(voice_value)
        
        # ==========================================
        # 1. UPDATE SESSION MAX & φ-THRESHOLDS
        # ==========================================
        if velocity_magnitude > self.observed_max:
            old_max = self.observed_max
            self.observed_max = velocity_magnitude
            
            # Only print if significantly different
            if abs(self.observed_max - self.last_reported_max) > 0.1:
                print(f"  🔊 New session maximum: {self.observed_max:.1f} (was {old_max:.1f})")
                self.last_reported_max = self.observed_max
            
            # Recalculate φ-scaled thresholds with new max
            self._update_phi_thresholds()
        
        # ==========================================
        # 2. CONVERT VELOCITY TO PHONEME POSITION
        # ==========================================
        if velocity_magnitude < self.min_threshold:
            target_position = 0.0
        else:
            # Use φ-scaled thresholds (NOT normalized 0-1!)
            target_position = self._velocity_to_phoneme_position(velocity_magnitude)
        
        # ==========================================
        # 3. PHYSICS-BASED MOVEMENT
        # ==========================================
        force = target_position - self.current_phoneme_position
        
        # Anti-stall micro-force
        if 0 < abs(force) < 0.001:
            force = 0.001 * np.sign(force)
        
        # Update velocity with damping
        self.pos_velocity = (self.pos_velocity * self.pos_damping) + (force * self.position_lerp_factor)
        self.pos_velocity = np.clip(self.pos_velocity, -self.max_velocity, self.max_velocity)
        
        # Apply velocity to position
        self.current_phoneme_position += self.pos_velocity
        self.current_phoneme_position = max(0.0, min(1.0, self.current_phoneme_position))
        
        # Store for logging
        self.current_velocity = self.current_phoneme_position * 1000.0
        
        # ==========================================
        # 4. UPDATE FORMANTS
        # ==========================================
        target_phoneme = self._get_interpolated_phoneme(self.current_phoneme_position)
        self._update_formant_targets(target_phoneme)
        self._lerp_formants()
        
        # ==========================================
        # 5. PITCH & AMPLITUDE
        # ==========================================
        freq_norm = np.clip((freq - 0.5) / 69.5, 0.0, 1.0)
        self.current_pitch = 80.0 * (1500.0 / 80.0) ** freq_norm
        
        self.current_amplitude = np.tanh(voice_value / 200.0)
        
        delay_norm = np.clip((delay - 1.0) / 99.0, 0.0, 1.0)
        self.current_articulation = 1.0 + delay_norm * 14.0
        
        return velocity_magnitude > self.min_threshold
    
    
    def generate_sound(self, frame_duration=0.02):
        """
        5-FORMANT HARMONIC GENERATOR with physics-driven articulation.
        Ultra-smooth, pop-free voice with φ-spaced articulation and momentum physics.
        """
        if abs(self.current_amplitude) < 0.01:
            return None
        
        num_samples = int(self.sr * frame_duration)
        sample_indices = np.arange(num_samples)
        
        # ==========================================
        # HARMONIC CONFIGURATION
        # ==========================================
        # if self.current_direction > 0:
        #     # Harmonic (aligned)
        #     harmonic_ratios = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        #     harmonic_weights = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
        # else:
        #     # Inharmonic (resistant)
        #     harmonic_ratios = [1.0, 1.5, 2.5, 3.5, 4.5, 5.5]
        #     harmonic_weights = [1.0, 0.7, 0.5, 0.3, 0.15, 0.07]
        
        if self.current_direction > 0:
            # Steep logarithmic decay (Darker)
            harmonic_ratios = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            harmonic_weights = [1.0, 0.4, 0.15, 0.08, 0.04, 0.01] 
        else:
        # Dirty but still muted
            harmonic_ratios = [1.0, 1.5, 2.5, 3.5, 4.5, 5.5]
            harmonic_weights = [1.0, 0.3, 0.1, 0.05, 0.02, 0.01]

        # ==========================================
        # PHASE-ALIGNED HARMONIC GENERATION
        # ==========================================
        audio_phase_inc = 2.0 * np.pi * self.current_pitch / self.sr
        base_phase = self.audio_phase + audio_phase_inc * sample_indices
        
        wave = np.zeros(num_samples)
        total_weight = 0.0
        
        for ratio, base_weight in zip(harmonic_ratios, harmonic_weights):
            harmonic_freq = self.current_pitch * ratio
            
            if harmonic_freq > self.sr / 2:
                continue
            
            formant_gain = self._get_harmonic_gain(harmonic_freq)
            combined_weight = base_weight * formant_gain
            
            wave += combined_weight * np.sin(base_phase * ratio)
            total_weight += combined_weight
        
        # ==========================================
        # NORMALIZE AND APPLY AMPLITUDE
        # ==========================================
        if total_weight > 0:
            wave = (wave / total_weight) * abs(self.current_amplitude)
        
        # ==========================================
        # ARTICULATION ENVELOPE
        # ==========================================
        if self.use_articulation:
            envelope_phase_inc = 2.0 * np.pi * self.current_articulation / self.sr
            envelope_phase = self.envelope_phase + envelope_phase_inc * sample_indices
            wave *= (0.5 + 0.5 * np.sin(envelope_phase))
            self.envelope_phase = (self.envelope_phase + envelope_phase_inc * num_samples) % (2.0 * np.pi)
        
        # ==========================================
        # CONSISTENT AMPLITUDE
        # ==========================================
        wave *= 2.2
        
        # Gentle soft clip
        peak = np.max(np.abs(wave))
        if peak > 0.95:
            wave = np.tanh(wave * 0.9) * 1.1
        
        # ==========================================
        # PHASE CONTINUITY
        # ==========================================
        self.audio_phase = (self.audio_phase + audio_phase_inc * num_samples) % (2.0 * np.pi)
        
        return wave.astype(np.float32)
    
    def generate_sound_simple(self, frame_duration=0.02):
        """Simple fallback"""
        if abs(self.current_amplitude) < 0.01:
            return None
        
        num_samples = int(self.sr * frame_duration)
        sample_indices = np.arange(num_samples)
        
        # Use F2 as pitch reference
        lerped_pitch = self.current_f2 / 5.0
        
        audio_phase_inc = 2.0 * np.pi * lerped_pitch / self.sr
        audio_phase = self.audio_phase + audio_phase_inc * sample_indices
        
        wave = self.current_amplitude * np.sin(audio_phase)
        
        if self.use_articulation:
            envelope_phase_inc = 2.0 * np.pi * self.current_articulation / self.sr
            envelope_phase = self.envelope_phase + envelope_phase_inc * sample_indices
            wave *= (0.5 + 0.5 * np.sin(envelope_phase))
            self.envelope_phase = (self.envelope_phase + envelope_phase_inc * num_samples) % (2.0 * np.pi)
        
        wave *= 2.0
        
        self.audio_phase = (self.audio_phase + audio_phase_inc * num_samples) % (2.0 * np.pi)
        
        return wave.astype(np.float32)
    
    def speak(self, magi_hive):
        is_active = self.update_from_magi(magi_hive)
        if is_active and self.is_sounding:
            audio = self.generate_sound()
            if audio is not None:
                try: 
                    self.stream.write(audio.tobytes())
                except Exception:
                    pass
    
    def cleanup(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        except:
            pass

class ScreenGrabber:
    """
    3x Resolution Screen Grabber using mss with transparent UI.
    Captures at 1920x1440 and downsamples to 640x480 for MaGi.
    """
    def __init__(self, root, width=640, height=480):
        self.root = root
        self.target_w = width
        self.target_h = height
        
        # Triple the capture/window resolution
        self.cap_w = width * 3
        self.cap_h = height * 3
        
        self.root.title(f"MaGi Grabber - View: {self.cap_w}x{self.cap_h}")
        self.root.geometry(f"{self.cap_w}x{self.cap_h}")
        self.root.resizable(False, False)
        
        # Initialize mss
        self.sct = mss.mss()
        
        # Transparent window setup
        self.root.attributes('-alpha', 0.3)
        self.root.configure(bg='black')
        
        self.current_frame = None
        self.window_visible = False
        
        # UI controls
        self.control_frame = tk.Frame(root, bg='gray20')
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        self.alpha_btn = tk.Button(
            self.control_frame,
            text="Toggle Window",
            command=self.toggle_alpha,
            bg='#555',
            fg='white',
            font=('Arial', 10)
        )
        self.alpha_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(
            self.control_frame, 
            text=f"MaGi Input: {width}x{height} (from {self.cap_w}x{self.cap_h})",
            font=('Arial', 9),
            bg='gray20',
            fg='white'
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

    def toggle_alpha(self):
        self.window_visible = not self.window_visible
        self.root.attributes('-alpha', 1.0 if self.window_visible else 0.3)

    def capture_frame(self):
        try:
            x, y = self.root.winfo_x(), self.root.winfo_y()
            off_x, off_y = (8, 31) if platform.system() == 'Windows' else (0, 25)
            
            monitor = {
                "top": y + off_y, 
                "left": x + off_x, 
                "width": self.cap_w, 
                "height": self.cap_h
            }
            
            # Capture 3x high-res frame
            screenshot = self.sct.grab(monitor)
            frame_bgra = np.array(screenshot)
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            
            # Downsample to MaGi target resolution
            self.current_frame = cv2.resize(frame_bgr, (self.target_w, self.target_h))
            return self.current_frame
            
        except Exception:
            return None

class RuntimeCommandListener:
    def __init__(self, magi_system):
        self.magi = magi_system
        self.running = True
        self.command_queue = queue.Queue()
        self.listener_thread = threading.Thread(target=self._listen_for_commands, daemon=True)
        self.listener_thread.start()
    
    def _listen_for_commands(self):
        """FIXED: Uses a blocking read. This reliably captures input but requires ENTER."""
        while self.running:
            try:
                # sys.stdin.readline() blocks until the user presses Enter.
                line = sys.stdin.readline().strip()
                if line:
                    self.command_queue.put(line)
            except:
                break
    
    def process_commands(self):
        while not self.command_queue.empty():
            self._execute_command(self.command_queue.get())
    
    def _execute_command(self, cmd):
        parts = cmd.lower().split()
        if not parts: return
        command = parts[0]
        
        if command == 'mode':
            if len(parts) < 2:
                print(f"Current mode: {self.magi.mode}")
                return
            new_mode = parts[1]
            args = parts[2:]
            
            if new_mode == 'webcam':
                self.magi.video_source.switch_to_webcam()
                self.magi.mode = 'webcam'
            elif new_mode == 'ale':
                if not args:
                    print("⚠️  Usage: mode ale <rom_path>")
                    return
                self.magi.video_source.switch_to_ale(args[0])
                self.magi.mode = 'ale'
            elif new_mode == 'screencap':
                region = args[0] if args else "0,0,640,480"
                self.magi.video_source.switch_to_screencap(region)
                self.magi.mode = 'screencap'
            elif new_mode == 'screen':
                self.magi.video_source.switch_to_screen_grab()
                self.magi.mode = 'screen'
            elif new_mode == 'viewer':
                folder = args[0] if args else "images"
                self.magi.video_source.switch_to_viewer(folder)
                self.magi.mode = 'viewer'
            elif new_mode == 'remote':
                if not args:
                    print("⚠️  Usage: mode remote <pi_ip>")
                    return
                pi_ip = args[0]
                if self.magi.remote_sender:
                    self.magi.remote_sender.close()
                self.magi.remote_sender = RemoteCommandSender(pi_ip)
                self.magi.video_source.switch_to_remote(pi_ip)
                self.magi.mode = 'remote'

        elif command == 'remote':
            # remote stat — show live battery status from Pi
            subcmd = parts[1] if len(parts) > 1 else 'stat'
            if subcmd == 'stat':
                if self.magi.remote_sender is None:
                    print("⚠️  Not in remote mode")
                    return
                print("🔋 Requesting battery from Pi...")
                result = self.magi.remote_sender.request_battery()
                if result:
                    pct, v, a = result
                    if pct < 0:
                        print("  Battery monitor unavailable on Pi")
                    else:
                        bar_len = int(pct / 5)
                        bar = "█" * bar_len + "░" * (20 - bar_len)
                        warn = " ⚠️  LOW" if pct < REMOTE_BATTERY_LOW_PCT else ""
                        print(f"  [{bar}] {pct:.1f}%{warn}")
                        print(f"  Voltage: {v:.3f} V")
                        print(f"  Current: {a:.3f} A")
                        print(f"  Power:   {v*a:.3f} W")
                else:
                    print("  ⚠️  No response from Pi")
            else:
                print("Usage: remote stat")

        # ✅ v141: AUDIO SOURCE CONTROL
        elif command == 'mic':
            if not hasattr(self.magi, 'av_capture') or self.magi.av_capture is None:
                print("⚠️  Audio capture not initialised yet.")
                return

            if len(parts) < 2:
                # Status
                cap = self.magi.av_capture
                src = cap.source.upper()
                if cap.source == 'remote':
                    print(f"🎙️  Mic: {src} @ {cap.remote_ip}  "
                          f"(audio UDP:{REMOTE_AUDIO_PORT}, DOA UDP:{REMOTE_DOA_PORT})")
                else:
                    print(f"🎙️  Mic: {src} (PyAudio @ 44.1 kHz mono)")
                speech, angle, age = cap.get_doa()
                if age is None:
                    print(f"   DOA: no packets received yet.")
                else:
                    state = "SPEECH" if speech else "silence"
                    print(f"   DOA: {state}  angle={angle}°  age={age:.1f}s")
                print("   Usage: mic local | mic remote <ip>")
                return

            sub = parts[1]
            if sub == 'local':
                self.magi.av_capture.switch_to_local()
            elif sub == 'remote':
                if len(parts) < 3:
                    print("⚠️  Usage: mic remote <worker_ip>")
                    return
                self.magi.av_capture.switch_to_remote(parts[2])
            elif sub == 'doa':
                # Quick DOA-only readout
                speech, angle, age = self.magi.av_capture.get_doa()
                if age is None:
                    print("   DOA: no packets received yet.")
                else:
                    state = "SPEECH" if speech else "silence"
                    print(f"   DOA: {state}  angle={angle}°  age={age:.1f}s")
            else:
                print("Usage: mic [local | remote <ip> | doa]")

        elif command == 'chord':
            m     = self.magi
            # Determine whether to show current or last match
            show_last = len(parts) > 1 and parts[1].lower() == 'last'
            which = parts[2].lower() if len(parts) > 2 and not show_last else \
                    parts[1].lower() if len(parts) > 1 and not show_last else 'both'
            show  = []
            if show_last:
                src_audio = m.last_chord_audio
                src_video = m.last_chord_video
                label = "Last known match"
            else:
                src_audio = m.n_gravity_audio
                src_video = m.n_gravity_video
                label = "Current carrier"

            if which in ('both', 'audio'): show.append(('audio', src_audio))
            if which in ('both', 'video'): show.append(('video', src_video))

            def _climate(ce):
                if   ce < 0.05: return "🌊 Laminar"
                elif ce < 0.15: return "🌀 Creative"
                else:           return "⚡ Turbulent"

            for name, gs in show:
                print(f"\n🗺️  Continental Chord — {name.upper()} carrier ({label})")
                if not gs or gs.get('chord_size', 0) == 0:
                    print(f"  No match yet (streak not met or no memory found)")
                    continue

                # ── Matched memory ────────────────────────────────────────
                mfreq  = gs.get('mem_freq',  -1.0)
                mdelay = gs.get('mem_delay', -1.0)
                m_tce  = gs.get('mem_tension_ce', 0.0)
                m_tae  = gs.get('mem_tension_ae', 0.0)
                mver   = gs.get('mem_version',  0.0)
                morg   = gs.get('mem_origin',   '?')
                age    = gs.get('top_age',  0.0)
                ac     = gs.get('top_access', 0.0)
                fd     = gs.get('freq_delta', 0.0)
                dup    = gs.get('doppler_uncertainty', 1.0)
                gscale = gs.get('geo_scale', 0.7)

                print(f"  Matched memory:")
                print(f"    freq={mfreq:.2f} Hz   delay={mdelay:.2f} ms   origin={morg}   v{mver:.0f}")
                print(f"    age={age:.4f}   access={ac:.0f}")
                print(f"    tension CE={m_tce:.3f} rad  AE={m_tae:.3f} rad  climate={_climate(m_tce)}")
                print(f"    freqΔ={fd:.2f} Hz   doppler uncertainty={dup:.3f}   geo scale={gscale:.2f}")

                # ── Live carrier state ────────────────────────────────────
                cidx = gs.get('carrier_idx')
                if cidx is not None:
                    c_freq  = m.freq[cidx].item()
                    c_delay = m.delay[cidx].item()
                    p       = m.pos_6d[cidx]
                    def _wrap(d):
                        d = d % (2*math.pi)
                        return min(d, 2*math.pi - d)
                    c_tce = _wrap(p[0].item() - p[3].item())
                    c_tae = _wrap(p[2].item() - p[3].item())
                    print(f"  Live carrier (idx {cidx}):")
                    print(f"    freq={c_freq:.2f} Hz   delay={c_delay:.2f} ms")
                    print(f"    tension CE={c_tce:.3f} rad  AE={c_tae:.3f} rad  climate={_climate(c_tce)}")
                    df = abs(c_freq - mfreq)
                    dd = abs(c_delay - mdelay)
                    print(f"  Δfreq={df:.2f} Hz   Δdelay={dd:.2f} ms from matched memory")
                    if df < 2.0 and dd < 10.0:
                        print(f"  ✅ Same continent (freq+delay close)")
                    else:
                        print(f"  ⚠️  Drifted from matched continent")
        # ✅ v133: ROBOT ARM COMMANDS
        elif command == 'robot':
            if len(parts) < 2:
                mode_str = f"mode {self.magi.robot_mode}" if self.magi.robot_mode > 0 else "DISABLED"
                ip_str = f" → {self.magi.robot_sender.sim_ip}" if self.magi.robot_sender else ""
                active = sorted(self.magi._robot_active_workers)
                print(f"🦾 Robot arm: {mode_str}{ip_str}")
                print(f"   Active workers: {active if active else 'none'}")
                if self.magi.robot_sender:
                    rs = self.magi.robot_sender
                    print(f"   Position: [{rs.target_pos[0]:.3f}, {rs.target_pos[1]:.3f}, {rs.target_pos[2]:.3f}]")
                    print(f"   Wrist: rot={rs.wrist_rot:.3f}  tilt={rs.wrist_tilt:.3f}")
                    print(f"   Gripper: {rs.gripper_state}")
                # Show all robot worker phases
                for idx in range(1558, 1564):
                    name = self.magi.upe.homes[idx]['name'] if idx in self.magi.upe.homes else f"W{idx}"
                    active_mark = "🟢" if idx in self.magi._robot_active_workers else "⚫"
                    val = self.magi.s_filtered[idx].item()
                    pos = self.magi.pos_6d[idx].cpu().numpy().round(2)
                    print(f"   {active_mark} {name:<10} [{idx}]: val={val:.1f}  pos={pos}")
                return

            subcmd = parts[1]

            if subcmd == 'enable':
                if len(parts) < 3:
                    print("⚠️  Usage: robot enable <sim_ip>")
                    return
                sim_ip = parts[2]
                if self.magi.robot_sender:
                    self.magi.robot_sender.close()
                self.magi.robot_sender = RobotArmSender(sim_ip)
                self.magi.robot_mode = 1
                self.magi._robot_active_workers = ROBOT_MODE_WORKERS[1].copy()
                print(f"🦾 Robot arm ENABLED → {sim_ip}:{ROBOT_UDP_PORT} (mode 1: X, Y, Gripper)")

            elif subcmd == 'disable':
                if self.magi.robot_sender:
                    self.magi.robot_sender.close()
                    self.magi.robot_sender = None
                self.magi.robot_mode = 0
                self.magi._robot_active_workers = set()
                print("🦾 Robot arm DISABLED (workers frozen)")

            elif subcmd in ('1mode', '2mode', '3mode', '4mode'):
                mode_num = int(subcmd[0])
                if self.magi.robot_sender is None:
                    print("⚠️  Robot not enabled. Use: robot enable <ip>")
                    return
                self.magi.robot_mode = mode_num
                self.magi._robot_active_workers = ROBOT_MODE_WORKERS[mode_num].copy()
                worker_names = [self.magi.upe.homes[i]['name'] for i in sorted(self.magi._robot_active_workers)]
                print(f"🦾 Robot mode {mode_num}: {', '.join(worker_names)}")

            else:
                print("Usage: robot [enable <ip> | disable | 1mode | 2mode | 3mode | 4mode]")

        # ✅ ADD VOICE CONTROL COMMANDS
        elif command == 'voice':
            if len(parts) < 2:
                # Show current voice status
                print("Voice status:")
                for mode, active in self.magi.voice_carrier.mode_active.items():
                    status = "🟢 ACTIVE" if active else "🔴 INACTIVE"
                    print(f"  {mode}: {status}")
                return
            
            if len(parts) < 3:
                print("Usage: voice <enable/disable> <mode>")
                print("Modes: webcam, ale, screencap, screen, viewer")
                return
            
            subcmd, mode = parts[1], parts[2]
            if subcmd == 'enable':
                self.magi.voice_carrier.enable_for_mode(mode, True)
            elif subcmd == 'disable':
                self.magi.voice_carrier.enable_for_mode(mode, False)
            else:
                print(f"Unknown voice command: {subcmd}")
                print("Usage: voice <enable/disable> <mode>")
            
        elif command == 'save':
            self.magi.memory_bank.save(MEMORY_FILE)
            
        elif command == 'stats':
            m = self.magi
            # Main bank
            cap_pct = (m.memory_bank.size / m.memory_bank.max_memories) * 100.0
            n_cap_pct = (m.n_bank.size / m.n_bank.max_memories) * 100.0
            bh_result = getattr(m, 'black_hole_last_result', {})
            nbh_result = getattr(m, 'n_bh_last_result', {})
            bh_tension = bh_result.get('tension_factor', abs(bh_result.get('output', 0.0)) / 1500.0)
            nbh_tension = nbh_result.get('tension_factor', abs(nbh_result.get('output', 0.0)) / 1500.0)
            bh_active  = '🟢' if bh_result.get('is_active', False) else '⚫'
            nbh_active = '🟢' if nbh_result.get('is_active', False) else '⚫'
            bh_r  = m.black_hole_base_radius * (1.0 + bh_tension)
            nbh_r = m.n_bh_base_radius       * (1.0 + nbh_tension)

            print(f"\n📊 MaGi v102 — Mode: {m.mode.upper()}")
            print(f"  Workers: {m.n:,} | Step: {m.global_age:,} | Kernel: {'⚡ CUDA' if m.use_cuda_kernel else '🐢 Python'}")
            print()
            print(f"  ┌─ MAIN HYPERSPHERE ─────────────────────────────────")
            print(f"  │  Bank:    {m.memory_bank.size:>8,} / {m.memory_bank.max_memories:,}  ({cap_pct:.1f}%)")
            print(f"  │  BH idx:  {m.black_hole_worker_idx}  {bh_active}  tension={bh_tension:.3f}  r={bh_r:.4f}")
            print(f"  │  In field: {m.black_hole_memories_in_field:,}  |  deletions(session): {m.black_hole_daily_deletions:,}")
            print(f"  │  Phase:   {m.pos_6d[m.black_hole_worker_idx].cpu().numpy().round(3)}")
            if m.black_hole_daily_deletions > 0:
                ratio = m.black_hole_creation_count / m.black_hole_daily_deletions
                print(f"  │  C:D ratio: {ratio:.1f}:1  (created {m.black_hole_creation_count:,})")
            print(f"  └─────────────────────────────────────────────────────")
            print()
            print(f"  ┌─ N HYPERSPHERE (narrative / temporal) ─────────────")
            print(f"  │  Bank:    {m.n_bank.size:>8,} / {m.n_bank.max_memories:,}  ({n_cap_pct:.1f}%)")
            print(f"  │  N BH idx:{m.n_bh_worker_idx}  {nbh_active}  tension={nbh_tension:.3f}  r={nbh_r:.4f}")
            print(f"  │  In field: {m.n_bh_memories_in_field:,}  |  deletions(step): {m.n_bh_step_deletions}")
            if m.n_bh_session_deletions > 0:
                n_ratio = m.n_bh_creation_count / m.n_bh_session_deletions
                print(f"  │  Session: created {m.n_bh_creation_count:,}  deleted {m.n_bh_session_deletions:,}  C:D {n_ratio:.1f}:1")
            print(f"  │  Phase:   {m.pos_6d[m.n_bh_worker_idx].cpu().numpy().round(3)}")
            n_audio_sim = m.n_gravity_audio.get('chord_size', 0)
            n_video_sim = m.n_gravity_video.get('chord_size', 0)
            n_audio_acc = m.n_gravity_audio.get('top_access', 0.0)
            n_video_acc = m.n_gravity_video.get('top_access', 0.0)
            n_audio_age = m.n_gravity_audio.get('top_age', 0.0)
            n_video_age = m.n_gravity_video.get('top_age', 0.0)
            print(f"  │  Carriers — audio: chord={n_audio_sim} top_access={n_audio_acc:.0f} top_age={n_audio_age:.5f}")
            print(f"  │           — video: chord={n_video_sim} top_access={n_video_acc:.0f} top_age={n_video_age:.5f}")
            print(f"  └─────────────────────────────────────────────────────")
            print()
            # ── Kinetic Manifold ──────────────────────────────────────────────
            km  = m.upe.km_config
            cc  = m.dream_coupling
            ale_names = {1542:'LEFT', 1543:'RIGHT', 1544:'FIRE',
                         1545:'UP',   1546:'DOWN',  1547:'NOOP'}
            print(f"  ┌─ KINETIC MANIFOLD (v127) ───────────────────────────")
            print(f"  │ Phase 1 — ALE Vibration Beacons")
            for idx in range(1542, 1548):
                beacon = m.ale_beacons[idx]
                name   = ale_names[idx]
                omega  = km['ale_vib_omega_base'] + km['ale_omega_offsets'][idx]
                print(f"  │   {name:<6} [{idx}]: phase={beacon.phase_accum:.0f}  ω={omega:.3f} Hz")
            print(f"  │ Phase 2 — Dream Mirror Workers → see DREAM TRIPLETS below")
            print(f"  └─────────────────────────────────────────────────────")
            print(f"  ┌─ DREAM TRIPLETS ─────────────────────────────────────────")
            print(f"  │ Drift N-Reader    [1552]: pos={m.pos_6d[1552].cpu().numpy().round(2)}  vel={torch.norm(m.vel_6d[1552]).item():.6f}")
            print(f"  │ Drift Main-Anchor [1553]: pos={m.pos_6d[1553].cpu().numpy().round(2)}  vel={torch.norm(m.vel_6d[1553]).item():.6f}")
            print(f"  │ Chord N-Reader    [1554]: pos={m.pos_6d[1554].cpu().numpy().round(2)}  vel={torch.norm(m.vel_6d[1554]).item():.6f}")
            print(f"  │ Chord Main-Anchor [1555]: pos={m.pos_6d[1555].cpu().numpy().round(2)}  vel={torch.norm(m.vel_6d[1555]).item():.6f}")
            print(f"  │ Physics N-Reader  [1556]: pos={m.pos_6d[1556].cpu().numpy().round(2)}  vel={torch.norm(m.vel_6d[1556]).item():.6f}")
            print(f"  │ Physics Main-Anchor[1557]: pos={m.pos_6d[1557].cpu().numpy().round(2)}  vel={torch.norm(m.vel_6d[1557]).item():.6f}")
            print(f"  │ Teleport count: {m.chord_coupling.teleport_count}")
            print(f"  └─────────────────────────────────────────────────────────")
            # ── v133: Robot Arm ──────────────────────────────────────────────
            if m.robot_mode > 0:
                rs = m.robot_sender
                print(f"  ┌─ ROBOT ARM (mode {m.robot_mode}) ──────────────────────────")
                if rs:
                    print(f"  │  Target: [{rs.target_pos[0]:.3f}, {rs.target_pos[1]:.3f}, {rs.target_pos[2]:.3f}]")
                    print(f"  │  Wrist:  rot={rs.wrist_rot:.3f}  tilt={rs.wrist_tilt:.3f}")
                    print(f"  │  Gripper: {rs.gripper_state}")
                km = m.upe.km_config
                for idx in range(1558, 1564):
                    name = m.upe.homes[idx]['name'] if idx in m.upe.homes else f"W{idx}"
                    active = "🟢" if idx in m._robot_active_workers else "⚫"
                    val = m.s_filtered[idx].item()
                    if idx in m.robot_beacons:
                        beacon = m.robot_beacons[idx]
                        omega = km['robot_vib_omega_base'] + km['robot_omega_offsets'][idx]
                        print(f"  │  {active} {name:<10} [{idx}]: val={val:>7.1f}  ω={omega:.3f}  phase={beacon.phase_accum:.0f}")
                    else:
                        print(f"  │  {active} {name:<10} [{idx}]: val={val:>7.1f}")
                print(f"  └─────────────────────────────────────────────────────")

        elif command == 'blackhole' or command == 'bh':
            if len(parts) < 2:
                status = "🟢 ENABLED" if self.magi.black_hole_deletion_enabled else "🔴 DISABLED"
                print(f"\n🕳️  MAIN Black Hole Worker: {status}")
                print(f"  Index: {self.magi.black_hole_worker_idx}")
                print(f"  Phase: {self.magi.pos_6d[self.magi.black_hole_worker_idx].cpu().numpy()}")
                print(f"  Value: {self.magi.s_filtered[self.magi.black_hole_worker_idx].item():.1f}")
                print(f"  Memory: {self.magi.memory_bank.size:,} / {self.magi.memory_bank.max_memories:,} ({(self.magi.memory_bank.size/self.magi.memory_bank.max_memories)*100:.1f}%)")
                print(f"  In field: {self.magi.black_hole_memories_in_field} memories")
                print(f"  Deletions (session): {self.magi.black_hole_daily_deletions}")
                print(f"  Creations (session): {self.magi.black_hole_creation_count}")
                bh_vel = torch.norm(self.magi.vel_6d[self.magi.black_hole_worker_idx]).item()
                snap_firing = bh_vel < 0.001
                print(f"  Velocity: {bh_vel:.6f} | Snap: {'🏠 FIRING' if snap_firing else '🌀 ROAMING'}")
                if self.magi.black_hole_daily_deletions > 0:
                    ratio = self.magi.black_hole_creation_count / self.magi.black_hole_daily_deletions
                    print(f"  Creation:Deletion Ratio: {ratio:.1f}:1")

                # N BH
                m = self.magi
                nbh_result  = getattr(m, 'n_bh_last_result', {})
                nbh_tension = nbh_result.get('tension_factor', abs(nbh_result.get('output', 0.0)) / 1500.0)
                nbh_r       = m.n_bh_base_radius * (1.0 + nbh_tension)
                nbh_active  = '🟢 ACTIVE' if nbh_result.get('is_active', False) else '⚫ quiet'
                n_cap_pct   = (m.n_bank.size / m.n_bank.max_memories) * 100.0
                nbh_vel     = torch.norm(m.vel_6d[m.n_bh_worker_idx]).item()
                nbh_snap    = '🏠 FIRING' if nbh_vel < 0.001 else '🌀 ROAMING'
                print(f"\n🌀 N Black Hole Worker: {nbh_active}")
                print(f"  Index: {m.n_bh_worker_idx}")
                print(f"  Phase: {m.pos_6d[m.n_bh_worker_idx].cpu().numpy()}")
                print(f"  Value: {m.s_filtered[m.n_bh_worker_idx].item():.1f}")
                print(f"  N Bank: {m.n_bank.size:,} / {m.n_bank.max_memories:,} ({n_cap_pct:.1f}%)")
                print(f"  In field: {m.n_bh_memories_in_field} memories")
                print(f"  Deletions (step): {m.n_bh_step_deletions} | tension: {nbh_tension:.4f} | r: {nbh_r:.4f}")
                print(f"  Deletions (session): {m.n_bh_session_deletions:,}  Creations: {m.n_bh_creation_count:,}")
                # Origin split — how many audio vs video memories in N bank
                if m.n_bank.size > 0:
                    n_audio_ct = int((m.n_bank.metadata_origin[:m.n_bank.size] < 0.5).sum().item())
                    n_video_ct = m.n_bank.size - n_audio_ct
                    print(f"  Origin split: audio={n_audio_ct:,}  video={n_video_ct:,}")
                print(f"  Velocity: {nbh_vel:.6f} | Snap: {nbh_snap}")
                n_audio_chord = m.n_gravity_audio.get('chord_size', 0)
                n_video_chord = m.n_gravity_video.get('chord_size', 0)
                n_audio_acc   = m.n_gravity_audio.get('top_access', 0.0)
                n_video_acc   = m.n_gravity_video.get('top_access', 0.0)
                n_audio_age   = m.n_gravity_audio.get('top_age', 0.0)
                n_video_age   = m.n_gravity_video.get('top_age', 0.0)
                print(f"  Carriers — audio: chord={n_audio_chord} top_access={n_audio_acc:.0f} top_age={n_audio_age:.5f}")
                print(f"           — video: chord={n_video_chord} top_access={n_video_acc:.0f} top_age={n_video_age:.5f}")
                return
            
            subcmd = parts[1]
            args = parts[2:]
            
            if subcmd == 'enable':
                self.magi.black_hole_deletion_enabled = True
                print("🕳️ Black Hole Worker: ENABLED")
            elif subcmd == 'disable':
                self.magi.black_hole_deletion_enabled = False
                print("⏸️ Black Hole Worker: DISABLED")
            elif subcmd == 'reset':
                self.magi.black_hole_daily_deletions = 0
                self.magi.black_hole_creation_count = 0
                self.magi.reset_black_hole_window()
                print("♻️ Black Hole counters reset")
            elif subcmd == 'stats':
                metrics = self.magi.get_black_hole_metrics()
                if metrics:
                    print(f"\n🕳️  MAIN Black Hole — worker {self.magi.black_hole_worker_idx}  (last {metrics['window_duration']:.1f}s)")
                    print(f"  Position:")
                    print(f"    Phase (HB): {self.magi.pos_6d[self.magi.black_hole_worker_idx].cpu().numpy()}")
                    print(f"    Frequency: {metrics['worker_freq']:.2f} Hz")
                    print(f"    Delay: {metrics['worker_delay']:.2f} ms")
                    print(f"  Dynamics:")
                    print(f"    Worker value: {metrics['worker_value']:.2f}")
                    print(f"    Effective radius: {metrics['effective_radius']:.4f}")
                    print(f"    Memories in field: {metrics['memories_in_field']}")
                    print(f"  Activity:")
                    print(f"    Creation rate: {metrics['creation_rate']:.2f}/sec")
                    print(f"    Deletion rate: {metrics['deletion_rate']:.2f}/sec")
                    print(f"    Ratio: {metrics['creation_deletion_ratio']:.1f}:1")
                    print(f"  Totals:")
                    print(f"    Created: {metrics['total_creations']:,}")
                    print(f"    Deleted: {metrics['total_deletions']:,}")
                    print(f"    Current: {self.magi.memory_bank.size:,} ({metrics['capacity_pct']:.1f}%)")
                else:
                    print("⏳ Insufficient data for main BH statistics (need >1 second)")

                # N BH stats
                m = self.magi
                nbh_result  = getattr(m, 'n_bh_last_result', {})
                nbh_tension = nbh_result.get('tension_factor', abs(nbh_result.get('output', 0.0)) / 1500.0)
                nbh_r       = m.n_bh_base_radius * (1.0 + nbh_tension)
                nbh_active  = '🟢 ACTIVE' if nbh_result.get('is_active', False) else '⚫ quiet'
                n_cap_pct   = (m.n_bank.size / m.n_bank.max_memories) * 100.0
                log_t       = m.get_log_time()

                print(f"\n🌀 N Black Hole — worker {m.n_bh_worker_idx}  ({nbh_active})")
                print(f"  Position:")
                print(f"    Phase (HB): {m.pos_6d[m.n_bh_worker_idx].cpu().numpy()}")
                print(f"    Frequency: {m.freq[m.n_bh_worker_idx].item():.2f} Hz")
                print(f"    Delay: {m.delay[m.n_bh_worker_idx].item():.2f} ms")
                print(f"  Dynamics:")
                print(f"    Tension: {nbh_tension:.4f}  |  Effective radius: {nbh_r:.4f}")
                print(f"    Memories in field: {m.n_bh_memories_in_field}")
                print(f"    Deletions (this step): {m.n_bh_step_deletions}")
                if m.n_bh_session_deletions > 0:
                    n_ratio = m.n_bh_creation_count / m.n_bh_session_deletions
                    print(f"  Activity:")
                    print(f"    Creation:Deletion Ratio: {n_ratio:.1f}:1")
                print(f"  Totals:")
                print(f"    Created: {m.n_bh_creation_count:,}")
                print(f"    Deleted: {m.n_bh_session_deletions:,}")
                print(f"    Current: {m.n_bank.size:,} ({n_cap_pct:.1f}%)")
                print(f"  N Bank:")
                print(f"    Size: {m.n_bank.size:,} / {m.n_bank.max_memories:,}  ({n_cap_pct:.1f}%)")
                print(f"    elapsed: {log_t:.1f} days  (since Jan 1 2025)")
                n_audio_chord  = m.n_gravity_audio.get('chord_size', 0)
                n_audio_acc    = m.n_gravity_audio.get('top_access', 0.0)
                n_audio_age    = m.n_gravity_audio.get('top_age', 0.0)
                n_audio_energy = m.n_gravity_audio.get('chord_energy', 0.0)
                n_video_chord  = m.n_gravity_video.get('chord_size', 0)
                n_video_acc    = m.n_gravity_video.get('top_access', 0.0)
                n_video_age    = m.n_gravity_video.get('top_age', 0.0)
                n_video_energy = m.n_gravity_video.get('chord_energy', 0.0)
                print(f"  Carrier N-retrieval:")
                print(f"    Audio carrier:  chord={n_audio_chord}  top_access={n_audio_acc:.0f}  top_age={n_audio_age:.5f}  energy={n_audio_energy:.0f}")
                print(f"    Video carrier:  chord={n_video_chord}  top_access={n_video_acc:.0f}  top_age={n_video_age:.5f}  energy={n_video_energy:.0f}")
            elif subcmd == 'window':
                self.magi.reset_black_hole_window()
                print("🔄 Measurement window reset")
        elif command == 'upe' or command == 'homes':
            stats = self.magi.upe.get_home_stats()
            if stats:
                # Get constants from first worker
                first_data = next(iter(stats.values()))
                threshold = first_data['threshold']
                window_hours = first_data['window_hours']
                
                print(f"\n⚡️ UPE PHYSICS - DUAL BARS + MOMENTUM")
                print(f"Snap Threshold: {threshold:.6f} | Window: {window_hours:.0f}h")
                print("=" * 70)
                
                for idx, data in stats.items():
                    # Dual bars
                    snap_bar_len = min(10, int(data['snap_pct'] / 10))
                    intent_bar_len = min(10, int(data['intent_pct'] / 10))
                    
                    snap_bar = "█" * snap_bar_len + "░" * (10 - snap_bar_len)
                    intent_bar = "█" * intent_bar_len + "░" * (10 - intent_bar_len)
                    
                    # Status (based on TRUTH - Snap bar)
                    if data['snap_ready']:
                        status = "⚡️ SNAP READY"
                        color = "🔵"
                        snap_indicator = "🔥"
                    elif data['snap_pct'] >= 99.0:
                        status = "⚠️  ON EDGE"
                        color = "🟡"
                        snap_indicator = "⚠️"
                    elif data['snap_pct'] >= 50.0:
                        status = "🟢 CHARGING"
                        color = "🟢"
                        snap_indicator = "⚡"
                    else:
                        status = "🔴 BUILDING"
                        color = "🔴"
                        snap_indicator = "🔄"
                    
                    print(f"\n  {color} {data['name']}: {data['momentum_trend']} {snap_indicator}")
                    
                    # === PRIMARY: SNAP BAR (The Truth) ===
                    print(f"    Snap (Truth):  [{snap_bar}] {data['snap_pct']:.1f}%")
                    print(f"      {data['snap_pressure']:.6f} / {threshold:.6f}")
                    
                    # === SECONDARY: INTENT BAR (The Trend) ===
                    print(f"    Intent ({window_hours:.0f}h): [{intent_bar}] {data['intent_pct']:.1f}%")
                    
                    print(f"    Status: {status}")
                    
                    # Momentum detail
                    momentum = data['momentum']
                    if abs(momentum) > 0.00001:
                        momentum_str = f"+{momentum:.6f}" if momentum > 0 else f"{momentum:.6f}"
                        print(f"    Momentum: {momentum_str}/step")
                    
                    # Reliability indicator
                    if data['rate_reliable']:
                        print(f"    Rate: ✓ Reliable")
                    elif data['total_samples'] > 0:
                        print(f"    Rate: ? Need samples (R{data['recent_samples']}/O{data['older_samples']})")
                    
                    # Time context
                    drift_time = data['time_since_drift']
                    if drift_time > 300:
                        if drift_time > 3600:
                            print(f"    Stable: {drift_time/3600:.1f}h")
                        else:
                            print(f"    Stable: {drift_time/60:.1f}min")
                
                print("\n" + "=" * 70)
                print("SNAP (Truth): All-time decayed pressure. 100% = home will snap.")
                print(f"INTENT ({window_hours:.0f}h): Recent pressure trend.")
                print("Momentum: ↗=Building, ↘=Decaying, →=Stable")
                print("🔥=Snap ready, ⚡=Charging, 🔄=Building")
        elif command == 'scalestats' or command == 'ss':
            """Show what ranges MaGi has discovered"""
            if len(parts) < 2:
                # Summary of all scalers
                print("\n📊 SCALESTATS - Discovered Ranges")
                print("=" * 60)
                for name, scaler in self.magi.scalers.items():
                    info = scaler.get_info()
                    status = "🟢" if info['stats']['active_frames'] > 0 else "⚫"
                    print(f"  {status} {name:10s} | "
                          f"Range: [{info['range']['min']:5.1f}→{info['range']['max']:6.1f}] | "
                          f"Signal: {info['signal_tracking']['current_strength']:.2f} | "
                          f"Exp: {info['stats']['expansions']:3d}")
            else:
                # Show detailed stats for specific scaler
                name = parts[1]
                if name in self.magi.scalers:
                    scaler = self.magi.scalers[name]
                    info = scaler.get_info()
                    
                    print(f"\n📊 SCALESTATS - {name.upper()}")
                    print("=" * 60)
                    print(f"Mode: {info['mode']} | Spacing: {info['spacing']}")
                    print(f"\n📈 Range:")
                    print(f"  Min Threshold: {info['range']['min']:.2f}")
                    print(f"  Current Max: {info['range']['max']:.2f}")
                    print(f"  Absolute Max: {info['range']['absolute_max']:.2f}")
                    print(f"\n📡 Signal Tracking:")
                    print(f"  Strength: {info['signal_tracking']['current_strength']:.3f}")
                    print(f"  Window: {info['signal_tracking']['window']}s")
                    print(f"  Decay Rate: {info['signal_tracking']['decay_rate']}")
                    print(f"\n📊 Stats:")
                    print(f"  Expansions: {info['stats']['expansions']}")
                    print(f"  Decays: {info['stats']['decays']}")
                    print(f"  Active Frames: {info['stats']['active_frames']}")
                    
                    if info['mode'] == 'steps':
                        print(f"\n🔢 Steps: {info['steps']['n_steps']}")
                        print(f"  Thresholds: {[f'{t:.1f}' for t in info['steps']['thresholds'][:5]]}...")
                    else:
                        print(f"\n📐 Output Range: [{info['range_mapping']['output'][0]} → {info['range_mapping']['output'][1]}]")
                else:
                    print(f"❌ Unknown scaler: {name}")
                    print(f"   Available: {list(self.magi.scalers.keys())}")

        elif command == 'bridge':
            if self.magi.bridge_controller.handle_command(parts):
                return

class UnifiedVideoSource:
    def __init__(self):
        self.mode = 'webcam'
        self.webcam_cap = None
        self.ale = None
        self.sct = None
        self.monitor = None
        # Screen Grabber State
        self.tk_root = None
        self.screen_grabber = None
        # Viewer State
        self.viewer_images = []
        self.viewer_index = 0
        self.switch_to_webcam()
    
    def switch_to_webcam(self):
        self._cleanup()
        self.mode = 'webcam'
        self.webcam_cap = cv2.VideoCapture(0)
        self.webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("📷 Input: Webcam")
    
    def switch_to_ale(self, rom_path):
        self._cleanup()
        self.mode = 'ale'
        try:
            from ale_py import ALEInterface
            self.ale = ALEInterface()
            self.ale.loadROM(rom_path)
            print(f"🎮 Input: ALE ({rom_path})")
            
            # Log legal actions for context (use ALE_ACTION_MAP for clean names)
            legal_actions = self.ale.getLegalActionSet()
            # Convert Action objects to integers and sort
            legal_action_ints = sorted([int(a) for a in legal_actions])
            legal_names = [ALE_ACTION_MAP.get(a, f"ACTION_{a}") for a in legal_action_ints]
            print(f"ℹ️ Legal Actions: {legal_names}")
            
            # KEEP EXISTING WORKER MAPPING
            print(f"🎯 Using workers 1542-1547: LEFT, RIGHT, FIRE, UP, DOWN, NOOP")
            
            self.ale.reset_game()
        except Exception as e:
            print(f"❌ ALE Error: {e}")
            self.switch_to_webcam()
    
    def switch_to_screencap(self, region):
        self._cleanup()
        self.mode = 'screencap'
        try:
            import mss
            self.sct = mss.mss()
            x, y, w, h = map(int, region.split(','))
            self.monitor = {"top": y, "left": x, "width": w, "height": h}
            print(f"🖥️  Input: ScreenCap {region}")
        except Exception as e:
            print(f"❌ ScreenCap Error: {e}")
            self.switch_to_webcam()

    def switch_to_screen_grab(self):
        self._cleanup()
        self.mode = 'screen'
        try:
            self.tk_root = tk.Tk()
            self.screen_grabber = ScreenGrabber(self.tk_root, width=640, height=480)
            print("🖥️  Input: Screen Grabber (Move window to capture)")
        except Exception as e:
            print(f"❌ Screen Grabber Error: {e}")
            self.switch_to_webcam()

    def switch_to_viewer(self, folder_path):
        self._cleanup()
        self.mode = 'viewer'
        self.viewer_images = []
        self.viewer_index = 0
        
        if not os.path.exists(folder_path):
            print(f"❌ Viewer Error: Folder '{folder_path}' not found.")
            return

        extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
        try:
            self.viewer_images = sorted([
                os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                if f.lower().endswith(extensions)
            ])
            print(f"🖼️ Input: Viewer ({len(self.viewer_images)} images)")
        except Exception as e:
            print(f"❌ Viewer Error: {e}")

    def viewer_nav(self, direction):
        if not self.viewer_images: return
        if direction == 'NEXT':
            self.viewer_index = (self.viewer_index + 1) % len(self.viewer_images)
        elif direction == 'PREV':
            self.viewer_index = (self.viewer_index - 1) % len(self.viewer_images)

    def get_frame(self):
        frame = None
        if self.mode == 'webcam' and self.webcam_cap:
            ret, frame = self.webcam_cap.read()
        elif self.mode == 'ale' and self.ale:
            s = self.ale.getScreenRGB()
            frame = cv2.resize(s, (640, 480)) # Resize for consistent processing
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif self.mode == 'screencap' and self.sct:
            s = self.sct.grab(self.monitor)
            frame = cv2.cvtColor(np.array(s), cv2.COLOR_BGRA2BGR)
        elif self.mode == 'screen' and self.screen_grabber:
            # Capture frame from Tkinter window position
            frame = self.screen_grabber.capture_frame()
            # Update Tkinter GUI (keep window responsive without mainloop)
            if self.tk_root:
                try:
                    self.tk_root.update_idletasks()
                    self.tk_root.update()
                except tk.TclError:
                    pass # Window closed
        elif self.mode == 'remote':
            with self._remote_frame_lock:
                frame = self._remote_frame.copy() if self._remote_frame is not None else None
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
        elif self.mode == 'viewer':
            if self.viewer_images:
                try:
                    img_path = self.viewer_images[self.viewer_index]
                    img = cv2.imread(img_path)
                    if img is not None:
                        frame = cv2.resize(img, (640, 480))
                except Exception:
                    pass
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8) # Black frame fallback
                
        return frame

    def execute_action(self, action_idx):
        """Execute ALE action directly by index (0-17)"""
        if self.mode == 'ale' and self.ale:
            # Direct ALE action - no translation needed
            if 0 <= action_idx < 18:
                self.ale.act(action_idx)
            else:
                print(f"⚠️ Invalid ALE action index: {action_idx}")

    def switch_to_remote(self, pi_ip):
        """
        Connect to pi_server.py TCP video stream (port 5002).
        Frames arrive as 4-byte big-endian length + JPEG bytes.
        Runs a background thread; get_frame() returns latest decoded frame.
        Falls back to black frame if connection drops.
        """
        self._cleanup()
        self.mode        = 'remote'
        self._remote_ip  = pi_ip
        self._remote_frame     = None
        self._remote_frame_lock = threading.Lock()
        self._remote_running    = True

        def _recv_loop():
            VIDEO_PORT = 5002
            buf = b''
            while self._remote_running:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5.0)
                    sock.connect((pi_ip, VIDEO_PORT))
                    sock.settimeout(3.0)
                    print(f"🎥 Remote video connected: {pi_ip}:{VIDEO_PORT}")
                    buf = b''
                    while self._remote_running:
                        while len(buf) < 4:
                            chunk = sock.recv(4096)
                            if not chunk: raise ConnectionResetError("stream closed")
                            buf += chunk
                        length = struct.unpack('>I', buf[:4])[0]
                        buf = buf[4:]
                        while len(buf) < length:
                            chunk = sock.recv(65536)
                            if not chunk: raise ConnectionResetError("stream closed")
                            buf += chunk
                        jpg = buf[:length]
                        buf = buf[length:]
                        arr   = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            frame = cv2.resize(frame, (640, 480))
                            with self._remote_frame_lock:
                                self._remote_frame = frame
                except Exception as e:
                    print(f"⚠️  Remote video: {e} — retrying in 2s")
                    time.sleep(2.0)
                finally:
                    try: sock.close()
                    except: pass

        self._remote_thread = threading.Thread(target=_recv_loop, daemon=True)
        self._remote_thread.start()
        print(f"📡 Remote mode: {pi_ip}  (video TCP:5002  commands UDP:5003)")

    def _cleanup(self):
        if self.webcam_cap: self.webcam_cap.release()
        # Stop remote video thread if running
        if hasattr(self, '_remote_running'):
            self._remote_running = False
        if self.tk_root:
            try:
                self.tk_root.destroy()
            except:
                pass
            self.tk_root = None
        self.ale = None
        self.sct = None
        self.screen_grabber = None

# ==========================================
# 🧠 PERSISTENT VECTOR MEMORY
# ==========================================
class EnhancedHypersphereMemory:
    def __init__(self, dim=130, max_memories=3000000, device='cuda'): 
        self.dim = dim
        self.device = device
        self.max_memories = max_memories
        self.memories = torch.zeros((max_memories, dim), device=device, dtype=torch.float32)
        self.meta_freq = torch.zeros(max_memories, device=device, dtype=torch.float32)
        self.meta_delay = torch.zeros(max_memories, device=device, dtype=torch.float32)
        self.timestamps = torch.zeros(max_memories, device=device, dtype=torch.float64)
        self.access_counts = torch.zeros(max_memories, device=device, dtype=torch.float32)
        self.size = 0
        self._proj_matrix = None
        self.memory_influence_strength = 0.05
        # v130: 6D coordinate storage [child, youth, adult, elder, freq_phase, delay_phase]
        self.mem_coords_6d = torch.zeros((max_memories, 6), device=device, dtype=torch.float32)
    
    def _init_projection(self, input_dim):
        g = torch.Generator(device=self.device); g.manual_seed(42)
        self._proj_matrix = torch.randn(self.dim, input_dim, device=self.device, dtype=torch.float32, generator=g) / math.sqrt(input_dim)

    def save(self, filename):
        data = {
            'memories': self.memories[:self.size].clone().cpu(),
            'meta_freq': self.meta_freq[:self.size].clone().cpu(),
            'meta_delay': self.meta_delay[:self.size].clone().cpu(),
            'timestamps': self.timestamps[:self.size].clone().cpu(),
            'access_counts': self.access_counts[:self.size].clone().cpu(),
            'mem_coords_6d': self.mem_coords_6d[:self.size].clone().cpu(),
            'size': self.size,
            'influence': self.memory_influence_strength,
            'saved_max_memories': self.max_memories,
            'dim': self.dim,
            'mapping_version': 1,  # v131: log-mapped coords
        }
        torch.save(data, filename)
        print(f"💾 Saved {self.size} memories.")

    def load(self, filename):
        # v131: Try new filename, fall back to v130/v129 for migration
        actual_file = filename
        if not os.path.exists(filename) and os.path.exists(V129_MEMORY_FILE):
            print(f"🔄 v131 migration: loading from {V129_MEMORY_FILE} → will save as {filename}")
            actual_file = V129_MEMORY_FILE
        if not os.path.exists(actual_file): return
        try:
            data = torch.load(actual_file, map_location=self.device)
            sz = min(data['size'], self.max_memories)
            self.memories[:sz] = data['memories'][:sz]
            self.meta_freq[:sz] = data['meta_freq'][:sz]
            self.meta_delay[:sz] = data['meta_delay'][:sz]
            self.timestamps[:sz] = data['timestamps'][:sz]
            self.access_counts[:sz] = data['access_counts'][:sz]
            self.size = sz
            self.memory_influence_strength = data.get('influence', 0.05)
            
            mapping_ver = data.get('mapping_version', 0)
            
            if 'mem_coords_6d' not in data:
                # v129: No 6D coords at all — build from Hz/ms metadata
                print(f"🔄 Converting {sz} v129 memories to v131 log coords...")
                self.mem_coords_6d[:sz, :4] = math.pi  # Neutral lens phases
                self.mem_coords_6d[:sz, 4] = freq_to_log_coord_t(data['meta_freq'][:sz].to(self.device))
                self.mem_coords_6d[:sz, 5] = delay_to_log_coord_t(data['meta_delay'][:sz].to(self.device))
                self.save(filename)
                print(f"✅ Converted {sz} memories → {filename}")
            elif mapping_ver < 1:
                # v130: Has 6D coords but with LINEAR phases → convert to unwrapped log
                print(f"🔄 Converting {sz} v130 memories to v131 log coords...")
                self.mem_coords_6d[:sz] = data['mem_coords_6d'][:sz].to(self.device)
                # Overwrite freq/delay dims with unwrapped log coords from absolute Hz/ms
                self.mem_coords_6d[:sz, 4] = freq_to_log_coord_t(data['meta_freq'][:sz].to(self.device))
                self.mem_coords_6d[:sz, 5] = delay_to_log_coord_t(data['meta_delay'][:sz].to(self.device))
                self.save(filename)
                print(f"✅ Converted {sz} memories → {filename}")
            else:
                # v131: Already log-mapped
                self.mem_coords_6d[:sz] = data['mem_coords_6d'][:sz].to(self.device)
                if actual_file != filename:
                    self.save(filename)
                    print(f"💾 Migrated {sz} memories → {filename}")
            print(f"✅ Loaded {self.size} memories.")
        except Exception as e:
            print(f"❌ Load failed: {e}")

    def encode(self, state_data, all_workers=False):
        if all_workers:
            phases_hb = state_data['phases_hb']
            phases_s = state_data['phases_s']
            global_coh = state_data['global_coh'].unsqueeze(1)
            cross_tension = state_data['cross_tension'].unsqueeze(1)
            # v131: Log-space normalization for embedding
            f_val = torch.log(state_data['freq'].clamp(min=MIN_FREQ) / MIN_FREQ) / LOG_FREQ_STEP / TWO_PI
            d_val = torch.log(state_data['delay'].clamp(min=MIN_DELAY) / MIN_DELAY) / LOG_DELAY_STEP / TWO_PI
            freq_norm = f_val.unsqueeze(1)
            delay_norm = d_val.unsqueeze(1)
        else:
            def ensure_2d(val):
                val = torch.as_tensor(val, device=self.device, dtype=torch.float32)
                return torch.atleast_2d(val)
            phases_hb     = state_data['phases_hb'].unsqueeze(0)
            phases_s      = state_data['phases_s'].unsqueeze(0)
            global_coh    = ensure_2d(state_data['global_coh'])
            cross_tension = ensure_2d(state_data['cross_tension'])
            # v131: Log-space normalization
            freq_norm     = ensure_2d(torch.log(torch.as_tensor(state_data['freq'], device=self.device, dtype=torch.float32).clamp(min=MIN_FREQ) / MIN_FREQ) / LOG_FREQ_STEP / TWO_PI)
            delay_norm    = ensure_2d(torch.log(torch.as_tensor(state_data['delay'], device=self.device, dtype=torch.float32).clamp(min=MIN_DELAY) / MIN_DELAY) / LOG_DELAY_STEP / TWO_PI)

        features = [
            torch.sin(phases_hb).flatten(start_dim=1),
            torch.cos(phases_hb).flatten(start_dim=1),
            torch.sin(phases_s).flatten(start_dim=1),
            torch.cos(phases_s).flatten(start_dim=1),
            global_coh, cross_tension, freq_norm, delay_norm
        ]
        state_vectors = torch.cat(features, dim=1)
        if self._proj_matrix is None: self._init_projection(state_vectors.shape[-1])
        embeddings = torch.matmul(state_vectors, self._proj_matrix.t())
        return F.normalize(embeddings, p=2, dim=-1)

    def is_novel(self, new_embedding, threshold=0.90, check_recent=50):
        if self.size == 0: return True
        start_idx = max(0, self.size - check_recent)
        recent_mems = self.memories[start_idx:self.size]
        similarities = torch.matmul(new_embedding, recent_mems.t())[0]
        return similarities.max().item() < threshold

    def store(self, embedding, metadata):
        target_idx = self.size
        if self.size >= self.max_memories:
            ages = time.time() - self.timestamps
            scores = self.access_counts / (ages + 1.0)
            target_idx = torch.argmin(scores).item()
        else:
            self.size += 1
        self.memories[target_idx] = embedding.squeeze(0)
        self.meta_freq[target_idx] = metadata['freq']
        self.meta_delay[target_idx] = metadata['delay']
        self.timestamps[target_idx] = time.time()
        self.access_counts[target_idx] = 1
        # v131: Store UNWRAPPED log coordinates in mem_coords_6d (octave-aware matching)
        freq_log_coord = freq_to_log_coord(metadata['freq'])
        delay_log_coord = delay_to_log_coord(metadata['delay'])
        
        # Lens dims: use current worker phase if available in metadata, else neutral π
        if 'coords_6d' in metadata:
            self.mem_coords_6d[target_idx] = metadata['coords_6d']
        else:
            self.mem_coords_6d[target_idx, :4] = math.pi  # Neutral
            self.mem_coords_6d[target_idx, 4] = freq_log_coord
            self.mem_coords_6d[target_idx, 5] = delay_log_coord
        
    def retrieve_gravity(self, query_embedding, similarity_threshold=0.85):
        if self.size == 0:
            self.memory_influence_strength *= 0.99995
            return None

        # Stage 1: randint sample — O(sample_size) not O(N)
        if self.size > 20000:
            sample_idx = torch.randint(0, self.size, (MEMORY_SAMPLE_SIZE,), device=self.device)
            active_memories = self.memories[sample_idx]
        else:
            sample_idx = torch.arange(self.size, device=self.device)
            active_memories = self.memories[:self.size]

        similarities = torch.matmul(query_embedding, active_memories.t())[0]

        # Stage 2: topk(20) → threshold — preserves sharp attractor identity
        k = min(20, similarities.shape[0])
        top_sims, top_local_idx = torch.topk(similarities, k)
        mask = top_sims > similarity_threshold

        if not mask.any() or torch.isnan(top_sims).any():
            self.memory_influence_strength *= 0.99995
            return None

        local_sims    = top_sims[mask]
        local_indices = sample_idx[top_local_idx[mask]]

        # Softmax with temperature — stable, sharpened attractor pull
        weights = torch.softmax(local_sims * SOFTMAX_TEMP, dim=0)

        # Fully vectorized — no Python loop, no .item() per hit
        target_freq      = (self.meta_freq[local_indices]  * weights).sum()
        target_delay     = (self.meta_delay[local_indices] * weights).sum()
        attractor_vector = (self.memories[local_indices].t() @ weights.unsqueeze(1)).squeeze()

        # Integer increment — preserves BH deletion ordering
        self.access_counts[local_indices] += 1

        max_sim    = local_sims.max().item()
        avg_access = self.access_counts[local_indices].float().mean().item()

        # Adaptive strength — restored
        target_cap = 0.50 + (max_sim * 0.50)
        self.memory_influence_strength = min(
            self.memory_influence_strength + 0.00001, target_cap
        )

        return {
            'freq':               target_freq,
            'delay':              target_delay,
            'strength':           self.memory_influence_strength,
            'center_embedding':   F.normalize(attractor_vector, p=2, dim=-1),
            'sensory_modulation': max_sim * 0.8,
            'phase_amplitude':    (avg_access / 50.0) * max_sim,
            'phase_velocity':     1.0 + max_sim,
            'similarity':         max_sim,
            'avg_access':         avg_access
        }


# ==========================================
# 🌀 N HYPERSPHERE — TEMPORAL NARRATIVE MEMORY
# ==========================================
class NHypersphereMemory:
    """
    N Hypersphere — Temporal Narrative Layer.

    7D coordinate space:
      [phase_0, phase_1, phase_2, phase_3, log_time_norm, freq_norm*0.7, delay_norm*0.7]
    Dims 0-3: raw oscillator phase / (2π)  — primary geometry
    Dim  4:   log_time_norm                — temporal age
    Dims 5-6: freq/delay normalized [0,1] × 0.7 — soft spectral neighbourhood filter
              Downweighted so phase+time remain primary drivers of cosine similarity.

    Stores what KEEPS HAPPENING across time, not just what is happening now.
    Phase patterns recurring at multiple log_times form natural temporal attractors
    in 5D — no explicit sequence model needed. The geometry IS the narrative.

    N only stores when the main bank stores (piggybacking), so N is always
    coupled to meaningful moments, never accumulating noise on its own clock.
    """
    N_MEMORY_FILE = "magi_torus_n_memory.pt"   # v130: 6D toroidal N bank
    V129_N_MEMORY_FILE = "n_v111_memory.pt"   # v100-v129 — auto-converted on first load

    def __init__(self, max_memories=3000000, device='cuda'):
        self.device = device
        self.max_memories = max_memories
        self.size = 0

        # 7D coordinates stored directly — no projection matrix
        self.coords          = torch.zeros((max_memories, 7), device=device, dtype=torch.float32)
        # Full lens vector at storage time
        self.metadata_child  = torch.zeros(max_memories, device=device, dtype=torch.float32)
        self.metadata_youth  = torch.zeros(max_memories, device=device, dtype=torch.float32)
        self.metadata_adult  = torch.zeros(max_memories, device=device, dtype=torch.float32)
        self.metadata_elder  = torch.zeros(max_memories, device=device, dtype=torch.float32)
        self.metadata_freq   = torch.zeros(max_memories, device=device, dtype=torch.float32)
        self.metadata_delay  = torch.zeros(max_memories, device=device, dtype=torch.float32)
        self.metadata_origin   = torch.zeros(max_memories, device=device, dtype=torch.float32)  # 0=audio 1=video
        # Phase 1 continental diagnostics
        self.metadata_tension_ce = torch.zeros(max_memories, device=device, dtype=torch.float32)  # Child-Elder delta
        self.metadata_tension_ae = torch.zeros(max_memories, device=device, dtype=torch.float32)  # Adult-Elder delta
        self.metadata_version    = torch.zeros(max_memories, device=device, dtype=torch.float32)  # coord system version
        self.timestamps      = torch.zeros(max_memories, device=device, dtype=torch.float64)
        self.access_counts   = torch.ones(max_memories,  device=device, dtype=torch.float32)
        self.retrieve_last_iv = 0   # index of last N bank match — for telemetry
        # Temporal clock — persisted across restarts so log_time is continuous
        self._system_epoch      = MAGI_EPOCH
        self._max_log_time_seen = 0.001

    def store(self, coords_7d, metadata):
        """Store a 7D coordinate with full lens metadata.
        v130: coords_7d: [phase_0..3, log_time_norm, freq_phase, delay_phase] — full [0,2π] phases
        metadata must include: child, youth, adult, elder, freq, delay
        """
        if self.size < self.max_memories:
            target_idx = self.size
            self.size += 1
        else:
            target_idx = torch.argmin(self.access_counts[:self.size]).item()

        # v130: Store 7D coords with dims 5,6 as full phases [0, 2π] (not normalized*0.7)
        # coords_7d expected: [phase0, phase1, phase2, phase3, log_time, freq_phase, delay_phase]
        if coords_7d.shape[0] == 7:
            # Ensure freq/delay are full phases
            self.coords[target_idx] = coords_7d.detach().to(self.device)
        else:
            # Legacy 5D input - pad with neutral phases
            pad = torch.tensor([0.5, math.pi, math.pi], device=self.device)  # log_time=0.5, freq=π, delay=π
            self.coords[target_idx] = torch.cat([coords_7d[:4], pad])
        self.metadata_child[target_idx] = metadata['child']
        self.metadata_youth[target_idx] = metadata['youth']
        self.metadata_adult[target_idx] = metadata['adult']
        self.metadata_elder[target_idx] = metadata['elder']
        self.metadata_freq[target_idx]  = metadata['freq']
        self.metadata_delay[target_idx] = metadata['delay']
        self.metadata_origin[target_idx]   = 1.0 if metadata.get('origin') == 'video' else 0.0
        self.metadata_tension_ce[target_idx] = metadata.get('tension_ce', 0.0)
        self.metadata_tension_ae[target_idx] = metadata.get('tension_ae', 0.0)
        self.metadata_version[target_idx]    = metadata.get('version',    121.0)
        self.timestamps[target_idx]     = time.time()
        self.access_counts[target_idx]  = 1.0

    def retrieve(self, query_7d, similarity_threshold=0.80, k=3):
        """
        Cosine similarity retrieval in 7D space.
        Returns weighted adult/elder lens values and access metrics, or None.
        """
        if self.size == 0:
            return None

        q  = F.normalize(query_7d.float().unsqueeze(0), p=2, dim=1)
        db = F.normalize(self.coords[:self.size],        p=2, dim=1)
        sims = torch.matmul(q, db.t())[0]

        mask = sims > similarity_threshold
        if not mask.any():
            return None

        top_vals, rel_idx = torch.topk(sims[mask], min(k, mask.sum().item()))
        real_idx = torch.where(mask)[0][rel_idx]

        tw = child_s = youth_s = adult_s = elder_s = freq_s = access_s = 0.0

        for i, idx in enumerate(real_idx):
            w  = top_vals[i].item()
            iv = idx.item()
            self.access_counts[iv] += 1
            child_s  += self.metadata_child[iv].item()  * w
            youth_s  += self.metadata_youth[iv].item()  * w
            adult_s  += self.metadata_adult[iv].item()  * w
            elder_s  += self.metadata_elder[iv].item()  * w
            freq_s   += self.metadata_freq[iv].item()   * w
            access_s += self.access_counts[iv].item()   * w
            tw       += w

        if tw <= 0:
            return None

        return {
            'child':      child_s  / tw,
            'youth':      youth_s  / tw,
            'adult':      adult_s  / tw,
            'elder':      elder_s  / tw,
            'freq':       freq_s   / tw,
            'avg_access': access_s / tw,
            'similarity': top_vals[0].item(),
        }

    def retrieve_chord(self, query_7d):
        """
        Exact-match chord lookup against N bank coords.
        No cosine scan — direct comparison, all 7 dims must match.
        A match means 'I have been in this exact dynamical state before.'
        """
        if self.size == 0:
            return []

        q_rounded  = query_7d.float().round(decimals=2)
        db_rounded = self.coords[:self.size].round(decimals=2)

        matches = (db_rounded == q_rounded).all(dim=1)
        if not matches.any():
            return []

        iv = matches.nonzero()[0].item()
        self.access_counts[iv] += 1
        self.retrieve_last_iv = iv   # expose for telemetry
        return [{
            'access_count':  self.access_counts[iv].item(),
            'log_time_norm': self.coords[iv, 4].item(),
        }]
    
    def save(self, filename=None):
        filename = filename or self.N_MEMORY_FILE
        data = {
            'coords':        self.coords[:self.size].clone().cpu(),
            'child':         self.metadata_child[:self.size].clone().cpu(),
            'youth':         self.metadata_youth[:self.size].clone().cpu(),
            'adult':         self.metadata_adult[:self.size].clone().cpu(),
            'elder':         self.metadata_elder[:self.size].clone().cpu(),
            'freq':          self.metadata_freq[:self.size].clone().cpu(),
            'delay':         self.metadata_delay[:self.size].clone().cpu(),
            'origin':        self.metadata_origin[:self.size].clone().cpu(),
            'tension_ce':    self.metadata_tension_ce[:self.size].clone().cpu(),
            'tension_ae':    self.metadata_tension_ae[:self.size].clone().cpu(),
            'version':       self.metadata_version[:self.size].clone().cpu(),
            'timestamps':    self.timestamps[:self.size].clone().cpu(),
            'access_counts': self.access_counts[:self.size].clone().cpu(),
            'size':          self.size,
            'max_memories':  self.max_memories,
            'system_epoch':      self._system_epoch,
            'max_log_time_seen': self._max_log_time_seen,
            'mapping_version': 1,  # v131: log-mapped coords
        }
        torch.save(data, filename)
        print(f"💾 N bank: Saved {self.size} narrative memories → {filename}")

    def load(self, filename=None):
        filename = filename or self.N_MEMORY_FILE
        # v130: Try new filename, fall back to v129 filename for migration
        actual_file = filename
        if not os.path.exists(filename) and os.path.exists(self.V129_N_MEMORY_FILE):
            print(f"🔄 v130 N-bank migration: loading from {self.V129_N_MEMORY_FILE} → will save as {filename}")
            actual_file = self.V129_N_MEMORY_FILE
        if not os.path.exists(actual_file):
            return
        try:
            data = torch.load(actual_file, map_location=self.device)
            sz = min(data['size'], self.max_memories)
            # coords handled below with 5D→7D backward compat
            self.metadata_child[:sz]  = data['metadata_child'][:sz] if 'metadata_child' in data else data['child'][:sz]
            self.metadata_youth[:sz]  = data['metadata_youth'][:sz] if 'metadata_youth' in data else data['youth'][:sz]
            self.metadata_adult[:sz]  = data['metadata_adult'][:sz] if 'metadata_adult' in data else data['adult'][:sz]
            self.metadata_elder[:sz]  = data['metadata_elder'][:sz] if 'metadata_elder' in data else data['elder'][:sz]
            self.metadata_freq[:sz]   = data['metadata_freq'][:sz]  if 'metadata_freq'  in data else data['freq'][:sz]
            self.metadata_delay[:sz]  = data['delay'][:sz] if 'delay' in data else torch.full((sz,), 5.0, dtype=torch.float32)
            self.metadata_origin[:sz]   = data['origin'][:sz].to(self.device)     if 'origin'     in data else torch.zeros(sz).to(self.device)
            self.metadata_tension_ce[:sz] = data['tension_ce'][:sz].to(self.device) if 'tension_ce' in data else torch.zeros(sz).to(self.device)
            self.metadata_tension_ae[:sz] = data['tension_ae'][:sz].to(self.device) if 'tension_ae' in data else torch.zeros(sz).to(self.device)
            self.metadata_version[:sz]    = data['version'][:sz].to(self.device)    if 'version'    in data else torch.zeros(sz).to(self.device)
            self.timestamps[:sz]      = data['timestamps'][:sz]
            self.access_counts[:sz]   = data['access_counts'][:sz]
            
            mapping_ver = data.get('mapping_version', 0)
            loaded_coords = data['coords'][:sz]
            
            # v131: Load lens dims (0-3) and log_time (4) from coords, always valid
            if loaded_coords.shape[1] >= 5:
                self.coords[:sz, :5] = loaded_coords[:, :5].to(self.device)
            elif loaded_coords.shape[1] == 5:
                self.coords[:sz, :5] = loaded_coords.to(self.device)
            else:
                # Very old format — pad
                pad = torch.full((sz, 7 - loaded_coords.shape[1]), math.pi, dtype=torch.float32, device=self.device)
                self.coords[:sz] = torch.cat([loaded_coords.to(self.device), pad], dim=1)
            
            if mapping_ver >= 1:
                # v131: Already log-mapped — load dims 5-6 directly
                if loaded_coords.shape[1] >= 7:
                    self.coords[:sz, 5:7] = loaded_coords[:, 5:7].to(self.device)
            else:
                # v129/v130: Rebuild dims 5-6 from absolute Hz/ms metadata
                print(f"🔄 Converting {sz} N-memories to v131 log coords...")
                hz = self.metadata_freq[:sz].to(self.device)
                ms = self.metadata_delay[:sz].to(self.device)
                self.coords[:sz, 5] = freq_to_log_coord_t(hz)
                self.coords[:sz, 6] = delay_to_log_coord_t(ms)
                self.metadata_version[:sz] = 131.0
                self.save(filename)
                print(f"✅ Converted {sz} N-memories to log coords")
            
            self.size = sz
            # Restore temporal clock — keeps log_time continuous across restarts
            self._system_epoch      = data.get('system_epoch',      MAGI_EPOCH)
            self._max_log_time_seen = data.get('max_log_time_seen', 0.001)
            # v130: If loaded from old filename, save to new filename
            if actual_file != filename:
                self.save(filename)
                print(f"💾 Migrated {sz} N-memories → {filename}")
            print(f"✅ N bank: Loaded {self.size} narrative memories. log_time resumes at {self._max_log_time_seen:.3f}")
        except Exception as e:
            print(f"❌ N bank load failed: {e}")



class MaGiHive:
    def __init__(self, num_workers, device):
        self.n = num_workers
        self.dev = device
        self.mode = 'webcam'
        self.video_source = None
        self.remote_sender = None   # RemoteCommandSender, set by 'mode remote <ip>'
        # v133: Robot arm state
        self.robot_sender = None             # RobotArmSender, set by 'robot enable <ip>'
        self.robot_mode = 0                  # 0=off, 1-4=progressive complexity
        self._robot_active_workers = set()   # indices active in current mode

        self.upe = UniversalPlasticityEngine(device=self.dev)
        
        # v130: 6D Toroidal State [child, youth, adult, elder, freq_phase, delay_phase]
        self.pos_6d = torch.zeros((self.n, 6), device=self.dev)
        self.vel_6d = torch.zeros((self.n, 6), device=self.dev)
        
        # Initialize lens dims (0-3) with canonical quadrants
        _base_lens = torch.tensor([0.0, 1.57, 3.14, 4.71], device=self.dev)
        self.pos_6d[:, :4] = _base_lens.unsqueeze(0).expand(self.n, -1)
        
        # Initialize freq/delay as phases (4-5) from default values (1.0 Hz, 5.0 ms)
        self.pos_6d[:, 4] = freq_to_phase(1.0)
        self.pos_6d[:, 5] = delay_to_phase(5.0)
        
        # v131: Phase-space momentum accumulators (radians, not Hz)
        self.freq_phase_momentum = torch.zeros(self.n, device=self.dev)
        self.delay_phase_momentum = torch.zeros(self.n, device=self.dev)
        # v131: Wrap counters — persist across restarts for absolute Hz
        self.freq_wrap  = torch.zeros(self.n, dtype=torch.int32, device=self.dev)
        self.delay_wrap = torch.zeros(self.n, dtype=torch.int32, device=self.dev)
        # v131: Impulse buffers — BH writes here, applied after velocity assembly
        # This prevents velocity assembly (=) from overwriting BH impulses (+=)
        self._impulse_vel_6d = torch.zeros((self.n, 6), device=self.dev)
        self._impulse_vel_s  = torch.zeros((self.n, 4), device=self.dev)
        
        self.hb_sim_phase = torch.zeros(self.n, device=self.dev)
        self.phases_s  = torch.tensor([0.1, 1.67, 3.24, 4.81], device=self.dev).repeat(self.n, 1)
        self.vel_s  = torch.zeros((self.n, 4), device=self.dev)
        self.hb_filtered = torch.full((self.n,), 250.0, device=self.dev); self.hb_last = self.hb_filtered.clone(); self.hb_integral = self.hb_filtered.clone()
        self.s_filtered = torch.full((self.n,), 250.0, device=self.dev); self.s_last = self.s_filtered.clone(); self.s_integral = self.s_filtered.clone()
        
        self.global_coh = torch.zeros(self.n, device=self.dev)
        self.hb_coh = torch.zeros(self.n, device=self.dev)
        self.s_coh = torch.zeros(self.n, device=self.dev)
        self.cross_tension = torch.zeros(self.n, device=self.dev)
        self.adult_dir = torch.zeros(self.n, device=self.dev)
        self.elder_dir = torch.zeros(self.n, device=self.dev)
        self.alignment_diff = torch.zeros(self.n, device=self.dev)
        self.quadrant_counts = torch.zeros((self.n, 4), device=self.dev)
        self.total_steps = 0
        
        self.memory_bank = EnhancedHypersphereMemory(dim=130, max_memories=MAX_MEMORIES, device=self.dev)
        self.current_gravity_context = {}
        
        self.last_chord_audio = {}
        self.last_chord_video = {}
        
       # 0x01: Motor Array (1542-1547) - Zero Initialized for Bipolar
        # Unified Worker Setup (Replaces hardcoded ALE/VOICE loops)
        for idx, data in self.upe.homes.items():
            # v131: Apply 6D home position [lens4, freq_phase, delay_phase]
            home_6d = data['home']
            if home_6d.shape[0] == 4:  # v129 legacy home
                # Convert 4D to 6D
                fp = freq_to_phase(1.0)
                dp = delay_to_phase(5.0)
                home_6d = torch.cat([home_6d, torch.tensor([fp, dp], device=self.dev)])
                data['home'] = home_6d  # Update stored home
            
            self.pos_6d[idx] = home_6d.clone().to(self.dev)
            self.phases_s[idx] = (self.pos_6d[idx, :4] + 0.5) % (2 * math.pi)
            self.vel_6d[idx] = 0.0
            # Zero initialization for bipolar swing
            self.s_filtered[idx] = 0.0
            self.s_last[idx] = 0.0
            self.s_integral[idx] = 0.0
            self.vel_s[idx] = 0.0
        
        # v131: Restore wrap counters from UPE save (persisted for absolute Hz)
        upe_meta = self.upe._saved_metadata
        if '_freq_wrap' in upe_meta:
            saved_fw = upe_meta['_freq_wrap'].to(self.dev)
            saved_dw = upe_meta['_delay_wrap'].to(self.dev)
            if saved_fw.shape[0] < self.n:
                self.freq_wrap[:saved_fw.shape[0]] = saved_fw
                self.delay_wrap[:saved_dw.shape[0]] = saved_dw
                print(f"✅ Restored wrap counters (padded {saved_fw.shape[0]} → {self.n})")
            else:
                self.freq_wrap = saved_fw
                self.delay_wrap = saved_dw
                print(f"✅ Restored wrap counters (freq range: {self.freq_wrap.min().item()} to {self.freq_wrap.max().item()})")

        self.scalers = {}

        # In your scaler initialization loop:

        # Fire is still unipolar (0-1) - fire is fire
        # NOOP should be bipolar (-1 to 1) - can be "anti-action"

        ale_names = ['ale_left', 'ale_right', 'ale_fire', 'ale_up', 'ale_down', 'ale_noop']
        for i, name in enumerate(ale_names):
            worker_idx = 1542 + i
            
            # NOOP is now bipolar too!
            is_directional = name in ['ale_left', 'ale_right', 'ale_up', 'ale_down', 'ale_noop']
            
            self.scalers[name] = AdaptiveScaler(
                name=name,
                mode='range',
                input_range_min=0.1,
                input_range_max=1000.0,
                output_range_min=-1.0 if is_directional else 0.0,  # NOOP gets -1 to 1
                output_range_max=1.0,
                spacing='linear',
                bipolar=is_directional,  # NOOP gets bipolar=True
                min_threshold=0.2,
                initial_max=10.0,
                track_signal=True,
                track_window_seconds=600.0,
                track_decay_rate=0.9999,
                dead_zone_mode='linear',
                dead_zone_value=0.1
            )

        
        # BLACK HOLE SCALER (1549) - Stage 3
        self.scalers['black_hole'] = AdaptiveScaler(
            name="black_hole",
            mode='range',
            input_range_min=0.01,
            input_range_max=115.0,
            output_range_min=-1500.0,
            output_range_max=1500.0,
            spacing='linear',
            bipolar=True,
            min_threshold=0.05,
            initial_max=400.0,
            track_signal=True,
            track_window_seconds=600.0,
            track_decay_rate=0.9999,
            dead_zone_mode='linear',
            dead_zone_value=0.08,
            clip_output=True
        )

        # v133: Robot arm scalers (bipolar, like directional ALE workers)
        robot_names = ['robot_x', 'robot_y', 'robot_z', 'robot_rot', 'robot_tilt', 'robot_gripper']
        for i, name in enumerate(robot_names):
            self.scalers[name] = AdaptiveScaler(
                name=name,
                mode='range',
                input_range_min=0.1,
                input_range_max=1000.0,
                output_range_min=-1.0,
                output_range_max=1.0,
                spacing='linear',
                bipolar=True,
                min_threshold=0.2,
                initial_max=10.0,
                track_signal=True,
                track_window_seconds=600.0,
                track_decay_rate=0.9999,
                dead_zone_mode='linear',
                dead_zone_value=0.1
            )

        # 0x02: Voice Worker (1548) - Isolated Quadrant
        self.voice_worker_idx = 1548
        # self.voice_carrier = PureCarrierVoice()
        
        # # Quadrant flip [- - + +] at Radius 3.8 (Far from BH [0,0,0,0] and ALE [+ +])
        # voice_dir = torch.tensor([4.71, 3.14, 1.57, 0.0], device=self.dev)
        # # old: 4.71, 3.14, 1.57, 0.0
        # # new: -1.2, -0.8, 1.1, 1.3
        # self.phases_hb[self.voice_worker_idx] = F.normalize(voice_dir, p=2, dim=0) * 3.8
        # self.phases_s[self.voice_worker_idx] = (self.phases_hb[self.voice_worker_idx] + 0.5) % (2 * math.pi)
        
        # # Zero initialization
        # self.s_filtered[self.voice_worker_idx] = 0.0
        # self.s_last[self.voice_worker_idx] = 0.0
        # self.s_integral[self.voice_worker_idx] = 0.0
        # self.vel_hb[self.voice_worker_idx] = 0.0
        self.voice_carrier = PureCarrierVoice()
        
        # v130: Voice worker 6D position - lens from UPE, freq/delay from defaults
        if self.voice_worker_idx in self.upe.homes:
            voice_home = self.upe.homes[self.voice_worker_idx]['home']
            if voice_home.shape[0] == 4:
                fp = freq_to_phase(1.0)
                dp = delay_to_phase(5.0)
                voice_home = torch.cat([voice_home, torch.tensor([fp, dp], device=self.dev)])
            self.pos_6d[self.voice_worker_idx] = voice_home
        else:
            self.pos_6d[self.voice_worker_idx, :4] = torch.tensor([4.71, 3.14, 1.57, 0.0], device=self.dev)
            self.pos_6d[self.voice_worker_idx, 4] = freq_to_phase(1.0)
            self.pos_6d[self.voice_worker_idx, 5] = delay_to_phase(5.0)
        
        self.phases_s[self.voice_worker_idx] = (self.pos_6d[self.voice_worker_idx, :4] + 0.5) % (2 * math.pi)
        


        # Neutral starting values
        self.s_filtered[self.voice_worker_idx] = 0.0
        self.s_last[self.voice_worker_idx] = 0.0
        self.s_integral[self.voice_worker_idx] = 0.0

        # ⚡ Bare metal CUDA physics kernel — optional, graceful fallback
        self.use_cuda_kernel = False

        if _MAGI_CUDA_V117_AVAILABLE:
            try:
                self.magi_cuda = MagiCUDAv117(verbose=True)
                self.use_cuda_kernel = self.magi_cuda.is_available()
            except Exception as e:
                print(f"⚠️  CUDA kernel init failed: {e} — Python path active")

        
        
        print(f"🎵 Pure Carrier Voice at worker {self.voice_worker_idx}: [0,0,0,0] phase")

        # v135: Resonance Bridge controller — discovers type='bridge' workers
        # from UPE by parsing the BRIDGE<id>_<ENT|DEST> name pattern.
        self.bridge_controller = BridgeController(magi_hive=self)

                # Black Hole Worker (Memory Deletion)
        self.black_hole_worker_idx = 1549
        self.black_hole_deletion_enabled = True

        # Centered in the void (consensus placement)
        # v130: Black Hole 6D position
        bh_home = torch.tensor([0.01, -0.01, 0.01, -0.01], device=self.dev)
        bh_fp = freq_to_phase(1.0)
        bh_dp = delay_to_phase(5.0)
        self.pos_6d[self.black_hole_worker_idx] = torch.cat([bh_home, torch.tensor([bh_fp, bh_dp], device=self.dev)])
        self.phases_s[self.black_hole_worker_idx] = (self.pos_6d[self.black_hole_worker_idx, :4] + 0.5) % (2 * math.pi)

        # Neutral starting values
        self.s_filtered[self.black_hole_worker_idx] = 0.0
        self.s_last[self.black_hole_worker_idx] = 0.0
        self.s_integral[self.black_hole_worker_idx] = 0.0

        # Singularity Parameters
        self.black_hole_base_radius = 0.05    # Sensing threshold (at Value=0)
        self.black_hole_eps_max = 2.5e-2        # Max Power (Industrial Erasure)
        self.black_hole_eps_floor = 1e-4      # Min Power (Sensing/Safety)
        self.black_hole_collapse_threshold = 1e-6 # Event Horizon

        # Tracking metrics
        self.black_hole_daily_deletions = 0
        self.black_hole_step_deletions = 0
        self.black_hole_memories_in_field = 0     # NEW: Track "pressure without deletion"
        self.black_hole_creation_count = 0        # NEW: Track creation for ratio
        self.black_hole_window_start_time = time.time()  # NEW: For windowed metrics

        # ==========================================
        # v102: N HYPERSPHERE — TEMPORAL NARRATIVE LAYER
        # ==========================================
        self.n_bank = NHypersphereMemory(max_memories=3000000, device=self.dev)
        self.n_bank.load()

        # Temporal clock — lives inside n_bank so it persists across restarts.
        # After load(), _system_epoch and _max_log_time_seen are restored from disk.
        # On first run they default to time.time() / 0.001 (set in NHypersphereMemory.__init__).

        # N Black Hole at worker 1551 — excluded from UPE, free to roam anywhere
        self.n_bh_worker_idx = 1551

        # Start on the opposite side of the hypersphere from main BH [0.01, -0.01, 0.01, -0.01]
        # π offset on all dims → maximally distant starting position
        self.pos_6d[self.n_bh_worker_idx, :4] = torch.tensor([3.20, 3.10, 3.20, 3.10], device=self.dev)
        # v130: N Black Hole 6D position (opposite side of torus)
        nbh_fp = freq_to_phase(50.0)  # Different freq band
        nbh_dp = delay_to_phase(50.0)
        self.pos_6d[self.n_bh_worker_idx, 4] = nbh_fp
        self.pos_6d[self.n_bh_worker_idx, 5] = nbh_dp
        self.phases_s[self.n_bh_worker_idx] = (self.pos_6d[self.n_bh_worker_idx, :4] + 0.5) % (2 * math.pi)
        self.s_filtered[self.n_bh_worker_idx] = 0.0
        self.s_last[self.n_bh_worker_idx]     = 0.0
        self.s_integral[self.n_bh_worker_idx] = 0.0

        # N BH singularity params — mirrors main BH, no special rules
        # MaGi moves it wherever the field takes it
        self.n_bh_base_radius        = 0.05   # matches main BH base_radius
        self.n_bh_eps_max            = 2.5e-2   # matches main BH eps_max
        self.n_bh_eps_floor          = 1e-4   # matches main BH eps_floor
        self.n_bh_collapse_threshold = 1e-6   # matches main BH collapse_threshold
        self.n_bh_step_deletions     = 0
        self.n_bh_memories_in_field  = 0
        self.n_bh_session_deletions  = 0   # mirrors black_hole_daily_deletions
        self.n_bh_creation_count     = 0   # mirrors black_hole_creation_count
        self.n_bh_window_start_time  = time.time()

        # N BH scaler — exact mirror of main BH
        self.scalers['n_black_hole'] = AdaptiveScaler(
            name="n_black_hole",
            mode='range',
            input_range_min=0.01,
            input_range_max=150.0,
            output_range_min=-1500.0,
            output_range_max=1500.0,
            spacing='linear',
            bipolar=True,
            min_threshold=0.05,
            initial_max=400.0,
            track_signal=True,
            track_window_seconds=600.0,
            track_decay_rate=0.9999,
            dead_zone_mode='linear',
            dead_zone_value=0.08,
            clip_output=True
        )
        self.n_bh_last_result = {'is_active': False, 'output': 0.0, 'signal_strength': 0.0}

        # Carrier N-retrieval results (audio + video, updated by query_n_carriers)
        self.n_gravity_audio = {}
        self.n_gravity_video = {}

        # Temporal chord vibration constants
        self.N_VIB_MAX   = 0.002   # per-note amplitude ceiling
        self.N_VIB_CLAMP = 0.005   # total per-step vel_s clamp per dim
        self.n_vib_phase = {}      # per-carrier phase accumulator {carrier_idx: float}

        # N query neighborhood gate — only query N when carrier cluster holds
        # focus for N_STREAK_REQUIRED consecutive steps. Tracks by freq band
        # not worker index, so baton-passing within a cluster doesn't reset streak.
        self.N_STREAK_REQUIRED  = 3
        self._audio_streak      = 0
        self._last_audio_freq   = -1.0
        self._video_streak      = 0
        self._last_video_freq   = -1.0
        self.N_FREQ_BAND        = 1.5   # Hz — neighbourhood tolerance

        print(f"🌀 N Hypersphere: bank 200k cap | N BH at worker {self.n_bh_worker_idx}")
        print(f"   Main BH (1549) unchanged — Ale lives in main hypersphere")

        # v130: ALE workers (1542-1547) get neutral freq/delay phases
        # They will drift emergently; no special springs or tethers.
        _ale_neutral_freq = 250.0  # Center of gamma band
        _ale_neutral_delay = 5.0   # Neutral delay

        for idx in range(1542, 1548):
            # Keep lens phases from UPE homes (already set)
            # Set freq/delay phases to neutral
            self.pos_6d[idx, 4] = freq_to_phase(_ale_neutral_freq)
            self.pos_6d[idx, 5] = delay_to_phase(_ale_neutral_delay)

        # ==========================================
        # v126: KINETIC MANIFOLD — ALE Beacons + Dream Mirror
        # ==========================================

        # ── Part 1: ALE Vibration Beacons (one per ALE worker) ──────────────
        self.ale_beacons = {}
        for ale_idx, sig in ALE_VIB_SIGNATURES.items():
            self.ale_beacons[ale_idx] = AleVibrationBeacon(
                ale_idx, sig, self.upe, self.dev
            )

        # ── Part 2: Dream Mirror Workers ────────────────────────────────────
        km    = self.upe.km_config
        n_idx = km['dream_n_reader_idx']   # 1552
        m_idx = km['dream_main_anchor_idx'] # 1553

        # N-Reader: π/2 offset from both black holes
        # Main BH: [0.01,-0.01,0.01,-0.01]   N BH: [3.20,3.10,3.20,3.10]
        # v130: Dream Mirror workers 6D positions
        # N-Reader (1552)
        self.pos_6d[n_idx, :4] = torch.tensor([1.57, 1.57, -1.57, -1.57], device=self.dev)
        self.pos_6d[n_idx, 4] = freq_to_phase(10.0)
        self.pos_6d[n_idx, 5] = delay_to_phase(10.0)
        
        # Main-Anchor (1553)  
        self.pos_6d[m_idx, :4] = torch.tensor([4.71, 0.01, 1.57, 3.14], device=self.dev)
        self.pos_6d[m_idx, 4] = freq_to_phase(20.0)
        self.pos_6d[m_idx, 5] = delay_to_phase(20.0)
        
        for widx in [n_idx, m_idx]:
            self.phases_s[widx] = (self.pos_6d[widx, :4] + 0.5) % (2 * math.pi)
            self.vel_6d[widx] = 0.0
            self.s_filtered[widx] = 0.0
            self.s_last[widx]     = 0.0
            self.s_integral[widx] = 0.0
            self.vel_s[widx].zero_()

        self.dream_coupling = DreamMirrorCoupling(self.upe, self.dev)

        # ── Part 3: Chord (Teleport) pair initial positions ─────────────────
        # v130: Chord Teleport pair 6D
        self.pos_6d[1554, :4] = torch.tensor([0.0, 1.57, 3.14, 4.71], device=self.dev)
        self.pos_6d[1554, 4] = freq_to_phase(5.0)
        self.pos_6d[1554, 5] = delay_to_phase(5.0)

        self.pos_6d[1555, :4] = torch.tensor([3.14, 4.71, 0.0, 1.57], device=self.dev)
        self.pos_6d[1555, 4] = freq_to_phase(15.0)
        self.pos_6d[1555, 5] = delay_to_phase(15.0)

        # ── Part 4: Physics (Lens-Driven) pair initial positions ─────────────
        # v130: Physics pair 6D
        self.pos_6d[1556, :4] = torch.tensor([1.0, 2.5, 4.0, 5.5], device=self.dev)
        self.pos_6d[1556, 4] = freq_to_phase(30.0)
        self.pos_6d[1556, 5] = delay_to_phase(30.0)

        self.pos_6d[1557, :4] = torch.tensor([4.0, 5.5, 1.0, 2.5], device=self.dev)
        self.pos_6d[1557, 4] = freq_to_phase(40.0)
        self.pos_6d[1557, 5] = delay_to_phase(40.0)

        # Wrap to [0, 2π] and zero all state for new workers
        for idx in (1554, 1555, 1556, 1557):
            self.pos_6d[idx] = self.pos_6d[idx] % (2 * math.pi)
            self.phases_s[idx] = (self.pos_6d[idx, :4] + 0.5) % (2 * math.pi)
            self.s_filtered[idx] = 0.0
            self.s_last[idx]     = 0.0
            self.s_integral[idx] = 0.0
            self.vel_6d[idx] = 0.0
            self.vel_s[idx].zero_()

        self.chord_coupling   = ChordTeleportCoupling(self.upe, self.dev)
        self.physics_coupling = PhysicsCoupling(self.upe, self.dev)

        # Buffer for inputs_tensor — read by DreamMirrorCoupling.update()
        self._inputs_buffer = torch.zeros(self.n, device=self.dev)

        print(f"🎯 Kinetic Manifold v128:")
        print(f"   ALE Beacons: workers 1542-1547 (continuous directional vibration)")
        print(f"   Dream Mirror Drift:   N-Reader [1552] ↔ Main-Anchor [1553] (memory-directed impulse + drift)")
        print(f"   Dream Mirror Chord:   N-Reader [1554] ↔ Main-Anchor [1555] (episodic teleport + drift)")
        print(f"   Dream Mirror Physics: N-Reader [1556] ↔ Main-Anchor [1557] (lens-driven attraction + drift)")

        # ── v133: Robot Arm Workers (1558-1563) ──────────────────────────────
        # Initialize positions from UPE homes (same pattern as ALE workers)
        for idx in range(1558, 1564):
            if idx in self.upe.homes:
                home_6d = self.upe.homes[idx]['home']
                if home_6d.shape[0] == 4:  # Legacy conversion: 4D → 6D
                    fp = freq_to_phase(1.0)
                    dp = delay_to_phase(5.0)
                    home_6d = torch.cat([home_6d, torch.tensor([fp, dp], device=self.dev)])
                    self.upe.homes[idx]['home'] = home_6d
                self.pos_6d[idx] = home_6d.clone().to(self.dev)
                self.phases_s[idx] = (self.pos_6d[idx, :4] + 0.5) % (2 * math.pi)
                self.vel_6d[idx] = 0.0
                self.s_filtered[idx] = 0.0
                self.s_last[idx] = 0.0
                self.s_integral[idx] = 0.0
                self.vel_s[idx].zero_()

        # Robot vibration beacons (RobotVibrationBeacon reads robot_vib_* km_config)
        self.robot_beacons = {}
        for rob_idx, sig in ROBOT_VIB_SIGNATURES.items():
            self.robot_beacons[rob_idx] = RobotVibrationBeacon(
                rob_idx, sig, self.upe, self.dev
            )

        print(f"🦾 Robot Arm: workers 1558-1563 (disabled — use 'robot enable <ip>' to activate)")
        print(f"   Modes: 1=XY+Grip  2=XYZ+Grip  3=XYZR+Grip  4=XYZRT+Grip (all 6)")



        # lens_weights    — learned, updated each step
        # baseline_weights — frozen anchor, NEVER updated after init
        # Identity change happens via Hz drift, not baseline mutation.
        # ALE/Voice/BH workers (1542+) are excluded from updates.
        # ==========================================
        self.lens_weights     = torch.zeros((self.n, 4), device=self.dev)
        self.baseline_weights = torch.zeros((self.n, 4), device=self.dev)
        for w in range(self.n):
            band = get_band(self.freq[w].item())
            vec  = torch.tensor(
                [BAND_BASELINES[band][k] for k in ('child', 'youth', 'adult', 'elder')],
                device=self.dev
            )
            self.lens_weights[w]     = vec
            self.baseline_weights[w] = vec  # frozen — never written after this line

        self.global_age = 0  # step counter

    @property
    def freq(self):
        """v131: Absolute Hz from phase + wrap count (unbounded)."""
        return phase_to_freq_t(self.pos_6d[:, 4], self.freq_wrap)

    @property
    def delay(self):
        """v131: Absolute ms from phase + wrap count (unbounded)."""
        return phase_to_delay_t(self.pos_6d[:, 5], self.delay_wrap)

    def calculate_lens_output(self, type_idx, val, deriv, integral):
        # v100: hive workers (0:1542) use per-worker lens_weights
        # ALE/Voice/BH workers (1542+) use original hardcoded constants — unchanged
        child     = self.lens_weights[:, 0].clone(); child[1542:]     = CHILD_SENSITIVITY
        youth     = self.lens_weights[:, 1].clone(); youth[1542:]     = YOUTH_GAIN
        threshold = self.lens_weights[:, 2].clone(); threshold[1542:] = ADULT_THRESHOLD
        elder     = self.lens_weights[:, 3].clone(); elder[1542:]     = ELDER_TIME_CONSTANT
        if type_idx == 0:
            return (torch.abs(deriv) / 40.0 * child) * torch.exp(-torch.abs(deriv)/40.0 * torch.abs(deriv)/40.0 / 2.0)
        elif type_idx == 1:
            return torch.clamp(youth * (val / 500.0), 0.0, 1.0)
        elif type_idx == 2:
            inp = (0.6 * torch.clamp(youth * (val / 500.0), 0.0, 1.0) + 0.4 * torch.abs(deriv) / 25.0) - threshold
            return torch.clamp(inp / (1.0 + torch.exp(-8.0 * inp)), 0.0, 1.0)
        elif type_idx == 3:
            return (torch.tanh((integral - 250.0) * (4.0 / 300.0) - 2.0) + 1.0) / 2.0 * elder
        return torch.zeros_like(val)

    def update_quadrant_stats(self):
        adult = self.adult_dir
        ne_mask = (adult < 45) | (adult >= 315); se_mask = (adult >= 45) & (adult < 135)
        sw_mask = (adult >= 135) & (adult < 225); nw_mask = (adult >= 225) & (adult < 315)
        self.quadrant_counts[:, 0] += ne_mask.float(); self.quadrant_counts[:, 1] += se_mask.float()
        self.quadrant_counts[:, 2] += sw_mask.float(); self.quadrant_counts[:, 3] += nw_mask.float()
        self.total_steps += 1
        
    def get_quadrant_metrics(self, worker_idx):
        if self.total_steps == 0: return 0.0, 0.0
        quadrant_pct = (self.quadrant_counts[worker_idx] / self.total_steps) * 100.0
        return torch.max(quadrant_pct).item(), quadrant_pct[0].item()

    # ── v130: N-bank metadata → 6D main-bank target phase ──────────────────────
    def memory_to_main_target(self, mem_idx):
        """Translate N-bank metadata for mem_idx into a 6D main-bank target phase."""
        freq = self.n_bank.metadata_freq[mem_idx].item()
        delay = self.n_bank.metadata_delay[mem_idx].item()
        
        # Convert to phases
        freq_phase = freq_to_phase(freq)
        delay_phase = delay_to_phase(delay)
        
        # Lens dims from stored coords
        child = self.n_bank.coords[mem_idx, 0].item()
        youth = self.n_bank.coords[mem_idx, 1].item()
        adult = self.n_bank.coords[mem_idx, 2].item()
        elder = self.n_bank.coords[mem_idx, 3].item()
        
        return torch.tensor([child, youth, adult, elder, freq_phase, delay_phase], device=self.dev)

    # ── v130: N-bank metadata → persistent 6D drift vector ─────────────────────
    def memory_to_drift_vector(self, mem_idx):
        """Compute a persistent 6D drift vector from lens weights + access count for mem_idx."""
        access     = self.n_bank.access_counts[mem_idx].item()
        child      = self.n_bank.metadata_child[mem_idx].item()
        adult      = self.n_bank.metadata_adult[mem_idx].item()
        elder      = self.n_bank.metadata_elder[mem_idx].item()
        tension_ce = self.n_bank.metadata_tension_ce[mem_idx].item()

        speed         = min(access / 100.0, 1.0)
        dir_freq      = child * 2.0 - 1.0
        dir_delay     = elder * 2.0 - 1.0
        dir_adult     = adult * 2.0 - 1.0
        tension_factor = tension_ce / math.pi
        
        # v130: Return 6D drift vector
        freq_phase_target = freq_to_phase(self.n_bank.metadata_freq[mem_idx])
        delay_phase_target = delay_to_phase(self.n_bank.metadata_delay[mem_idx])

        direction = torch.tensor([
            dir_freq,
            dir_delay,
            dir_adult + tension_factor,
            0.0,
            math.sin(freq_phase_target.item() if hasattr(freq_phase_target, 'item') else freq_phase_target) * 0.5,  # Freq dim oscillation
            math.sin(delay_phase_target.item() if hasattr(delay_phase_target, 'item') else delay_phase_target) * 0.5  # Delay dim oscillation
        ], device=self.dev)
        norm = torch.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        drift_strength = self.upe.km_config.get('main_drift_strength', 0.001)
        return direction * speed * drift_strength

    def process_step(self, inputs_tensor):
        # Store input buffer for DreamMirrorCoupling energy reading
        self._inputs_buffer = inputs_tensor

        # v130: Sensory low-pass filter (Anti-Whiplash) - lerp 0.1 toward target
        target_s = 0.6 * self.s_filtered + 0.4 * inputs_tensor
        self.s_filtered = 0.9 * self.s_filtered + 0.1 * target_s
        s_deriv = self.s_filtered - self.s_last
        self.s_last = self.s_filtered.clone()
        self.s_integral = 0.80 * self.s_integral + 0.20 * self.s_filtered

        # UPE physics (6D gravity applied to homes)
        self.apply_universal_plasticity()

        # HB Simulation (closed loop: freq/delay properties read pos_6d)
        delta_time = self.delay / 1000.0
        delta_phase = 2.0 * math.pi * self.freq * delta_time
        self.hb_sim_phase = (self.hb_sim_phase + delta_phase) % (2 * math.pi)

        sim_hb = torch.abs(torch.sin(self.hb_sim_phase)) * HB_SINE_SCALE
        self.hb_filtered = 0.8 * self.hb_filtered + 0.2 * sim_hb
        hb_deriv = self.hb_filtered - self.hb_last
        self.hb_last = self.hb_filtered.clone()
        self.hb_integral = ELDER_TIME_CONSTANT * self.hb_integral + (1.0 - ELDER_TIME_CONSTANT) * self.hb_filtered

        hb_norm = self.hb_filtered / 500.0
        hb_deriv_norm = torch.abs(hb_deriv) / 10.0
        hb_int_norm = self.hb_integral / 500.0

        # Velocity Assembly (6D)
        if self.use_cuda_kernel:
            # Kernel stays 4D - only lens dims
            lw = self.lens_weights.clone()
            lw[1542:, 0] = CHILD_SENSITIVITY
            lw[1542:, 1] = YOUTH_GAIN
            lw[1542:, 2] = ADULT_THRESHOLD
            lw[1542:, 3] = ELDER_TIME_CONSTANT

            vel_lens = torch.empty((self.n, 4), device=self.dev, dtype=torch.float32)
            vel_s = torch.empty((self.n, 4), device=self.dev, dtype=torch.float32)
            
            self.magi_cuda.compute(
                self.pos_6d[:, :4].contiguous(),  # Pass lens slice only
                self.phases_s,
                self.s_filtered, s_deriv, self.s_integral,
                hb_norm, hb_deriv_norm, hb_int_norm,
                lw, inputs_tensor,
                vel_lens, vel_s, self.n
            )
            self.vel_6d[:, :4] = vel_lens
            self.vel_s = vel_s
        else:
            # Python path
            base_vel = torch.tensor([0.04, 0.025, 0.015, 0.005], device=self.dev).repeat(self.n, 1)
            self.vel_6d[:, :4] = base_vel + 0.15 * hb_deriv_norm.unsqueeze(1) + 0.05 * hb_norm.unsqueeze(1)

            # S-oscillator (4D, unchanged)
            mask = torch.ones(self.n, device=self.dev)
            mask[1548] = 0.0
            mask[1549] = 0.0
            mask[1551] = 0.0
            for dream_idx in range(1552, 1558):
                mask[dream_idx] = 0.0
                
            for i in range(4):
                out = self.calculate_lens_output(i, self.s_filtered, s_deriv, self.s_integral)
                base = [0.05, 0.03, 0.02, 0.01][i]
                gravity = torch.sin(self.pos_6d[:, i] - self.phases_s[:, i]) * 0.02 * (inputs_tensor / 500.0)
                self.vel_s[:, i] = base + (0.08 * out.flatten() * mask) + gravity.flatten()

        # v131: Phase-space momentum → velocity (no Jacobian, no 1/f asymmetry)
        # Momentum IS velocity — clamped at π for Nyquist stability
        self.vel_6d[:, 4] = self.freq_phase_momentum
        self.vel_6d[:, 5] = self.delay_phase_momentum

        # ── v131: Apply buffered impulses + beacons AFTER assembly, BEFORE integration ──
        # BH impulses were buffered during apply_universal_plasticity() to survive
        # the velocity assembly's = overwrites above.
        # Beacons fire here (not in main loop) so they also survive.
        self.vel_6d += self._impulse_vel_6d       # 6D BH directional push
        self.vel_s  += self._impulse_vel_s        # 4D BH future-location indicator
        self._impulse_vel_6d.zero_()              # Clear for next step
        self._impulse_vel_s.zero_()
        # ALE beacons: constant 6D identity broadcast (on top of assembled velocity)
        for beacon in self.ale_beacons.values():
            beacon.vibrate(self)
        # v133: Robot beacons — only fire for workers active in current robot mode
        for beacon in self.robot_beacons.values():
            if beacon.idx in self._robot_active_workers:
                beacon.vibrate(self)
            # else: phase_accum frozen, worker silent

        # v135: Resonance Bridge beacons — fire pre-integration like ALE/robot
        # beacons so terrain signature on vel_6d[4:5] gets integrated this frame.
        # Writing to vel_6d after integration would be clobbered next frame by
        # velocity assembly's `vel_6d[:, 4] = freq_phase_momentum` overwrite.
        self.bridge_controller.fire_beacons(self, self.mode)

        # Track wraps BEFORE mod (for absolute Hz recovery)
        old_freq_phase  = self.pos_6d[:, 4].clone()
        old_delay_phase = self.pos_6d[:, 5].clone()
        
        # Toroidal Integration (No Walls)
        self.pos_6d = (self.pos_6d + self.vel_6d) % (2 * math.pi)
        self.phases_s = (self.phases_s + self.vel_s) % (2 * math.pi)

        # Count wraps via unwrapped position
        unwrapped_freq  = old_freq_phase + self.vel_6d[:, 4]
        unwrapped_delay = old_delay_phase + self.vel_6d[:, 5]
        self.freq_wrap  += torch.div(unwrapped_freq,  TWO_PI, rounding_mode='floor').to(torch.int32)
        self.delay_wrap += torch.div(unwrapped_delay, TWO_PI, rounding_mode='floor').to(torch.int32)

        # v135: Resonance Bridge teleport — runs AFTER modulo AND wrap counting.
        # Overwrites destination's dims 0-1 with entrance's post-integration
        # dims 0-1 as final position write for the frame. Also zeros dest's
        # vel_6d[0:1] (stale velocity on a pinned dim would double-count next
        # frame). Touches only dims 0-1; wrap counters already settled from 4-5.
        self.bridge_controller.teleport_destinations(self, self.mode)

        self.voice_carrier.speak(self)
        self.global_age += 1

    def update_metrics(self):
        def calc_group_coh(phases):
            p_sum = torch.zeros(self.n, device=self.dev)
            for a in range(4):
                for b in range(a+1, 4):
                    diff = torch.abs(phases[:, a] - phases[:, b])
                    p_sum += torch.cos(torch.where(diff > math.pi, 2*math.pi - diff, diff))
            return (p_sum / 6.0 + 1.0) / 2.0

        self.hb_coh = calc_group_coh(self.pos_6d[:, :4])  # Lens dims only
        self.s_coh = calc_group_coh(self.phases_s)
        self.cross_tension = torch.cos(self.pos_6d[:, :4] - self.phases_s).mean(dim=1)
        self.global_coh = (self.hb_coh + self.s_coh + (self.cross_tension + 1.0)/2.0) / 3.0
        
        self.adult_dir = (self.pos_6d[:, 2] * 180.0 / math.pi) % 360.0
        self.elder_dir = (self.pos_6d[:, 3] * 180.0 / math.pi) % 360.0
        diff = torch.abs(self.adult_dir - self.elder_dir)
        self.alignment_diff = torch.where(diff > 180, 360 - diff, diff)
        self.update_quadrant_stats()
        
    def _get_full_state_snapshot(self):
        return {
            'phases_hb': self.pos_6d[:, :4],   # v130: Lens dims only (4D for encoder)
            'phases_s': self.phases_s,
            'global_coh': self.global_coh, 'cross_tension': self.cross_tension,
            'freq': self.freq,                # Uses property derived from pos_6d
            'delay': self.delay,              # Uses property derived from pos_6d
        }

    def apply_natural_physics_and_memory(self, step):
        # v131: Cross-tension removed from freq/delay (inflationary in log space)
        # Freq/delay forces come only from memory gravity and centered coherence coupling
        freq_phase_force = torch.zeros(self.n, device=self.dev)
        delay_phase_force = (0.5 - self.global_coh) * DELAY_PHASE_K  # centered: oscillates, no drift
        
        self.current_gravity_context = {}
        best_idx = torch.argmax(self.global_coh).item()
        
        best_emb = self.memory_bank.encode({
            'phases_hb': self.pos_6d[best_idx, :4], 'phases_s': self.phases_s[best_idx],
            'global_coh': self.global_coh[best_idx], 'cross_tension': self.cross_tension[best_idx],
            'freq': self.freq[best_idx], 'delay': self.delay[best_idx]
        }, all_workers=False)
        
        if (self.global_coh[best_idx] > STORE_COHERENCE_STABLE and self.memory_bank.is_novel(best_emb) and step % 1 == 0): 
            # v131: Build 6D coords with lens phases + unwrapped log freq/delay
            store_coords_6d = torch.cat([
                self.pos_6d[best_idx, :4],
                torch.tensor([freq_to_log_coord(self.freq[best_idx].item()),
                              delay_to_log_coord(self.delay[best_idx].item())], device=self.dev)
            ])
            self.memory_bank.store(best_emb, {
                'freq': self.freq[best_idx].item(),
                'delay': self.delay[best_idx].item(),
                'coords_6d': store_coords_6d
            })
            self.black_hole_creation_count += 1

            # ==========================================
            # v102: N PIGGYBACK STORE
            # Fires every time main bank stores — couples N to meaningful moments.
            # Two separate memories: single most coherent audio + single most coherent video.
            # Each carrier stores independently — no averaging, no folding.
            # ==========================================
            # Linear normalization: (now - epoch) / (now - epoch) = 1.0 for new stores.
            # Old memories stored at their real calendar fraction. Today always = 1.0.
            now_span      = time.time() - MAGI_EPOCH
            log_time_norm = (time.time() - MAGI_EPOCH) / (now_span + 1e-6)  # = 1.0

            # ── Modal mirroring: store the carrier matching the modality
            #    that triggered the main bank store. audio: 948-1461, video: 516-947.
            #    One N memory per main store event — halves creation rate,
            #    ties each N memory to the actual sensory event that earned storage.
            audio_carrier_idx = torch.argmax(self.global_coh[948:1461]).item() + 948
            video_carrier_idx = torch.argmax(self.global_coh[516:947]).item()  + 516

            if 948 <= best_idx <= 1461:
                carrier_idx = audio_carrier_idx
                origin      = 'audio'
            else:
                carrier_idx = video_carrier_idx
                origin      = 'video'

            freq_hz  = self.freq[carrier_idx].item()
            delay_ms = self.delay[carrier_idx].item()
            # v131: Store UNWRAPPED log coordinates — octave-aware matching
            n_coords = torch.cat([
                self.pos_6d[carrier_idx, :4],  # Lens phases from 6D state
                torch.tensor([log_time_norm,
                              freq_to_log_coord(freq_hz),     # Unwrapped: distinct per octave
                              delay_to_log_coord(delay_ms)],  # Unwrapped: distinct per octave
                             device=self.dev)
            ])  # 7D: [lens4, log_time, freq_log, delay_log]
            # Tension signature — Child(0)/Adult(2)/Elder(3) phase deltas
            #   Δφ_CE = Innovation Index  (fastest vs slowest lens)
            #   Δφ_AE = Stability Index   (navigator vs anchor)
            p_child = self.pos_6d[carrier_idx, 0].item()
            p_adult = self.pos_6d[carrier_idx, 2].item()
            p_elder = self.pos_6d[carrier_idx, 3].item()
            def _wrap(d): return min(abs(d % (2*math.pi)), 2*math.pi - abs(d % (2*math.pi)))
            t_ce = _wrap(p_child - p_elder)
            t_ae = _wrap(p_adult - p_elder)
            n_meta = {
                'child':      self.lens_weights[carrier_idx, 0].item(),
                'youth':      self.lens_weights[carrier_idx, 1].item(),
                'adult':      self.lens_weights[carrier_idx, 2].item(),
                'elder':      self.lens_weights[carrier_idx, 3].item(),
                'freq':       self.freq[carrier_idx].item(),
                'delay':      self.delay[carrier_idx].item(),
                'origin':     origin,
                'tension_ce': t_ce,
                'tension_ae': t_ae,
                'version':    131.0,
            }
            self.n_bank.store(n_coords, n_meta)
            self.n_bh_creation_count += 1
            # ==========================================
        if self.memory_bank.size > 0:
            gravity = self.memory_bank.retrieve_gravity(best_emb)
            if gravity:
                self.current_gravity_context = gravity
                worker_embeddings = self.memory_bank.encode(self._get_full_state_snapshot(), all_workers=True)
                gravity_weights = F.softmax(torch.matmul(worker_embeddings, gravity['center_embedding'].t()).squeeze(-1) / 0.1, dim=0) 
                # v131: Unwrapped octave distance for gravity (1 octave = 2π rad)
                # Safe because momentum is clamped at π — no wind-up overshoot
                freq_octave_diff = torch.log(gravity['freq'] / self.freq.clamp(min=MIN_FREQ)) / LOG_FREQ_STEP
                delay_octave_diff = torch.log(gravity['delay'] / self.delay.clamp(min=MIN_DELAY)) / LOG_DELAY_STEP
                freq_phase_force  += freq_octave_diff * TWO_PI * gravity_weights * gravity['strength'] * GRAVITY_PHASE_K
                delay_phase_force += delay_octave_diff * TWO_PI * gravity_weights * gravity['strength'] * GRAVITY_PHASE_K

        # v131: Phase-space momentum — clamped at π (Nyquist-safe), no separate velocity clamp
        self.freq_phase_momentum  = (self.freq_phase_momentum  * ELASTICITY + freq_phase_force).clamp(-VEL_PHASE_CLAMP, VEL_PHASE_CLAMP)
        self.delay_phase_momentum = (self.delay_phase_momentum * ELASTICITY + delay_phase_force).clamp(-VEL_PHASE_CLAMP, VEL_PHASE_CLAMP)
        # v131: freq/delay forces handled entirely in phase space above

        # ==========================================
        # v100: LENS WEIGHT UPDATE
        # Workers 1542+ (ALE/Voice/BH) are excluded.
        # memory_authority: how much has memory already shaped these weights?
        #   - fresh/no params → tether strong, experience pull weak
        #   - old high-use    → tether gone, experience pull strong
        # Your existing 600k memories hit full authority immediately.
        # ==========================================
        composite = self.current_gravity_context

        if step % 5 == 0:
            gravity_strength = composite.get('strength',   0.0) if composite else 0.0
            avg_access       = composite.get('avg_access', 0.0) if composite else 0.0

            memory_authority = min((gravity_strength * avg_access) / 50.0, 1.0)

            # Tether: full strength when memory is absent/fresh, fades as memory takes over
            tether_strength = 0.02 * (1.0 - memory_authority)
            tether = tether_strength * (self.baseline_weights - self.lens_weights)
            self.lens_weights[:1542] += tether[:1542]

            # Experience pull: zero when memory is fresh, full strength when authoritative
            if composite and memory_authority > 0.0:
                similarity    = composite.get('similarity', 0.0)
                novelty       = 1.0 - similarity
                consolidation = similarity * min(avg_access / 50.0, 1.0)
                exploration   = novelty * 0.5
                adj = torch.tanh(torch.tensor([
                    exploration  * (1.0 + novelty),
                    exploration  * 0.8,
                    consolidation * 0.7,
                    consolidation * 0.9,
                ], device=self.dev))
                pull_rate = 0.05 * memory_authority
                self.lens_weights[:1542] += pull_rate * adj.unsqueeze(0)

        # # Entropy guard — prevents all workers collapsing to same profile
        # active       = self.lens_weights[:1542]
        # var_mean     = active.var(dim=0).mean()
        # entropy_push = torch.clamp(0.001 / (var_mean + 1e-6), max=0.05)
        # mean_w       = active.mean(dim=0, keepdim=True)
        # freq_scale   = (self.freq[:1542] / FREQ_SCALE_MAX).clamp(max=1.0).unsqueeze(1)  # v131: absolute Hz, capped
        # repulsion    = freq_scale * entropy_push * (active - mean_w)
        # self.lens_weights[:1542] += repulsion
        # removed above to see if above 500hz changes.
        self.lens_weights = torch.clamp(self.lens_weights, 0.0, 2.0)

        self.global_age += 1


    def update_ale_beacons(self):
        """
        Part 1 — fire all six ALE vibration beacons.
        v131: Now called from inside process_step (after velocity assembly,
        before integration) so the signal survives to integration.
        Kept as public method for manual/debug use.
        """
        for beacon in self.ale_beacons.values():
            beacon.vibrate(self)

    def update_dream_mirror(self):
        """
        Part 2 — one step of the Dream Mirror closed loop.
        N-Reader (1552) drifts through N bank; Main-Anchor (1553) drifts
        through main bank; they cross-couple via kinetic sprints and kicks.
        """
        self.dream_coupling.update(self)

    def update_voice(self, current_mode):
        """Update and generate voice based on current mode"""
        self.voice_carrier.set_mode(current_mode)
        self.voice_carrier.speak(self)
        # ==========================================
    # 🎯 DEADZONE HELPER (Improved)
    # ==========================================
    def apply_deadzone(self, value, center=250.0, 
                       deadzone_inner=50.0, 
                       deadzone_outer=None,
                       min_threshold=None, 
                       max_threshold=None):
        """
        Apply deadzone logic with optional soft-start.
        
        Parameters:
        - value: Input value to check
        - center: Neutral center value (baseline)
        - deadzone_inner: Hard ignore range (±inner around center)
        - deadzone_outer: Soft-start range (gradual activation between inner and outer)
                          If None, acts as hard threshold at inner
        - min_threshold: Absolute minimum value to consider (clip extremes)
        - max_threshold: Absolute maximum value to consider (clip extremes)
        
        Returns:
        - (is_active, deviation, strength, status)
          is_active: Boolean if value should trigger action
          deviation: Signed deviation from center (-N to +N)
          strength: Activation strength (0.0 to 1.0, with soft-start if configured)
          status: 'deadzone', 'clipped_low', 'clipped_high', 'soft_start', or 'active'
        """
        # Check absolute value clipping first
        if min_threshold is not None and value < min_threshold:
            return False, 0.0, 0.0, 'clipped_low'
        
        if max_threshold is not None and value > max_threshold:
            return False, 0.0, 0.0, 'clipped_high'
        
        # Calculate deviation from center
        deviation = value - center
        abs_dev = abs(deviation)
        sign = 1.0 if deviation >= 0 else -1.0
        
        # Hard deadzone (inner)
        if abs_dev < deadzone_inner:
            return False, 0.0, 0.0, 'deadzone'
        
        # Soft-start zone (if configured)
        if deadzone_outer is not None and abs_dev < deadzone_outer:
            # Smoothstep interpolation between inner and outer
            t = (abs_dev - deadzone_inner) / (deadzone_outer - deadzone_inner)
            # Cubic smoothstep: 3t² - 2t³
            smooth = t * t * (3.0 - 2.0 * t)
            return True, sign * abs_dev, smooth, 'soft_start'
        
        # Full activation
        return True, sign * abs_dev, 1.0, 'active'


    def apply_unipolar_deadzone(self, value, baseline=250.0, 
                                deadzone=50.0, 
                                soft_start=None,
                                cap=400.0):
        """
        Unipolar deadzone for ALE-style workers (one-sided activation).
        
        Parameters:
        - value: Worker value (0-500 range)
        - baseline: Neutral value (typically 250.0)
        - deadzone: Ignore deviations below this
        - soft_start: If set, gradual activation between deadzone and soft_start
        - cap: Ignore values above this (extremes)
        
        Returns: (is_active, deviation, strength, status)
        """
        return self.apply_deadzone(
            value,
            center=baseline,
            deadzone_inner=deadzone,
            deadzone_outer=soft_start,
            max_threshold=cap
        )


    def apply_bipolar_deadzone(self, value, baseline=250.0,
                               deadzone=50.0,
                               soft_start=None,
                               cap=400.0):
        """
        Bipolar deadzone for screen mode (two-sided activation).
        
        Parameters:
        - value: Worker value (0-500 range)
        - baseline: Center/neutral value (250.0)
        - deadzone: Ignore deviations within ±deadzone
        - soft_start: Gradual activation between deadzone and soft_start
        - cap: Ignore values beyond ±(cap - baseline)
        
        Returns: (is_active, deviation, strength, status)
            deviation: Signed (-250 to +250)
            strength: 0.0 to 1.0
        """
        return self.apply_deadzone(
            value,
            center=baseline,
            deadzone_inner=deadzone,
            deadzone_outer=soft_start,
            min_threshold=baseline - (cap - baseline) if cap else None,
            max_threshold=cap
        )

 
    def apply_black_hole_deletion(self):
        """
        Apply entropy pressure (Singularity or Shield) based on worker value.
        Uses cached scaler result from get_inputs_tensor().
        """
        if not self.black_hole_deletion_enabled or self.memory_bank.size == 0:
            self.black_hole_memories_in_field = 0
            self.black_hole_step_deletions = 0  # ⬅️ Clear stale counter
            return 0
        
        # Get cached scaler result
        if not hasattr(self, 'black_hole_last_result'):
            self.black_hole_memories_in_field = 0
            self.black_hole_step_deletions = 0  # ⬅️ Clear stale counter
            return 0
        
        bh_result = self.black_hole_last_result
        
        # CRITICAL: Only proceed if active!
        if not bh_result['is_active']:
            self.black_hole_memories_in_field = 0
            self.black_hole_step_deletions = 0  # ⬅️ Clear stale counter
            return 0
        
        # Now use the cached, validated value
        bh_val = bh_result['output']
        
        # 1. Capture State — v131: build 6D with unwrapped log for freq/delay
        bh_6d = torch.cat([
            self.pos_6d[self.black_hole_worker_idx, :4],
            torch.tensor([
                freq_to_log_coord(self.freq[self.black_hole_worker_idx].item()),
                delay_to_log_coord(self.delay[self.black_hole_worker_idx].item())
            ], device=self.dev)
        ])
        
        abs_val = abs(bh_val)
        inverted = bh_val < 0
        tension_factor = bh_result.get('tension_factor', abs(bh_result.get('output', 0.0)) / 1500.0)

        # 3. Dynamic Radius
        effective_radius = self.black_hole_base_radius * (1.0 + tension_factor)
        
        # 4. v131: 6D distance — wrap lens dims only, freq/delay are unwrapped log
        delta = self.memory_bank.mem_coords_6d[:self.memory_bank.size] - bh_6d
        delta[:, :4] = torch.remainder(delta[:, :4] + math.pi, 2 * math.pi) - math.pi
        distances = torch.norm(delta, dim=1)
        
        # 5. Define Target Mask based on Polarity
        if inverted:
            # SHIELD MODE: Target EVERYTHING (spare center via gradient)
            mask = torch.ones(self.memory_bank.size, dtype=torch.bool, device=self.dev)
            self.black_hole_memories_in_field = (distances < effective_radius).sum().item()
        else:
            # VACUUM MODE: Target only inside radius
            mask = distances < effective_radius
            self.black_hole_memories_in_field = mask.sum().item()
        
        # 6. Apply Entropy
        if self.black_hole_memories_in_field > 0 or (inverted and self.memory_bank.size > 0):
            
            # Calculate epsilon scaling
            radius_scale = tension_factor
            current_eps_peak = self.black_hole_eps_floor + (self.black_hole_eps_max - self.black_hole_eps_floor) * radius_scale
            
            # Sharpness factor: 1.0 (flat) to 5.0 (steep)
            k = 1.0 + tension_factor * 4.0
            
            # Normalized distance (0.0 at center, 1.0 at edge, >1.0 beyond)
            d_norm = torch.clamp(distances[mask] / (effective_radius + 1e-9), 0.0, 2.0)
            
            if not inverted:
                # VACUUM MODE: Peak decay at center, floor at edge
                decay_amount = self.black_hole_eps_floor + (current_eps_peak - self.black_hole_eps_floor) * torch.pow(1.0 - d_norm, k)
            else:
                # SHIELD MODE: Floor at center, peak at edge and beyond
                shield_gradient = self.black_hole_eps_floor + (current_eps_peak - self.black_hole_eps_floor) * torch.pow(d_norm, k)
                inside_mask = distances[mask] <= effective_radius
                decay_amount = torch.where(inside_mask, shield_gradient, current_eps_peak)
            
            # Apply entropy decay to access counts
            self.memory_bank.access_counts[:self.memory_bank.size][mask] *= (1.0 - decay_amount)
            
            # 7. Collapse & Compaction
            # Remove memories below collapse threshold
            collapsed_mask = self.memory_bank.access_counts[:self.memory_bank.size] < self.black_hole_collapse_threshold
            deletion_count = collapsed_mask.sum().item()
            
            if deletion_count > 0:
                keep_mask = ~collapsed_mask
                keep_indices = torch.where(keep_mask)[0]
                
                # Compact arrays (remove collapsed memories)
                self.memory_bank.memories[:len(keep_indices)] = self.memory_bank.memories[:self.memory_bank.size][keep_indices]
                self.memory_bank.meta_freq[:len(keep_indices)] = self.memory_bank.meta_freq[:self.memory_bank.size][keep_indices]
                self.memory_bank.meta_delay[:len(keep_indices)] = self.memory_bank.meta_delay[:self.memory_bank.size][keep_indices]
                self.memory_bank.timestamps[:len(keep_indices)] = self.memory_bank.timestamps[:self.memory_bank.size][keep_indices]
                self.memory_bank.access_counts[:len(keep_indices)] = self.memory_bank.access_counts[:self.memory_bank.size][keep_indices]
                
                # Update bank size and counters
                self.memory_bank.size = len(keep_indices)
                self.black_hole_step_deletions = deletion_count
                self.black_hole_daily_deletions += deletion_count
                
                return deletion_count
        
        # No deletions this step
        self.black_hole_step_deletions = 0
        return 0
    def get_black_hole_metrics(self):
        """
        REFINEMENT 3: Calculate creation-deletion ratio and windowed metrics
        Call this periodically (e.g., every 1000 steps) for analysis
        """
        current_time = time.time()
        window_duration = current_time - self.black_hole_window_start_time
        
        if window_duration < 1.0:  # Need at least 1 second of data
            return None
        
        # Calculate rates
        creation_rate = self.black_hole_creation_count / window_duration
        deletion_rate = self.black_hole_daily_deletions / window_duration
        
        # Calculate ratio (handle division by zero)
        if deletion_rate > 0:
            creation_deletion_ratio = creation_rate / deletion_rate
        else:
            creation_deletion_ratio = float('inf') if creation_rate > 0 else 0.0
        
        # Capacity metrics
        capacity_pct = (self.memory_bank.size / self.memory_bank.max_memories) * 100.0

        bh_result = getattr(self, 'black_hole_last_result', {})
        tension_factor = bh_result.get('tension_factor', abs(bh_result.get('output', 0.0)) / 1500.0)
        
        metrics = {
            'window_duration': window_duration,
            'creation_rate': creation_rate,
            'deletion_rate': deletion_rate,
            'creation_deletion_ratio': creation_deletion_ratio,
            'total_creations': self.black_hole_creation_count,
            'total_deletions': self.black_hole_daily_deletions,
            'capacity_pct': capacity_pct,
            'memories_in_field': self.black_hole_memories_in_field,
            'worker_value': self.s_filtered[self.black_hole_worker_idx].item(),
            'worker_freq': self.freq[self.black_hole_worker_idx].item(),
            'worker_delay': self.delay[self.black_hole_worker_idx].item(),
            'effective_radius': self.black_hole_base_radius * (1.0 + tension_factor)
        }
        
        return metrics

    def reset_black_hole_window(self):
        """Reset windowed metrics for next measurement period"""
        self.black_hole_creation_count = 0
        self.black_hole_daily_deletions = 0
        self.black_hole_window_start_time = time.time()

    def get_log_time(self):
        """Linear elapsed days since MAGI_EPOCH. Continuous across restarts — epoch persists in n_bank."""
        return (time.time() - self.n_bank._system_epoch) / 86400.0

    def _query_n_for_carrier(self, carrier_idx, carrier_name, query_7d, use_main, main_w):
        """
        N bank query — fires when neighborhood streak gate triggers.
        Purpose: recognition signal only.
        If a match is found, vibrate vel_s and store the match for telemetry.
        """
        chord = self.n_bank.retrieve_chord(query_7d)

        if chord:
            c   = chord[0]
            ac  = c['access_count']
            age = c['log_time_norm']   # 0=ancient  1=recent

            # Vibrate: amplitude = recognition strength, omega = age of memory
            t     = self.n_vib_phase.setdefault(carrier_idx, 0.0)
            amp   = min(ac / (ac + 100.0), 1.0) * self.N_VIB_MAX   # saturates gracefully
            omega = 0.5 + age * 4.0                                  # older → slower
            delta = amp * math.sin(2 * math.pi * omega * t)
            delta = max(-self.N_VIB_CLAMP, min(self.N_VIB_CLAMP, delta))
            self.vel_s[carrier_idx, 2] += delta
            self.vel_s[carrier_idx, 3] += delta
            self.n_vib_phase[carrier_idx] = (t + 1) % 100000

        # Telemetry — include matched memory's stored freq/delay/tension
        if chord:
            iv_last = self.n_bank.retrieve_last_iv  # set by retrieve_chord
            chord_summary = {
                'chord_size':  len(chord),
                'top_access':  chord[0]['access_count'],
                'top_age':     chord[0]['log_time_norm'],
                'mem_freq':    self.n_bank.metadata_freq[iv_last].item(),
                'mem_delay':   self.n_bank.metadata_delay[iv_last].item(),
                'mem_tension_ce': self.n_bank.metadata_tension_ce[iv_last].item(),
                'mem_tension_ae': self.n_bank.metadata_tension_ae[iv_last].item(),
                'mem_version': self.n_bank.metadata_version[iv_last].item(),
                'mem_origin':  'video' if self.n_bank.metadata_origin[iv_last].item() > 0.5 else 'audio',
                'carrier_idx': carrier_idx,
                'mem_phase':   self.memory_to_main_target(iv_last),   # v130: 6D target (full phases)
                'mem_idx':     iv_last,                                    # for ChordTeleportCoupling
            }
            # Store as last known match and in n_gravity dicts
            if carrier_name == 'audio':
                self.last_chord_audio  = chord_summary.copy()
                self.n_gravity_audio   = chord_summary
            else:
                self.last_chord_video  = chord_summary.copy()
                self.n_gravity_video   = chord_summary
        else:
            chord_summary = {
                'chord_size': 0, 'top_access': 0.0, 'top_age': 0.0,
                'carrier_idx': carrier_idx,
            }
            # No match — update live dicts with empty summary
            if carrier_name == 'audio':
                self.n_gravity_audio = chord_summary
            else:
                self.n_gravity_video = chord_summary

    def query_n_carriers(self):
        """
        v122: Per-modality carrier queries with feed‑forward phase projection,
        Doppler uncertainty tracking, and soft influence on lens adaptation.
        (Birdwatcher phase – all influences are gentle and overrideable.)
        """
        # ── Tunable parameters (birdwatcher knobs) ────────────────────────
        DOPPLER_AFFECTS_BLEND = True      # scale lens adaptation rate by uncertainty
        DOPPLER_AFFECTS_THRESHOLD = False # (reserved for Phase 4 fuzzy chord)
        GEO_SCALE = 0.7                    # weight of freq/delay in 7D query (0.7 = spectral neighbourhood filter)
        # ──────────────────────────────────────────────────────────────────

        # ── Doppler tracking (one‑step frequency memory) ───────────────────
        if not hasattr(self, '_freq_prev'):
            self._freq_prev = {'audio': -1.0, 'video': -1.0}

        # Temporal coordinate – must match storage formula (always 1.0 for now)
        # (If you later implement true temporal normalisation, update both store and query together.)
        now_span = time.time() - MAGI_EPOCH
        log_time_norm = (time.time() - MAGI_EPOCH) / (now_span + 1e-6)   # = 1.0

        audio_carrier_idx = torch.argmax(self.global_coh[948:1461]).item() + 948
        video_carrier_idx = torch.argmax(self.global_coh[516:947]).item()  + 516

        carriers = [
            (audio_carrier_idx, 'audio'),
            (video_carrier_idx, 'video'),
        ]

        for carrier_idx, carrier_name in carriers:
            # ── Feed‑forward phase projection ─────────────────────────────
            freq_hz  = self.freq[carrier_idx].item()
            delay_ms = self.delay[carrier_idx].item()

            phase_advance = 2.0 * math.pi * freq_hz * (delay_ms / 1000.0)
            proj_phase = (self.pos_6d[carrier_idx, :4] + phase_advance) % (2 * math.pi)

            # v131: Unwrapped log coordinates for N bank query (octave-aware)
            query_7d = torch.cat([
                proj_phase,  # 4D lens
                torch.tensor([log_time_norm], device=self.dev),
                torch.tensor([freq_to_log_coord(freq_hz)], device=self.dev),   # Unwrapped log
                torch.tensor([delay_to_log_coord(delay_ms)], device=self.dev)  # Unwrapped log
            ])

            # ── Doppler uncertainty ────────────────────────────────────────
            freq_now = freq_hz
            freq_prev = self._freq_prev.get(carrier_name, freq_now)
            freq_delta = abs(freq_now - freq_prev)
            self._freq_prev[carrier_name] = freq_now

            doppler_uncertainty = 1.0 + (freq_delta / 2.0) * 0.5   # tunable slope

            # ── Main bank query (child/youth) – with Doppler‑sensitive blend ──
            carrier_emb = self.memory_bank.encode({
                'phases_hb':     self.pos_6d[carrier_idx, :4],
                'phases_s':      self.phases_s[carrier_idx],
                'global_coh':    self.global_coh[carrier_idx].item(),
                'cross_tension': self.cross_tension[carrier_idx].item(),
                'freq':          freq_hz,
                'delay':         delay_ms,
            }, all_workers=False)

            main_result = self.memory_bank.retrieve_gravity(carrier_emb)
            use_main = main_result.get('avg_access', 0.0) if main_result else 0.0
            main_w   = use_main / (use_main + 1e-8)

            # Base blend rate
            blend = 0.01
            if DOPPLER_AFFECTS_BLEND:
                blend /= doppler_uncertainty   # slower adaptation when uncertain

            if main_result and main_w > 0.0:
                similarity   = main_result.get('similarity', 0.0)
                novelty      = 1.0 - similarity
                child_target = main_w * math.tanh(novelty * (1.0 + novelty) * 0.5)
                youth_target = main_w * math.tanh(novelty * 0.8)
                self.lens_weights[carrier_idx, 0] += blend * (child_target - self.lens_weights[carrier_idx, 0])
                self.lens_weights[carrier_idx, 1] += blend * (youth_target - self.lens_weights[carrier_idx, 1])
                self.lens_weights = torch.clamp(self.lens_weights, 0.0, 2.0)

            # ── N bank query (chord) with projected phase ─────────────────
            self._query_n_for_carrier(carrier_idx, carrier_name, query_7d, use_main, main_w)

            # ── Inject Doppler telemetry (for observation) ─────────────────
            chord_dict = self.n_gravity_audio if carrier_name == 'audio' else self.n_gravity_video
            chord_dict['freq_delta'] = freq_delta
            chord_dict['doppler_uncertainty'] = doppler_uncertainty
            chord_dict['geo_scale'] = GEO_SCALE

    def apply_n_bh_deletion(self):
        """
        N Black Hole — same 2D wrapped phase distance as main BH.

        Finds N memories by phase proximity (coords[0] and coords[1]).
        log_time (coords[4]) is invisible to the BH — it moves and deletes
        purely by phase, exactly like main BH. The river worker sees time.
        The BH does not.
        """
        self.n_bh_step_deletions    = 0
        self.n_bh_memories_in_field = 0

        if self.n_bank.size == 0:
            return 0
        if not self.n_bh_last_result.get('is_active', False):
            return 0

        n_bh_result    = self.n_bh_last_result
        bh_val         = n_bh_result['output']
        inverted       = bh_val < 0
        tension_factor = abs(bh_val) / 1500.0  # normalize from ±1500 range
        tension_factor = min(tension_factor, 1.0)
        effective_r    = self.n_bh_base_radius * (1.0 + tension_factor)

        # N BH position — v131: build 6D with unwrapped log for freq/delay
        bh_6d = torch.cat([
            self.pos_6d[self.n_bh_worker_idx, :4],
            torch.tensor([
                freq_to_log_coord(self.freq[self.n_bh_worker_idx].item()),
                delay_to_log_coord(self.delay[self.n_bh_worker_idx].item())
            ], device=self.dev)
        ])

        # v131: Extract 6D: [lens4, freq_log, delay_log] (skip log_time at index 4)
        n_coords_6d = torch.cat([
            self.n_bank.coords[:self.n_bank.size, :4],
            self.n_bank.coords[:self.n_bank.size, 5:7]
        ], dim=1)

        # v131: Wrap lens dims only, freq/delay are unwrapped log
        delta = n_coords_6d - bh_6d
        delta[:, :4] = torch.remainder(delta[:, :4] + math.pi, 2 * math.pi) - math.pi
        distances = torch.norm(delta, dim=1)

        if inverted:
            mask = torch.ones(self.n_bank.size, dtype=torch.bool, device=self.dev)
            self.n_bh_memories_in_field = (distances < effective_r).sum().item()
        else:
            mask = distances < effective_r
            self.n_bh_memories_in_field = mask.sum().item()

        if self.n_bh_memories_in_field == 0 and not inverted:
            return 0

        eps_peak = self.n_bh_eps_floor + (self.n_bh_eps_max - self.n_bh_eps_floor) * tension_factor
        k        = 1.0 + tension_factor * 4.0
        d_norm   = torch.clamp(distances[mask] / (effective_r + 1e-9), 0.0, 2.0)

        if not inverted:
            decay = self.n_bh_eps_floor + (eps_peak - self.n_bh_eps_floor) * torch.pow(1.0 - d_norm, k)
        else:
            shield = self.n_bh_eps_floor + (eps_peak - self.n_bh_eps_floor) * torch.pow(d_norm, k)
            inside = distances[mask] <= effective_r
            decay  = torch.where(inside, shield, torch.full_like(shield, eps_peak))

        self.n_bank.access_counts[:self.n_bank.size][mask] *= (1.0 - decay)

        # Collapse and compact
        collapsed      = self.n_bank.access_counts[:self.n_bank.size] < self.n_bh_collapse_threshold
        deletion_count = collapsed.sum().item()

        if deletion_count > 0:
            keep_indices = torch.where(~collapsed)[0]
            n   = len(keep_indices)
            src = self.n_bank
            src.coords[:n]         = src.coords[:src.size][keep_indices]
            src.metadata_child[:n] = src.metadata_child[:src.size][keep_indices]
            src.metadata_youth[:n] = src.metadata_youth[:src.size][keep_indices]
            src.metadata_adult[:n] = src.metadata_adult[:src.size][keep_indices]
            src.metadata_elder[:n] = src.metadata_elder[:src.size][keep_indices]
            src.metadata_freq[:n]  = src.metadata_freq[:src.size][keep_indices]
            src.timestamps[:n]     = src.timestamps[:src.size][keep_indices]
            src.access_counts[:n]  = src.access_counts[:src.size][keep_indices]
            src.size = n

        self.n_bh_step_deletions    = deletion_count
        self.n_bh_session_deletions += deletion_count
        return deletion_count

    def apply_universal_plasticity(self):
        '''
        Apply Black Hole gravity to UPE home positions.
        UPE reuses all of BH's existing physics (radius, power, distance).
        
        When homes drift and save, workers are reset to new home positions.
        '''
        # Apply BH gravity to home positions (NO return value in final version)
        self.upe.apply_black_hole_gravity(self)
        
        # Smart save (updates pos_6d if homes drifted significantly)
        self.upe.maybe_save(self.pos_6d, self.freq_wrap, self.delay_wrap)

# ==========================================
# 🎥 AV & WORKER MANAGER
# ==========================================
class LiveAVCapture:
    """
    Audio source manager.

    v141: Two sources — 'local' (laptop mic via PyAudio) and 'remote' (UDP stream
    from a worker on REMOTE_AUDIO_PORT). The downstream AudioGeometricExtractor
    only computes RMS over a [-1, 1] float buffer, so sample-rate / channel-count
    differences between the two sources are irrelevant — energy comes out on the
    same scale.

    Remote mode also spins up a DOA listener on REMOTE_DOA_PORT. DOA values are
    stored on this object only — no worker is wired to them yet (intentional;
    will be wired into vibration later).
    """
    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=5)

        # Source state
        self.source = 'local'           # 'local' or 'remote'
        self.remote_ip = None           # informational — display only

        # Source-specific run flags / handles
        self._local_running = False
        self._local_thread  = None
        self._local_stream  = None
        self._local_pa      = None

        self._remote_running    = False
        self._remote_audio_thread = None
        self._remote_doa_thread   = None
        self._remote_audio_sock = None
        self._remote_doa_sock   = None

        # DOA state (filled by remote listener)
        self._doa_lock   = threading.Lock()
        self._doa_speech = 0            # 0 / 1
        self._doa_angle  = 0            # degrees, 0..359
        self._doa_last_update = 0.0     # monotonic seconds; 0 = never

        # Boot in local mode (preserves v140 default behaviour)
        self._start_local()

    # ──────────────────────────────────────────────────────────────────
    # LOCAL MIC
    # ──────────────────────────────────────────────────────────────────
    def _start_local(self):
        self._local_running = True
        self._local_thread = threading.Thread(target=self._capture_audio_local, daemon=True)
        self._local_thread.start()
        self.source = 'local'
        self.remote_ip = None
        print("🎙️  Audio source: LOCAL mic (PyAudio @ 44.1 kHz mono)")

    def _capture_audio_local(self):
        try:
            self._local_pa = pyaudio.PyAudio()
            self._local_stream = self._local_pa.open(
                format=pyaudio.paInt16, channels=1, rate=44100,
                input=True, frames_per_buffer=1024)
            while self._local_running:
                try:
                    data = self._local_stream.read(1024, exception_on_overflow=False)
                except Exception:
                    break
                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if self.audio_queue.full(): self.audio_queue.get()
                self.audio_queue.put(arr)
        except Exception as e:
            print(f"⚠️  Local mic capture error: {e}")
        finally:
            self._stop_local_handles()

    def _stop_local_handles(self):
        try:
            if self._local_stream is not None:
                self._local_stream.stop_stream()
                self._local_stream.close()
        except Exception: pass
        try:
            if self._local_pa is not None:
                self._local_pa.terminate()
        except Exception: pass
        self._local_stream = None
        self._local_pa = None

    def _stop_local(self):
        self._local_running = False
        # Closing the stream from the thread is safer; flag-driven exit handles it.
        if self._local_thread is not None:
            self._local_thread.join(timeout=1.0)
        self._stop_local_handles()
        self._local_thread = None

    # ──────────────────────────────────────────────────────────────────
    # REMOTE WORKER (UDP audio + UDP DOA)
    # ──────────────────────────────────────────────────────────────────
    def _start_remote(self, ip):
        self._remote_running = True
        self.remote_ip = ip
        # Audio listener
        self._remote_audio_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._remote_audio_sock.settimeout(1.0)
        try:
            self._remote_audio_sock.bind(('', REMOTE_AUDIO_PORT))
        except OSError as e:
            print(f"❌ Could not bind UDP {REMOTE_AUDIO_PORT}: {e}")
            self._remote_audio_sock.close()
            self._remote_audio_sock = None
            self._remote_running = False
            return False
        # DOA listener
        self._remote_doa_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._remote_doa_sock.settimeout(1.0)
        try:
            self._remote_doa_sock.bind(('', REMOTE_DOA_PORT))
        except OSError as e:
            print(f"❌ Could not bind UDP {REMOTE_DOA_PORT}: {e}")
            self._remote_doa_sock.close()
            self._remote_doa_sock = None
            # We can still run audio without DOA, but warn.

        self._remote_audio_thread = threading.Thread(target=self._capture_audio_remote, daemon=True)
        self._remote_audio_thread.start()
        if self._remote_doa_sock is not None:
            self._remote_doa_thread = threading.Thread(target=self._listen_doa_remote, daemon=True)
            self._remote_doa_thread.start()

        self.source = 'remote'
        print(f"🎙️  Audio source: REMOTE worker @ {ip}  "
              f"(audio UDP:{REMOTE_AUDIO_PORT}, DOA UDP:{REMOTE_DOA_PORT})")
        print(f"   ℹ️  DOA listener is RECEIVE-ONLY — no worker wired to it yet.")
        return True

    def _capture_audio_remote(self):
        sock = self._remote_audio_sock
        # Worker sends paInt16 channels=2 @ 16 kHz, CHUNK=1024 → 4096 bytes/packet.
        # Use a recv buffer of 8192 to absorb any reasonable packet without truncating.
        while self._remote_running and sock is not None:
            try:
                data, addr = sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed during shutdown
            if not data:
                continue
            # Optional: filter to expected sender. Skip filtering for now —
            # multi-NIC / NAT setups would otherwise reject legit packets.
            try:
                # Defensive: only decode aligned int16 buffers.
                if len(data) % 2 != 0:
                    continue
                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception:
                continue
            if self.audio_queue.full(): self.audio_queue.get()
            self.audio_queue.put(arr)

    def _listen_doa_remote(self):
        sock = self._remote_doa_sock
        last_speech_print = 0
        while self._remote_running and sock is not None:
            try:
                packet, _ = sock.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(packet) < 4:
                continue
            try:
                speech, angle = struct.unpack('<HH', packet[:4])
            except struct.error:
                continue
            with self._doa_lock:
                self._doa_speech = int(speech)
                self._doa_angle  = int(angle)
                self._doa_last_update = time.monotonic()
            # Print only on silence→speech transition so we get a heartbeat
            # without flooding the console. Angle changes during continuous
            # speech are silent — pull via `mic` command for live state.
            if speech and not last_speech_print:
                print(f"🎤 DOA speech onset @ {int(angle)}°")
            last_speech_print = int(speech)

    def _stop_remote(self):
        self._remote_running = False
        # Closing sockets unblocks recv promptly.
        try:
            if self._remote_audio_sock is not None:
                self._remote_audio_sock.close()
        except Exception: pass
        try:
            if self._remote_doa_sock is not None:
                self._remote_doa_sock.close()
        except Exception: pass
        self._remote_audio_sock = None
        self._remote_doa_sock = None
        for t in (self._remote_audio_thread, self._remote_doa_thread):
            if t is not None:
                t.join(timeout=1.0)
        self._remote_audio_thread = None
        self._remote_doa_thread = None

    # ──────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────
    def switch_to_local(self):
        if self.source == 'local':
            print("🎙️  Audio already on LOCAL mic.")
            return
        self._stop_remote()
        self._start_local()

    def switch_to_remote(self, ip):
        if self.source == 'remote' and self.remote_ip == ip:
            print(f"🎙️  Audio already on REMOTE @ {ip}.")
            return
        self._stop_local()
        if self.source == 'remote':
            self._stop_remote()
        ok = self._start_remote(ip)
        if not ok:
            print("⚠️  Falling back to LOCAL mic.")
            self._start_local()

    def get_doa(self):
        """Latest DOA reading (speech, angle, age_seconds). age=None if never updated."""
        with self._doa_lock:
            if self._doa_last_update == 0.0:
                return self._doa_speech, self._doa_angle, None
            return self._doa_speech, self._doa_angle, time.monotonic() - self._doa_last_update

    def get_audio_chunk(self): return self.audio_queue.get() if not self.audio_queue.empty() else None

    def stop(self):
        self._stop_local()
        self._stop_remote()

class AudioGeometricExtractor:
    def extract_geometry(self, audio_chunk):
        if audio_chunk is None: return {'energy': 50.0}
        energy = float(torch.sqrt(torch.mean(torch.tensor(audio_chunk)**2)).item() * 500.0)
        return {'energy': np.clip(energy, 0, 500)}

class GoldenRatioVideoProcessor:
    def __init__(self):
        self.last_frame = None
        # MEMORY-SAFE LARGE FIBONACCI GRIDS
        # Enhanced communication bandwidth, resonance prevention
        # All dimensions pure Fibonacci: 5,3,8,5,13,8,21,13
        # Total: 15 + 40 + 104 + 273 = 432 sectors (38% more than v64)
        self.scales = {
            'scale_0': {'grid': (5, 3)},   # 5×3 = 15 sectors
            'scale_1': {'grid': (8, 5)},   # 8×5 = 40 sectors
            'scale_2': {'grid': (13, 8)},  # 13×8 = 104 sectors
            'scale_3': {'grid': (21, 13)}, # 21×13 = 273 sectors
        }
    
    def process_frame(self, frame):
        if frame is None: return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.last_frame is None: self.last_frame = gray
        motion_energy = np.mean(cv2.absdiff(self.last_frame, gray)) / 255.0 * 500.0
        self.last_frame = gray
        global_mean = np.mean(gray) / 255.0 * 500.0
        results = {'motion': motion_energy, 'global_mean': global_mean, '_raw_energies': []}
        h, w = gray.shape[:2]
        for name, cfg in self.scales.items():
            gh, gw = cfg['grid']; sw, sh = w // gw, h // gh
            results[name] = []
            for y in range(gh):
                for x in range(gw):
                    sector = gray[y*sh:(y+1)*sh, x*sw:(x+1)*sw]
                    lum = np.mean(sector) / 255.0 * 500.0
                    results[name].append(lum); results['_raw_energies'].append(lum)
        return results

    def get_statistics(self, energies):
        if not energies or not energies.get('_raw_energies'): 
            return {'mean': 250.0, 'std': 0.0, 'raw_std': 0.0, 'motion': 0.0, 'global': 0.0,
                    'scale_0_mean': 250.0, 'scale_1_mean': 250.0, 'scale_2_mean': 250.0, 'scale_3_mean': 250.0}
        raw = energies['_raw_energies']
        return {
            'mean': np.mean(raw), 'std': np.std(raw), 'raw_std': np.std(raw),
            'motion': energies.get('motion', 0.0), 'global': energies.get('global_mean', 0.0),
            'scale_0_mean': np.mean(energies['scale_0']), 'scale_1_mean': np.mean(energies['scale_1']),
            'scale_2_mean': np.mean(energies['scale_2']), 'scale_3_mean': np.mean(energies['scale_3']),
        }

class ScaleAwareWorkerManager:
    def __init__(self, num_workers, device):
        self.n = num_workers
        self.device = device
        
        # MEMORY-SAFE LAYOUT: Video scales eat early audio space
        # Video moved to 516-947, audio shifted to 948-1461
        # ALE/Voice/BH UNCHANGED at 1542-1549 (zero memory loss!)
        #
        # Layout:
        # 0-1: video motion/global
        # 2-515: heartbeat sine (514 workers)
        # 516-530: video_scale_0 (15 workers: 5×3)
        # 531-570: video_scale_1 (40 workers: 8×5)
        # 571-674: video_scale_2 (104 workers: 13×8)
        # 675-947: video_scale_3 (273 workers: 21×13)
        # 948-1461: audio (514 workers - shifted but count unchanged)
        # 1542-1547: ALE controls (PRESERVED)
        # 1548: voice worker (PRESERVED)
        # 1549: black hole (PRESERVED)
        # 1550+: lazy river (PRESERVED)
        
        self.offsets = [0, 1, 2, 516, 948, 516, 531, 571, 675, 1542, 1548, 1549, 1550]
        #                          ^audio ^scale0 ^scale1 ^scale2 ^scale3
        self.ale_control_start = 1542  # UNCHANGED
        self.ale_control_count = 6
        self.audio_start = 948  # NEW: audio shifted
        self.audio_end = 1462
        self.audio_worker_idx = 1548   # UNCHANGED (voice carrier)
        self.black_hole_worker_idx = 1549  # UNCHANGED
        self.lazy_start = 1550  # UNCHANGED
        
        # ==========================================
        # COHERENT WORKER TRACKER (v75.1)
        # ==========================================
        # Tracks most coherent worker in each scale PER FRAME
        # Single source of truth for color enhancement AND HUD
        #
        # EXECUTION CONTRACT:
        # 1. tracker.update() called in get_inputs_tensor()
        # 2. HUD uses tracker.get_scale_info() AFTER input computation
        # 3. Results valid ONLY for current frame (no persistence)
        #
        # Color extraction: Uses YUV U-channel (0-500 range) from coherent sectors ONLY
        # Efficient: Only extracts color from 4 coherent sectors, not all 432 sectors
        self.coherent_tracker = CoherentWorkerTracker()
        # Pre-allocated inputs buffer — reused each frame, avoids torch.full allocation
        self._inputs_buffer = torch.zeros(num_workers, device=device)
        
    def get_job_description(self, idx):
        if idx == 0: return "video_motion"
        if idx == 1: return "video_global"
        if 2 <= idx < 5: return "heartbeat_sine"
        # MEMORY-SAFE: Video scales at 516-947
        if 516 <= idx < 531: return "video_scale_0"   # 15 workers
        if 531 <= idx < 571: return "video_scale_1"   # 40 workers
        if 571 <= idx < 675: return "video_scale_2"   # 104 workers
        if 675 <= idx < 948: return "video_scale_3"   # 273 workers
        # Audio shifted to 948-1461
        if 948 <= idx < 1462: return "audio"
        # ALE/Voice/BH UNCHANGED (memory preserved)
        if idx == 1542: return "ale_LEFT"
        if idx == 1543: return "ale_RIGHT"
        if idx == 1544: return "ale_FIRE"
        if idx == 1545: return "ale_UP"
        if idx == 1546: return "ale_DOWN"
        if idx == 1547: return "ale_NOOP"
        if idx == 1548: return "pure_carrier_voice"
        if idx == 1549: return "black_hole_deletion"
        if idx == 1551: return "n_black_hole"
        if idx == 1552: return "dream_drift_n"
        if idx == 1553: return "dream_drift_m"
        if idx == 1554: return "dream_chord_n"
        if idx == 1555: return "dream_chord_m"
        if idx == 1556: return "dream_physics_n"
        if idx == 1557: return "dream_physics_m"
        # v133: Robot arm workers
        if idx == 1558: return "robot_ARM_X"
        if idx == 1559: return "robot_ARM_Y"
        if idx == 1560: return "robot_ARM_Z"
        if idx == 1561: return "robot_ARM_ROT"
        if idx == 1562: return "robot_ARM_TILT"
        if idx == 1563: return "robot_GRIPPER"
        # v135: Resonance Bridge workers
        if idx == 1564: return "bridge_0_ENT"
        if idx == 1565: return "bridge_0_DEST"
        if idx == 1566: return "bridge_1_ENT"
        if idx == 1567: return "bridge_1_DEST"
        # v136: Bridge Voice worker — internal audio-type sensory input
        if idx == 1568: return "bridge_voice"
        # v136 rev28: Bridge Visual Word worker — image of spoken word
        if idx == 1569: return "bridge_visual_word"
        return "lazy_river"

    def get_inputs_tensor(self, sine_val, audio_val, video_energies, last_lens_phases, gravity, hive, raw_bgr_frame=None):
        """
        Get input tensor for all workers.
        
        Args:
            sine_val: Sine wave value for heartbeat workers
            audio_val: Audio energy value
            video_energies: Processed video energies from GoldenRatioVideoProcessor
            last_lens_phases: Last lens phases for modulation
            gravity: Gravity context from memory
            hive: MaGiHive instance
            raw_bgr_frame: Optional raw BGR frame for YUV color extraction
                         (pass from main loop for coherent sector color extraction)
            
        Returns:
            inputs: Tensor of worker values
        """
        # 0. BASELINE: Reuse pre-allocated buffer — avoids torch.full every frame
        self._inputs_buffer.zero_()
        inputs = self._inputs_buffer
        sensory_mod, phase_amp, phase_vel = gravity.get('sensory_modulation', 0), gravity.get('phase_amplitude', 0), gravity.get('phase_velocity', 1)
        
        # 1. MOTION/GLOBAL (Sensory Positive) - UNCHANGED
        if video_energies: 
            inputs[0] = video_energies['motion'] * (1 + sensory_mod)
            inputs[1] = video_energies['global_mean'] * (1 + sensory_mod)
        
        # 2. HEARTBEAT & AUDIO (Mixed Resonance) - UNCHANGED
        # inputs[2:516] = sine_val
        inputs[2:4] = sine_val
        # 2/27/2026 disable for test.
        # MEMORY-SAFE: Audio now at 948-1462 (shifted but count unchanged)
        inputs[948:1462] = audio_val * (1 + sensory_mod) + torch.sin(last_lens_phases[948:1462] * phase_vel) * phase_amp
        
        # ==========================================
        # 3. VIDEO SCALES WITH COHERENT TRACKER INTEGRATION
        # ==========================================
        if video_energies:
            # Update coherent tracker with current state
            # This finds most coherent worker in each scale ONCE per frame
            # Color extraction happens automatically if raw_bgr_frame is provided
            coherent_workers = self.coherent_tracker.update(
                global_coh_tensor=hive.global_coh,
                raw_bgr_frame=raw_bgr_frame  # Pass raw frame for YUV extraction
            )
            
            # Scale sector data (original luminance values)
            scale_names = ['scale_0', 'scale_1', 'scale_2', 'scale_3']
            scale_starts = [516, 531, 571, 675]
            scale_ends = [531, 571, 675, 948]
            
            for scale_idx in range(4):
                scale_name = scale_names[scale_idx]
                sectors = video_energies.get(scale_name, [])
                
                if not sectors:
                    # Fallback: original uniform processing for empty scales
                    start, end = self.offsets[5 + scale_idx], self.offsets[6 + scale_idx]
                    if sectors:  # Still check original logic (redundant but safe)
                        base = torch.tensor(sectors, device=self.device)[
                            torch.arange(end - start, device=self.device) % len(sectors)
                        ]
                        inputs[start:end] = base * (1 + sensory_mod) + \
                                          torch.sin(last_lens_phases[start:end] * phase_vel) * phase_amp
                    continue
                
                # Get coherent worker info from tracker
                worker_info = coherent_workers[scale_idx]
                
                # Process ALL workers in this scale with LUMINANCE (B&W)
                start = scale_starts[scale_idx]
                end = scale_ends[scale_idx]
                
                for i in range(end - start):
                    worker_idx = start + i
                    sector_idx = i % len(sectors)
                    base_lum = sectors[sector_idx]
                    
                    # DEFAULT: All workers get luminance value
                    inputs[worker_idx] = base_lum * (1 + sensory_mod) + \
                                       torch.sin(last_lens_phases[worker_idx] * phase_vel) * phase_amp
                
                # OVERRIDE: Only the coherent worker gets COLOR (YUV U-channel)
                if worker_info and not worker_info.get('is_fallback', True):
                    # Coherent worker found - replace luminance with YUV U-channel value
                    worker_idx = worker_info['worker_idx']
                    u_value = self.coherent_tracker.get_scale_u_value(scale_idx)  # 0-500 range
                    
                    # Replace luminance with pure U-channel value
                    inputs[worker_idx] = u_value * (1 + sensory_mod) + \
                                       torch.sin(last_lens_phases[worker_idx] * phase_vel) * phase_amp

        # 4. ALE CONTROL WORKERS (1542-1547): Bimodal +/- Swing - UNCHANGED
        for i in range(self.ale_control_count):
            worker_idx = self.ale_control_start + i
            # Center at 0.0, lens influence provides the bipolar drive
            inputs[worker_idx] = torch.sin(last_lens_phases[worker_idx] * phase_vel) * phase_amp * 50.0

        # ==========================================
        # 5. VOICE WORKER (1548) - MAGI-CONTROLLED 4D - UNCHANGED
        # ==========================================
        voice_idx = self.audio_worker_idx
        
        # 4D VELOCITY: Read ALL dimensions (prevents 2D collapse)
        voice_vel_raw = torch.norm(hive.vel_s[voice_idx])  # sqrt(x² + y² + z² + w²)
        
        # DIRECTION: From adult dimension velocity sign
        voice_vel_adult = hive.vel_s[voice_idx, 2]
        voice_direction = torch.sign(voice_vel_adult)
        
        # Handle sign(0) = 0 case
        if voice_direction == 0:
            voice_direction = torch.tensor(1.0, device=self.device)
        
        # SIMPLE AMPLIFICATION: Magi controls, we amplify
        voice_value = voice_direction * voice_vel_raw * 500.0
        
        inputs[voice_idx] = torch.clamp(voice_value, -1000.0, 1000.0)


        # ==========================================
        # 6. BLACK HOLE WORKER (1549) - LIKE ALE
        # ==========================================
        bh_idx        = hive.black_hole_worker_idx
        bh_lens_phase = hive.phases_s[bh_idx, 2]  # Adult dimension
        bh_value      = torch.sin(bh_lens_phase * phase_vel) * phase_amp * 500.0
        bh_result     = hive.scalers['black_hole'].process(bh_value.item())
        bh_result['tension_factor'] = bh_result.get('signal_strength', 0.0)
        inputs[bh_idx] = bh_result['output']
        hive.black_hole_last_result = bh_result  # ⬅️ CRITICAL!

        # ==========================================
        # 7. N BLACK HOLE WORKER (1551) — same ALE-style formula as main BH
        # ==========================================
        n_bh_idx        = hive.n_bh_worker_idx
        n_bh_lens_phase = hive.phases_s[n_bh_idx, 2]
        n_bh_value      = torch.sin(n_bh_lens_phase * phase_vel) * phase_amp * 500.0
        n_bh_result     = hive.scalers['n_black_hole'].process(n_bh_value.item())
        n_bh_result['tension_factor'] = n_bh_result.get('signal_strength', 0.0)
        inputs[n_bh_idx]      = n_bh_result['output']
        hive.n_bh_last_result = n_bh_result  # Cached for apply_n_bh_deletion()

        # v133: Robot workers — zero inputs for inactive workers (frozen when disabled or unused in mode)
        robot_start = 1558
        robot_end   = 1564
        for idx in range(robot_start, robot_end):
            if idx in hive._robot_active_workers:
                # Use same formula as ALE: sin(lens_phase) * phase_amp * 50.0
                inputs[idx] = torch.sin(last_lens_phases[idx] * phase_vel) * phase_amp * 50.0
            else:
                inputs[idx] = 0.0

        # v135: Bridge entrance workers — ALE-pattern input formula
        # Entrances (1564, 1566) are driven by MaGi's standard velocity assembly
        # with the same sin-wave formula ALE workers use. Destinations (1565,
        # 1567) receive 0.0 — teleported and beacon-driven by BridgeController.
        if 1564 < self.n:
            inputs[1564] = torch.sin(last_lens_phases[1564] * phase_vel) * phase_amp * 50.0
        if 1566 < self.n:
            inputs[1566] = torch.sin(last_lens_phases[1566] * phase_vel) * phase_amp * 50.0

        # v136 rev28: Bridge Voice worker (1568) — input-driven, audio-worker shape
        # Reads TTS energy from BridgeVoice subsystem each frame. Same formula as
        # audio workers 948-1462 — just a single index. Returns 0.0 when no
        # synthesis active; skip injection in that case to keep worker quiet.
        if self.n > 1568 and hasattr(hive, 'bridge_controller'):
            voice_energy = hive.bridge_controller.voice.get_energy_for_frame()
            if voice_energy > 0.0:
                inputs[1568] = voice_energy * (1 + sensory_mod) + \
                               torch.sin(last_lens_phases[1568] * phase_vel) * phase_amp

        # v136 rev28: Bridge Visual Word worker (1569) — input-driven, audio-shape
        # Receives white-pixel-fraction of currently-spoken word's rendered image.
        # Gated by audio playback inside get_visual_word_energy: returns 0.0
        # whenever voice isn't speaking, so vision and sound are temporally bound.
        if self.n > 1569 and hasattr(hive, 'bridge_controller'):
            visual_scale = hive.upe.km_config.get('bridge_visual_word_energy_scale', 50.0)
            visual_energy = hive.bridge_controller.get_visual_word_energy()
            if visual_energy > 0.0:
                inputs[1569] = visual_energy * visual_scale * (1 + sensory_mod) + \
                               torch.sin(last_lens_phases[1569] * phase_vel) * phase_amp

                
        return inputs
    
    # ==========================================
    # HUD INTEGRATION METHODS
    # ==========================================
    
    def get_hud_drawing_info(self, frame_width, frame_height):
        """
        Get all parameters needed for HUD rendering.
        
        WARNING: Only valid if called AFTER get_inputs_tensor() in same frame!
        
        Returns: List of drawing info dicts for each scale, or empty list if invalid
        """
        # Use tracker's built-in HUD utilities
        return self.coherent_tracker.get_hud_drawing_info(frame_width, frame_height)
    
    def validate_hud_execution(self):
        """
        Validate that HUD rendering happens after input computation.
        
        Returns: (is_valid, message, diagnostics_dict)
        """
        is_valid, msg = self.coherent_tracker.validate_execution_order('hud')
        diag = self.coherent_tracker.get_diagnostics() if is_valid else {}
        return is_valid, msg, diag
    
    # ==========================================
    # TELEMETRY/DEBUGGING METHODS
    # ==========================================
    
    def get_coherence_telemetry(self):
        """
        Get coherence data for telemetry logging.
        
        Returns: Dict with coherence metrics or empty dict if invalid
        """
        diag = self.coherent_tracker.get_diagnostics()
        if not diag.get('is_valid', False):
            return {}
        
        # Format for CSV telemetry
        telemetry = {
            'frame_id': diag.get('frame_id', 0),
            'execution_warnings': diag.get('execution_warnings', 0),
            'color_errors': diag.get('color_errors', 0),
            'avg_coherence': diag.get('avg_coherence', 0.0),
        }
        
        # Add individual scale U-values (YUV color) and coherence
        scale_info = self.coherent_tracker.get_all_scales_info()
        u_values = self.coherent_tracker.get_all_u_values()
        
        for idx in range(4):
            telemetry[f'scale_coherence_{idx}'] = scale_info[idx].get('coherence', 0.0) if idx < len(scale_info) else 0.0
            telemetry[f'scale_u_value_{idx}'] = u_values[idx] if idx < len(u_values) else 250.0  # Neutral fallback
            
        return telemetry
    
    # ==========================================
    # RESET FOR NEXT FRAME (CALLED BY MAIN LOOP)
    # ==========================================
    
    def reset_tracker_for_next_frame(self):
        """
        Prepare coherent tracker for next frame.
        
        CALL FROM MAIN LOOP after HUD rendering is complete.
        This maintains the execution contract.
        """
        self.coherent_tracker.reset_for_next_frame()
        
# ==========================================
# 🚀 MAIN RUNNER
# ==========================================

def seed_n_from_legacy_memory(magi):
    """
    One-time migration: if old magi_memory.pt exists and n_v10x_memory.pt does not,
    seed N bank from legacy main bank records.

    Legacy fields available:  meta_freq, meta_delay, timestamps, access_counts.
    We carry over ALL of these — access_counts preserves accumulated gravity,
    timestamps preserve real age, freq/delay become phase proxy coords.

    child/youth/adult/elder default to 0.5 (no lens data in legacy format).
    These self-correct as the system runs on the seeded memories.
    """
    leg_file      = LEGACY_MEMORY_FILE
    n_file        = magi.n_bank.N_MEMORY_FILE
    n_file_v129   = magi.n_bank.V129_N_MEMORY_FILE  # v130: also check old filename
    main_new_file = MEMORY_FILE

    # ─── Main bank migration ─────────────────────────────────────────────────
    if magi.memory_bank.size == 0 and os.path.exists(leg_file):
        print(f"📀 Loading legacy memory into main bank from {leg_file}...")
        magi.memory_bank.load(leg_file)
        if magi.memory_bank.size > 0:
            magi.memory_bank.save(main_new_file)
            print(f"💾 Saved main memory as {main_new_file}")

    # ─── N bank seeding ──────────────────────────────────────────────────────
    if os.path.exists(n_file) or os.path.exists(n_file_v129):
        return  # N already has its own file (new or old) — skip

    if not os.path.exists(leg_file):
        return  # No legacy file to migrate from

    print(f"🔄 Legacy migration: seeding N bank from {leg_file}...")
    try:
        data = torch.load(leg_file, map_location=magi.dev)
        sz   = data['size']
        if sz == 0:
            print("  Legacy file empty — nothing to migrate.")
            return

        sz = min(sz, magi.n_bank.max_memories)

        meta_freq      = data['meta_freq'][:sz].float()
        meta_delay     = data['meta_delay'][:sz].float()
        timestamps     = data['timestamps'][:sz].double()
        access_counts  = data['access_counts'][:sz].float()

        # ── Epoch anchored to data, linear normalization ───────────────────
        # epoch = oldest legacy timestamp minus 30 days.
        # Ensures the oldest memory normalises to ~0.22 (not 0.0 or 1.0),
        # giving real temporal spread across all legacy memories.
        # _system_epoch persists in n_v10x_memory.pt — stable across restarts.
        ONE_MONTH = 30 * 86400
        epoch     = timestamps.min().item() - ONE_MONTH
        now_span  = time.time() - epoch
        magi.n_bank._system_epoch = epoch

        lt_norm  = torch.clamp(
            (timestamps.float() - epoch) / (now_span + 1e-6), min=0.0, max=1.0
        )

        oldest_days = (timestamps.min().item() - epoch) / 86400.0
        newest_days = (timestamps.max().item() - epoch) / 86400.0
        print(f"  Calendar: oldest={oldest_days:.1f} days  newest={newest_days:.1f} days  "
              f"(norm {lt_norm.min():.5f}→{lt_norm.max():.5f})  |  "
              f"access_counts [{access_counts.min():.0f}–{access_counts.max():.0f}]")

        # ── Phase coords: freq/delay → v131 log wrapped phase ────────────────
        phase_0 = freq_to_phase_t(meta_freq.clamp(min=MIN_FREQ))  # wrapped [0, 2π)
        phase_1 = delay_to_phase_t(meta_delay.clamp(min=MIN_DELAY))
        phase_2 = torch.full_like(phase_0, math.pi)   # neutral — no v8x data
        phase_3 = torch.full_like(phase_0, math.pi)   # neutral — no v8x data

        # ── Dims 5-6: unwrapped log coordinates (octave-aware) ──────────────
        freq_log  = freq_to_log_coord_t(meta_freq.clamp(min=MIN_FREQ))
        delay_log = delay_to_log_coord_t(meta_delay.clamp(min=MIN_DELAY))

        # ── Bulk write directly — bypasses store() to preserve timestamps ────
        # v131: Unwrapped log coords for octave-aware matching
        coords_bulk = torch.stack(
            [phase_0, phase_1, phase_2, phase_3, lt_norm,
             freq_log, delay_log], dim=1
        ).to(magi.dev)

        nb = magi.n_bank
        nb.coords[:sz]          = coords_bulk
        nb.metadata_child[:sz]  = 0.5   # no lens data in legacy — self-corrects
        nb.metadata_youth[:sz]  = 0.5
        nb.metadata_adult[:sz]  = 0.5
        nb.metadata_elder[:sz]  = 0.5
        nb.metadata_freq[:sz]   = meta_freq.to(magi.dev)
        nb.metadata_delay[:sz]  = meta_delay.to(magi.dev)
        nb.timestamps[:sz]      = timestamps.to(magi.dev)        # real timestamps ✅
        nb.access_counts[:sz]   = access_counts.to(magi.dev)     # gravity preserved ✅
        nb.size                 = sz

        magi.n_bank.save()
        print(f"✅ Legacy migration complete: {sz:,} memories seeded into N bank.")
    except Exception as e:
        print(f"⚠️  Legacy migration failed: {e}")


def run_magi_v55(serial_port='COM9', display_video=True):
    print(f"🐝 MaGi Hive v55 [FULL ALE + VIEWER + SCREEN GRAB] Initializing on {DEVICE}")
    
    magi = MaGiHive(NUM_WORKERS, DEVICE)
    wm = ScaleAwareWorkerManager(NUM_WORKERS, DEVICE)
    Serial = ComSerial(serial_port, BAUD_RATE)
    video_source = UnifiedVideoSource()
    magi.video_source = video_source
    
    av_capture = LiveAVCapture()
    v_proc = GoldenRatioVideoProcessor()
    a_ext = AudioGeometricExtractor()
    cmd_list = RuntimeCommandListener(magi)
    # v141: expose av_capture to MaGiHive so the `mic` command can reach it.
    magi.av_capture = av_capture
    # v138 (carried into v139): expose command_queue to MaGiHive so
    # BridgeController._execute_command and _entry_send can route
    # bridge-fired commands into the same queue that the typed-input
    # listener feeds. Without this, bridge commands log but never execute.
    magi.command_queue = cmd_list.command_queue
    
    magi.memory_bank.load(MEMORY_FILE)
    seed_n_from_legacy_memory(magi)   # one-time N seeding from legacy bank if needed
    
    Serial.println("time_ms,freq_hz,delay_ms,freq_wrap,delay_wrap,adult_deg,elder_deg,align_diff,quadrant_balance,ne_pct,global_coh,hb_coh,s_coh,cross_tension,audio_energy,visual_mean,visual_std,raw_std,scale0,scale1,scale2,scale3,mem_count,sensory_mod,phase_amp,phase_vel,similarity,avg_access,mem_strength,sine_w,audio_w,vs0_w,vs1_w,vs2_w,vs3_w,mixed_w,best_worker_id,job_type,mode,action,action_val,bh_worker_val,bh_deletions,bh_in_field,capacity_pct,bh_effective_radius,n_bank_size,n_bh_deletions,n_bh_in_field,n_audio_sim,n_audio_access,n_video_sim,n_video_access,dream_n_field,dream_m_field,dream_n_boosted,dream_m_boosted")
    
    torch.set_grad_enabled(False)  # global no_grad — prevents autograd graph accumulation
    step, sim_time, sine_phase = 0, 0, 0.0
    if display_video: cv2.namedWindow('MaGi v55', cv2.WINDOW_NORMAL)
    
    print("\n🎮 COMMANDS: mode [webcam|ale path|screencap|screen|viewer path], save, stats")
    print("🎯 ALE: Using 6 workers for all 18 actions (LEFT, RIGHT, FIRE, UP, DOWN, NOOP)")
    print("🦾 ROBOT: robot [enable <ip> | disable | 1mode | 2mode | 3mode | 4mode]")
    print("🎙️  MIC: mic [local | remote <worker_ip> | doa]   (DOA listener active in remote mode)")
    print("🎨 COHERENT TRACKER: 4 scales with YUV color extraction from coherent sectors only")

    try:
        with torch.no_grad():
         pass
        while True:
            cmd_list.process_commands()
            
            # Capture frame and process
            frame = video_source.get_frame()
            aud = a_ext.extract_geometry(av_capture.get_audio_chunk())
            v_nrg = v_proc.process_frame(frame)
            v_stat = v_proc.get_statistics(v_nrg)
            
            # Generate sine wave for heartbeat
            sine_phase += 0.05
            sine_val = abs(math.sin(sine_phase)) * 500.0
            
            # ==========================================
            # 🎨 COHERENT WORKER PROCESSING
            # ==========================================
            # Pass raw BGR frame to worker manager for YUV color extraction
            # Coherent tracker will automatically extract color from coherent sectors
            inputs = wm.get_inputs_tensor(sine_val, aud['energy'], v_nrg, 
                                         magi.phases_s[:,0], magi.current_gravity_context, 
                                         magi, raw_bgr_frame=frame)
            
            # Process MaGi physics
            magi.process_step(inputs)
            magi.update_metrics()
            magi.apply_natural_physics_and_memory(step)
            deleted_this_step = magi.apply_black_hole_deletion()
            magi.query_n_carriers()           # N lens update: audio + video carriers
            # v131: ALE beacons now fire inside process_step (after assembly, before integration)
            magi.apply_n_bh_deletion()        # N BH: cluster-aware sparse deletion
            magi.dream_coupling.update(magi)   # v128 Part 2a: Drift pair   (1552-1553)
            magi.chord_coupling.update(magi)   # v128 Part 2b: Teleport pair (1554-1555)
            magi.physics_coupling.update(magi) # v128 Part 2c: Lens-driven pair (1556-1557)

            # # v128: Post-coupling velocity clamp — prevents runaway stacking
            # # (base hive + impulse + persistent drift + existing forces)
            # for _idx in range(1552, 1558):
            #     _v = magi.vel_hb[_idx]
            #     _mag = torch.norm(_v)
            #     _max_speed = 0.5 if _idx <= 1553 else 0.08   # drift pair roams freely; chord/physics tighter
            #     if _mag > _max_speed:
            #         magi.vel_hb[_idx] = _v / _mag * _max_speed
            # got rid of clamp.. to slow and not emergent. 

            magi.update_voice(magi.video_source.mode)
            
            # ==========================================
            # 🕹️ EMERGENT CONTROL
            # ==========================================
            action_str = "N/A"
            action_val = 0.0
            action_idx = 0  # Default NOOP
            
            # Process all ALE workers through scalers (1542-1547)
            left_result = magi.scalers['ale_left'].process(magi.s_filtered[1542].item())
            right_result = magi.scalers['ale_right'].process(magi.s_filtered[1543].item())
            fire_result = magi.scalers['ale_fire'].process(magi.s_filtered[1544].item())
            up_result = magi.scalers['ale_up'].process(magi.s_filtered[1545].item())
            down_result = magi.scalers['ale_down'].process(magi.s_filtered[1546].item())
            noop_result = magi.scalers['ale_noop'].process(magi.s_filtered[1547].item())

            # Get scaled values (0 = inactive, >0 = active)
            left_val = left_result['output'] if left_result['is_active'] else 0.0
            right_val = right_result['output'] if right_result['is_active'] else 0.0
            fire_val = fire_result['output'] if fire_result['is_active'] else 0.0
            up_val = up_result['output'] if up_result['is_active'] else 0.0
            down_val = down_result['output'] if down_result['is_active'] else 0.0
            noop_val = noop_result['output'] if noop_result['is_active'] else 0.0

            # Optional: Watch MaGi discover ranges
            if left_result['expanded']:
                print(f"⬅️ LEFT expanded to {left_result['observed_max']:.1f}")
            if right_result['expanded']:
                print(f"➡️ RIGHT expanded to {right_result['observed_max']:.1f}")
            if fire_result['expanded']:
                print(f"🔥 FIRE expanded to {fire_result['observed_max']:.1f}")

            if magi.mode == 'ale':
                # Use absolute values for direction strength
                up_strength = abs(up_val)
                down_strength = abs(down_val)
                left_strength = abs(left_val)
                right_strength = abs(right_val)
                
                # Collect active directions (scaler already filtered noise/deadzone)
                active_directions = []
                if up_strength > 0: 
                    active_directions.append(('UP', up_strength, 2))
                if down_strength > 0: 
                    active_directions.append(('DOWN', down_strength, 5))
                if left_strength > 0: 
                    active_directions.append(('LEFT', left_strength, 4))
                if right_strength > 0: 
                    active_directions.append(('RIGHT', right_strength, 3))
                
                active_directions.sort(key=lambda x: x[1], reverse=True)
                
                fire_active = fire_val > 0  # Simple boolean - scaler decides if fire is happening
                
                # Action selection logic (unchanged)
                if not active_directions and not fire_active:
                    action_idx = 0; action_str = 'NOOP'
                elif not active_directions and fire_active:
                    action_idx = 1; action_str = 'FIRE'
                else:
                    if len(active_directions) == 1:
                        dir_name, dir_strength, dir_idx = active_directions[0]
                        action_idx = dir_idx
                        action_str = dir_name
                        if fire_active:
                            if dir_idx == 2: action_idx = 10; action_str = 'UPFIRE'
                            elif dir_idx == 5: action_idx = 13; action_str = 'DOWNFIRE'
                            elif dir_idx == 4: action_idx = 12; action_str = 'LEFTFIRE'
                            elif dir_idx == 3: action_idx = 11; action_str = 'RIGHTFIRE'
                    else:
                        dir_names = [d[0] for d in active_directions]
                        if 'UP' in dir_names and 'RIGHT' in dir_names: action_idx = 6; action_str = 'UPRIGHT'
                        elif 'UP' in dir_names and 'LEFT' in dir_names: action_idx = 7; action_str = 'UPLEFT'
                        elif 'DOWN' in dir_names and 'RIGHT' in dir_names: action_idx = 8; action_str = 'DOWNRIGHT'
                        elif 'DOWN' in dir_names and 'LEFT' in dir_names: action_idx = 9; action_str = 'DOWNLEFT'
                        else: action_idx = active_directions[0][2]; action_str = active_directions[0][0]
                        
                        if fire_active and action_str in ['UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']:
                            action_idx += 8; action_str += 'FIRE'
                
                video_source.execute_action(action_idx)
                
                # Action value for telemetry
                if action_idx == 0: action_val = noop_val
                elif action_idx == 1: action_val = fire_val
                elif action_idx == 2: action_val = up_val
                elif action_idx == 5: action_val = down_val
                elif action_idx == 4: action_val = left_val
                elif action_idx == 3: action_val = right_val
                else:
                    relevant_vals = []
                    if 'UP' in action_str: relevant_vals.append(up_val)
                    if 'DOWN' in action_str: relevant_vals.append(down_val)
                    if 'LEFT' in action_str: relevant_vals.append(left_val)
                    if 'RIGHT' in action_str: relevant_vals.append(right_val)
                    if 'FIRE' in action_str: relevant_vals.append(fire_val)
                    action_val = max(relevant_vals) if relevant_vals else 0.0
                
                if magi.video_source.ale and magi.video_source.ale.game_over():
                    print(f"\n--- 🥅 GAME OVER DETECTED: Resetting Game (Last Action: {action_str}) ---")
                    magi.video_source.ale.reset_game()
                    print("--- New game started. ---")

            elif magi.mode == 'viewer':
                # Fully adaptive viewer - scaler decides what's active
                if left_result['is_active'] and not right_result['is_active']:
                    video_source.viewer_nav('PREV')
                    action_str = "PREV"
                    action_val = left_val
                elif right_result['is_active'] and not left_result['is_active']:
                    video_source.viewer_nav('NEXT')
                    action_str = "NEXT"
                    action_val = right_val
                else:
                    action_str = "N/A"
                    action_val = 0.0

            elif magi.mode == 'remote':
                # Battery low guard — fall back to webcam if below threshold
                if magi.remote_sender:
                    batt_pct, _, _ = magi.remote_sender.get_battery()
                    if 0.0 <= batt_pct < REMOTE_BATTERY_LOW_PCT:
                        print(f"🔋 Battery low ({batt_pct:.1f}%) — switching to webcam")
                        magi.remote_sender.close()
                        magi.remote_sender = None
                        magi.video_source.switch_to_webcam()
                        magi.mode = 'webcam'
                        action_str = "BATTERY_LOW"
                        action_val = 0.0
                        continue
                # Map LEFT/RIGHT/UP/DOWN workers → pan/tilt commands to Pi
                if magi.remote_sender:
                    # Only trigger on POSITIVE values
                    if left_result['is_active'] and left_val > 0 and not right_result['is_active']:
                        magi.remote_sender.pan_left()
                        action_str = "PAN_LEFT"
                        action_val = left_val
                    elif right_result['is_active'] and right_val > 0 and not left_result['is_active']:
                        magi.remote_sender.pan_right()
                        action_str = "PAN_RIGHT"
                        action_val = right_val
                    elif up_result['is_active'] and up_val > 0 and not down_result['is_active']:
                        magi.remote_sender.tilt_up()
                        action_str = "TILT_UP"
                        action_val = up_val
                    elif down_result['is_active'] and down_val > 0 and not up_result['is_active']:
                        magi.remote_sender.tilt_down()
                        action_str = "TILT_DOWN"
                        action_val = down_val
                    else:
                        action_str = "N/A"
                        action_val = 0.0

            elif magi.mode == 'screen':
                # No commands for now, as requested.
                action_str = "N/A"
                action_val = 0.0

            # ==========================================
            # 🦾 ROBOT ARM CONTROL (concurrent with any video mode)
            # ==========================================
            if magi.robot_sender is not None:
                # Process robot scalers
                rx_result    = magi.scalers['robot_x'].process(magi.s_filtered[1558].item())
                ry_result    = magi.scalers['robot_y'].process(magi.s_filtered[1559].item())
                rz_result    = magi.scalers['robot_z'].process(magi.s_filtered[1560].item())
                rrot_result  = magi.scalers['robot_rot'].process(magi.s_filtered[1561].item())
                rtilt_result = magi.scalers['robot_tilt'].process(magi.s_filtered[1562].item())
                rgrip_result = magi.scalers['robot_gripper'].process(magi.s_filtered[1563].item())

                # Convert active scalers to deltas
                dx     = rx_result['output'] * ROBOT_STEP_SIZE   if (rx_result['is_active']    and 1558 in magi._robot_active_workers) else 0.0
                dy     = ry_result['output'] * ROBOT_STEP_SIZE   if (ry_result['is_active']    and 1559 in magi._robot_active_workers) else 0.0
                dz     = rz_result['output'] * ROBOT_STEP_SIZE   if (rz_result['is_active']    and 1560 in magi._robot_active_workers) else 0.0
                d_rot  = rrot_result['output'] * ROBOT_WRIST_STEP if (rrot_result['is_active']  and 1561 in magi._robot_active_workers) else 0.0
                d_tilt = rtilt_result['output'] * ROBOT_WRIST_STEP if (rtilt_result['is_active'] and 1562 in magi._robot_active_workers) else 0.0

                # Send position if any movement
                if dx != 0.0 or dy != 0.0 or dz != 0.0 or d_rot != 0.0 or d_tilt != 0.0:
                    magi.robot_sender.nudge(dx, dy, dz, d_rot, d_tilt)

                # Gripper: edge-triggered from bipolar scaler
                if 1563 in magi._robot_active_workers and rgrip_result['is_active']:
                    magi.robot_sender.set_gripper(rgrip_result['output'] > 0)

            # ==========================================
            # 📊 TELEMETRY & LOGGING
            # ==========================================
            if step % 3 == 0:
                best = torch.argmax(magi.global_coh).item()
                quad_bal, ne_pct = magi.get_quadrant_metrics(best)
                gravity = magi.current_gravity_context

                # Black hole metrics
                bh_val = magi.s_filtered[magi.black_hole_worker_idx].item()
                capacity_pct = (magi.memory_bank.size / magi.memory_bank.max_memories) * 100.0
                bh_tension = magi.black_hole_last_result.get('tension_factor', abs(magi.black_hole_last_result.get('output', 0.0)) / 1500.0)
                bh_effective_radius = magi.black_hole_base_radius * (1.0 + bh_tension)
                
                # Build telemetry line
                line = (f"{sim_time},{magi.freq[best]:.3f},{magi.delay[best]:.1f},"
                   f"{magi.freq_wrap[best].item()},{magi.delay_wrap[best].item()},"
                   f"{magi.adult_dir[best]:.1f},{magi.elder_dir[best]:.1f},"
                   f"{magi.alignment_diff[best]:.1f},{quad_bal:.1f},{ne_pct:.1f},"
                   f"{magi.global_coh[best]:.3f},{magi.hb_coh[best]:.3f},"
                   f"{magi.s_coh[best]:.3f},{magi.cross_tension[best]:.3f},"
                   f"{aud['energy']:.3f},{v_stat['mean']:.3f},{v_stat['std']:.3f},"
                   f"{v_stat['raw_std']:.3f},{v_stat['scale_0_mean']:.3f},"
                   f"{v_stat['scale_1_mean']:.3f},{v_stat['scale_2_mean']:.3f},"
                   f"{v_stat['scale_3_mean']:.3f},{magi.memory_bank.size},"
                   f"{gravity.get('sensory_modulation',0):.3f},{gravity.get('phase_amplitude',0):.3f},"
                   f"{gravity.get('phase_velocity',1):.3f},{gravity.get('similarity',0):.3f},"
                   f"{gravity.get('avg_access',0):.1f},{magi.memory_bank.memory_influence_strength:.4f},"
                   f"2,516,948,516,531,571,0,{best},{wm.get_job_description(best)},"
                   f"{magi.mode},{action_str},{action_val:.1f},"
                   f"{bh_val:.1f},{magi.black_hole_step_deletions},{magi.black_hole_memories_in_field},{capacity_pct:.2f},{bh_effective_radius:.4f},"
                   f"{magi.n_bank.size},{magi.n_bh_step_deletions},{magi.n_bh_memories_in_field},"
                   f"{magi.n_gravity_audio.get('chord_size',0)},{magi.n_gravity_audio.get('top_access',0):.0f},"
                   f"{magi.n_gravity_video.get('chord_size',0)},{magi.n_gravity_video.get('top_access',0):.0f},"
                   f"{magi.dream_coupling.n_in_field},{magi.dream_coupling.m_in_field},"
                   f"{len(magi.dream_coupling.n_last_boosted)},{len(magi.dream_coupling.m_last_boosted)}")
                Serial.println(line)
                
                # ==========================================
                # 🎨 HUD VISUALIZATION
                # ==========================================
                if display_video and frame is not None:
                    hud = frame.copy()
                    
                    # Use worker manager's coherent tracker for HUD drawing
                    # This ensures visual-actuation alignment (same workers as input processing)
                    hud_drawing_info = wm.get_hud_drawing_info(hud.shape[1], hud.shape[0])
                    
                    # Draw attention squares for coherent workers
                    for draw_info in hud_drawing_info:
                        x1, y1, x2, y2 = draw_info['rect']
                        color = draw_info['color']
                        thickness = draw_info['thickness']
                        
                        # Draw rectangle
                        cv2.rectangle(hud, (x1, y1), (x2, y2), color, thickness)
                        
                        # Add label for dense scales
                        if draw_info.get('label'):
                            cv2.putText(hud, draw_info['label'], (x1 + 5, y1 + 20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # HUD Text Overlay
                    cv2.putText(hud, f"MODE: {magi.mode.upper()}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    cv2.putText(hud, f"COH: {magi.global_coh[best]:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    
                    if magi.mode == 'ale':
                        color = (0,255,0) if action_str != 'NOOP' else (200,200,200)
                        cv2.putText(hud, f"ACT: {action_str}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(hud, f"L:{left_val:.0f} R:{right_val:.0f} U:{up_val:.0f} D:{down_val:.0f}", 
                                  (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                        cv2.putText(hud, f"F:{fire_val:.0f} N:{noop_val:.0f}", 
                                  (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                    elif magi.mode == 'viewer':
                        cv2.putText(hud, f"NAV: {action_str}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
                        cv2.putText(hud, f"L:{left_val:.0f} R:{right_val:.0f}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                    
                    # Show HUD
                    cv2.imshow('MaGi v55', hud)
            
            # ==========================================
            # 📈 PERIODIC SAVING & MAINTENANCE
            # ==========================================
            if step % 2000 == 0 and step > 0:
                magi.memory_bank.save(MEMORY_FILE)
                magi.n_bank.save()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()   # periodic VRAM defrag
            
            # Prepare tracker for next frame
            wm.reset_tracker_for_next_frame()
            
            # Exit on 'q' key
            if display_video and cv2.waitKey(1) & 0xFF == ord('q'): 
                break
            
            step += 1
            sim_time += 50

    except KeyboardInterrupt:
        print("\n🛑 Halting...")
    finally:
        magi.memory_bank.save(MEMORY_FILE)
        magi.n_bank.save()
        av_capture.stop()
        video_source._cleanup()
        if magi.robot_sender:
            magi.robot_sender.close()
        magi.voice_carrier.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_magi_v55()
