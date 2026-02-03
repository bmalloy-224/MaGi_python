
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
# Contact: https://github.com/bmalloy-224/MaGi_python




import torch
import torch.nn.functional as F
import numpy as np
import math
import time
import threading
import queue
import cv2
import pyaudio
import serial
import os
import sys
import select
import tkinter as tk
from PIL import ImageGrab
import platform
import mss

# ==========================================
# 📌 CONFIGURATION
# ==========================================
TARGET_PORT = 'COM9'
BAUD_RATE = 115200
# was 115200
NUM_WORKERS = 8100  # Increased for 6 ALE workers + lazy river
# was 8100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEMORY_FILE = "magi_memory.pt"

# Physics Constants
CHILD_SENSITIVITY = 0.8
YOUTH_GAIN = 1.0
ADULT_THRESHOLD = 0.3
ELDER_TIME_CONSTANT = 0.95
HB_SINE_SCALE = 500.0
TENSION_FREQ_COUPLING = 0.05
COHERENCE_DELAY_COUPLING = 0.1
ELASTICITY = 0.92 
ADULT_GUIDANCE_STRENGTH = 0.01

# Memory Configuration
STORE_COHERENCE_STABLE = 0.70
STORE_COHERENCE_PEAK = 0.85
MAX_MEMORIES = 3000000 
MIN_FREQ, MAX_FREQ = 0.5, 70.0
MIN_DELAY, MAX_DELAY = 1.0, 100.0

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
# UPE
# ========================================== 


class UniversalPlasticityEngine:
    def __init__(self, device='cuda'):
        self.dev = device
        self.file_path = "motor_voice_map.pth"
        self.bh_idx = 1549
        
        # BASELINE: Hardened Geometry (Radius 2.15 across all axes)
        self.baseline = {
            1542: {'name': 'LEFT',  'type': 'ale',   'r': 2.15, 'home': [1.1, 0.9, -2.5,  0.5]},
            1543: {'name': 'RIGHT', 'type': 'ale',   'r': 2.15, 'home': [1.1, 0.9,  2.5, -0.5]},
            1544: {'name': 'FIRE',  'type': 'ale',   'r': 2.15, 'home': [1.1, 0.9,  0.5, -2.5]},
            1545: {'name': 'UP',    'type': 'ale',   'r': 2.15, 'home': [1.1, 0.9,  0.5,  2.5]},
            1546: {'name': 'DOWN',  'type': 'ale',   'r': 2.15, 'home': [1.1, 0.9, -1.0, -2.0]},
            1547: {'name': 'NOOP',  'type': 'ale',   'r': 2.15, 'home': [1.1, 0.9, -1.0,  1.0]},
            1548: {'name': 'VOICE', 'type': 'voice', 'r': 2.15, 'home': [4.71, 3.14, 1.57, 0.0]}
        }
        
        # UPE physics: Instant Snap + Long-Term Persistence
        self.home_drift_strength = 0.15      # High impact contact
        self.home_drift_damping = 0.95       
        self.window_duration = 8 * 60 * 60   # 8-Hour "Memory Lock"
        self.accumulation_threshold = 0.02   # Fast trigger
        self.pressure_decay_halflife = 24 * 60 * 60  # 24-Hour Persistence
        
        # Velocity impulse feedback: Binary Punch
        self.velocity_impulse_gain = 2.5     
        self.max_impulse = 0.5               
        self.min_impulse = 0.05              
        
        # Sensory feedback
        self.sensory_feedback_gain = 150.0
        
        # Tracking
        self.pressure_history = {}
        self.last_applied = {}
        self.last_pressure = {}  # For pressure delta calculation
        self.homes = self._load_or_init()
        self.steps_since_save = 0
        self.last_saved_homes = None

    def _load_or_init(self):
        """Load saved home positions or initialize from baseline"""
        if os.path.exists(self.file_path):
            saved = torch.load(self.file_path, map_location=self.dev)
            print(f"✅ UPE: Loaded saved home positions")
            return saved
        
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
        """
        Return a-b wrapped to [-π, π] in 4D.
        """
        diff = a - b
        diff = torch.where(diff > math.pi, diff - 2*math.pi, diff)
        diff = torch.where(diff < -math.pi, diff + 2*math.pi, diff)
        return diff

    def apply_singularity_bumper(self, magi_hive):
        """
        🛡️ UNIFIED COLLISION DETECTION (v5.3)
        Proactive guardian that prevents ALL coordinate collisions:
        1. Worker vs Black Hole (singularity protection)
        2. Worker vs Worker (ghost prevention)
        
        Runs every frame regardless of drift state.
        """
        bh_phase = magi_hive.phases_hb[self.bh_idx]
        bh_active = abs(magi_hive.s_filtered[self.bh_idx].item()) > 50.0
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 1: BLACK HOLE COLLISION DETECTION
        # ═══════════════════════════════════════════════════════════
        if bh_active:
            for idx, data in self.homes.items():
                home = data['home']
                
                # Get wrapped differences
                delta_freq = (home[0] - bh_phase[0] + math.pi) % (2 * math.pi) - math.pi
                delta_delay = (home[1] - bh_phase[1] + math.pi) % (2 * math.pi) - math.pi
                
                # Check if within danger zone in BOTH dimensions
                if abs(delta_freq) < 0.05 and abs(delta_delay) < 0.05:
                    # Determine bump direction (preserve sign, default +0.5)
                    target_freq = 0.5 if delta_freq >= 0 else -0.5
                    target_delay = 0.5 if delta_delay >= 0 else -0.5
                    
                    # Apply bump
                    new_home = home.clone()
                    new_home[0] = (bh_phase[0] + target_freq) % (2 * math.pi)
                    new_home[1] = (bh_phase[1] + target_delay) % (2 * math.pi)
                    
                    # Update
                    data['home'] = new_home
                    magi_hive.phases_hb[idx] = new_home
                    
                    print(f"🌀 BH BUMPER: {data['name']} cleared from singularity")
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 2: WORKER-vs-WORKER COLLISION DETECTION
        # ═══════════════════════════════════════════════════════════
        # Check ALL pairs for ghosting (runs every frame)
        worker_indices = list(self.homes.keys())
        
        for i, idx_a in enumerate(worker_indices):
            data_a = self.homes[idx_a]
            home_a = data_a['home']
            
            for idx_b in worker_indices[i+1:]:  # Only check each pair once
                data_b = self.homes[idx_b]
                home_b = data_b['home']
                
                # 2D collision detection (Freq/Delay only)
                d_f = torch.abs(home_a[0] - home_b[0])
                d_d = torch.abs(home_a[1] - home_b[1])
                d_f = torch.min(d_f, 2*math.pi - d_f)
                d_d = torch.min(d_d, 2*math.pi - d_d)
                
                # Ghost detection threshold (0.1 padding)
                if d_f < 0.1 and d_d < 0.1:
                    # COLLISION DETECTED - Bump worker B away from worker A
                    # (Deterministic: always bump the "later" worker in the list)
                    
                    new_home_b = home_b.clone()
                    new_home_b[0] = (home_b[0] + 0.2) % (2 * math.pi)
                    new_home_b[1] = (home_b[1] + 0.2) % (2 * math.pi)
                    
                    # Update worker B
                    data_b['home'] = new_home_b
                    magi_hive.phases_hb[idx_b] = new_home_b
                    
                    print(f"⚡ GHOST BUMPER: {data_b['name']} separated from {data_a['name']} "
                          f"[Δf={d_f:.3f}, Δd={d_d:.3f}]")
        
    def apply_black_hole_gravity(self, magi_hive):
        """
        Hybrid approach:
        - 2D DETECTION (matches BH exactly for freq/delay)
        - 4D MOVEMENT (full dimensional gravity for pressure)
        """
        current_time = time.time()
        
        bh_idx = self.bh_idx  # 1549
        bh_phase = magi_hive.phases_hb[bh_idx]  # [X,Y,Z,W] - ACTUAL 4D position!
        bh_val = magi_hive.s_filtered[bh_idx].item()
        
        if not self.pressure_history:
            for idx in self.homes:
                self.pressure_history[idx] = []
                self.last_applied[idx] = current_time
                self.last_pressure[idx] = 0.0
        
        if abs(bh_val) < 50.0:
            for idx in self.last_pressure:
                self.last_pressure[idx] *= 0.98
            return {}
        
        self.apply_singularity_bumper(magi_hive)
        # ✅ SYNCED WITH BH: Same tension and radius calculation
        abs_val = abs(bh_val)
        tension_factor = math.tanh(abs_val / 500.0)
        effective_radius = magi_hive.black_hole_base_radius * (1.0 + tension_factor)
        
        radius_scale = tension_factor
        current_eps_peak = magi_hive.black_hole_eps_floor + \
                          (magi_hive.black_hole_eps_max - magi_hive.black_hole_eps_floor) * radius_scale
        k = 1.0 + tension_factor * 4.0
        
        for idx, data in self.homes.items():
            home_pos = data['home']  # [X,Y,Z,W]
            
            # ✅ 2D DETECTION ONLY (matches BH exactly!)
            delta_freq = torch.abs(home_pos[0] - bh_phase[0])
            delta_delay = torch.abs(home_pos[1] - bh_phase[1])
            delta_freq = torch.min(delta_freq, 2*math.pi - delta_freq)
            delta_delay = torch.min(delta_delay, 2*math.pi - delta_delay)
            distance_2d = torch.sqrt(delta_freq**2 + delta_delay**2)
            
            cutoff_time = current_time - self.window_duration
            self.pressure_history[idx] = [
                (t, p_vec) for t, p_vec in self.pressure_history[idx]
                if t > cutoff_time
            ]
            
            if distance_2d < effective_radius:
                # ✅ 4D MOVEMENT VECTOR (full dimensional gravity)
                delta_wrapped = self._wrapped_difference(home_pos, bh_phase)
                to_bh_4d = -delta_wrapped  # Full 4D direction
                to_bh_norm_4d = to_bh_4d / (torch.norm(to_bh_4d) + 1e-12)
                
                # Normalized 2D distance for gradient (matches BH)
                d_norm_2d = torch.clamp(distance_2d / (effective_radius + 1e-12), 0.0, 1.0)
                
                if bh_val > 0:
                    # VACUUM MODE: Stronger toward center
                    decay_gradient = magi_hive.black_hole_eps_floor + \
                                    (current_eps_peak - magi_hive.black_hole_eps_floor) * \
                                    torch.pow(1.0 - d_norm_2d, k)
                    direction = 1.0
                    pressure_vector = to_bh_norm_4d * decay_gradient * self.home_drift_strength
                else:
                    # SHIELD MODE: Stronger toward edge
                    decay_gradient = magi_hive.black_hole_eps_floor + \
                                    (current_eps_peak - magi_hive.black_hole_eps_floor) * \
                                    torch.pow(d_norm_2d, k)
                    direction = -1.0
                    pressure_vector = -to_bh_norm_4d * decay_gradient * self.home_drift_strength
                
                self.pressure_history[idx].append((current_time, pressure_vector))
                
                if self.pressure_history[idx]:
                    accumulated_pressure = torch.zeros(4, device=self.dev)
                    for t, p_vec in self.pressure_history[idx]:
                        age = current_time - t
                        decay = math.exp(-age / self.pressure_decay_halflife)
                        accumulated_pressure += p_vec * decay
                    
                    pressure_magnitude_total = torch.norm(accumulated_pressure).item()
                    
                    # Velocity impulse feedback
                    pressure_delta = max(0.0, pressure_magnitude_total - self.last_pressure[idx])
                    
                    if pressure_delta > self.min_impulse:
                        impulse_magnitude = min(pressure_delta * self.velocity_impulse_gain, 
                                               self.max_impulse)
                        impulse_vector = to_bh_norm_4d * impulse_magnitude * direction
                        magi_hive.vel_hb[idx] += impulse_vector
                        magi_hive.s_filtered[idx] += impulse_magnitude * self.sensory_feedback_gain
                    
                    self.last_pressure[idx] = pressure_magnitude_total
                    
                    # Permanent home drift
                    if pressure_magnitude_total > self.accumulation_threshold:
                        drift_amount = min(self.accumulation_threshold / pressure_magnitude_total, 1.0)
                        drift_vector = accumulated_pressure * drift_amount
                        
                        old_home = home_pos.clone()
                        data['home'] = (data['home'] + drift_vector) % (2 * math.pi)
                        
                        tax = min(1.0, drift_amount)
                        self.pressure_history[idx] = [
                            (t, p_vec * (1.0 - tax)) 
                            for t, p_vec in self.pressure_history[idx]
                        ]
                        
                        remaining_mag = torch.norm(accumulated_pressure * (1.0 - tax)).item()
                        self.last_applied[idx] = current_time
                        self.last_pressure[idx] = remaining_mag
                        
                        move_distance = torch.norm(data['home'] - old_home).item()
                        if move_distance > 0.001:
                            print(f"🏠 {data['name']}: MOVE {move_distance:.3f}")
            else:
                if idx in self.last_pressure:
                    self.last_pressure[idx] *= 0.98
        
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
        """
        stats = {}
        current_time = time.time()
        
        # Momentum memory (short-term pulse)
        if not hasattr(self, '_momentum_memory'):
            self._momentum_memory = {}
        
        for idx, data in self.homes.items():
            # Dual accumulators in 4D - THE CORE PHYSICS
            snap_accumulator = torch.zeros(4, device=self.dev)   # All-time decayed pressure (THE TRUTH)
            intent_accumulator = torch.zeros(4, device=self.dev) # 8h window (TREND)
            
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
                    
                    # Intent (Trend): Only last 8 hours
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

    def maybe_save(self, phases_hb):
        """
        Smart save logic: only save if homes have moved significantly.
        
        When homes drift, we need to:
        1. Save the new home positions to disk
        2. Update phases_hb so workers use new homes immediately
        
        Args:
            phases_hb: MaGi's worker positions (updated when homes save)
        """
        self.steps_since_save += 1
        
        # Initialize last saved positions if first time
        if self.last_saved_homes is None:
            self.last_saved_homes = {}
            for idx, data in self.homes.items():
                self.last_saved_homes[idx] = data['home'].clone()
        
        # Calculate max drift from last saved positions (4D distance!)
        max_drift = 0.0
        for idx, data in self.homes.items():
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
            torch.save(save_data, self.file_path)
            
            # CRITICAL: Update phases_hb so workers use new home positions
            for idx, data in self.homes.items():
                phases_hb[idx] = data['home'].clone()
                self.last_saved_homes[idx] = data['home'].clone()
            
            self.steps_since_save = 0
            if max_drift > 0.1:
                print(f"💾 UPE: Saved home positions (max 4D drift: {max_drift:.3f})")
                print(f"    → Workers reset to new home positions")



# ==========================================
# 🔊 PURE CARRIER VOICE WORKER
# ==========================================

class PureCarrierVoice:
    """
    Dedicated audio worker with [0,0,0,0] starting phase.
    Enhanced with value-responsive harmonics that give semantic meaning
    to MaGi's positive/negative states through timbre.
    """
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.worker_idx = 1548  # Fixed position after ALE workers
        
        # Mode activation control
        self.mode_active = {
            'webcam': True,     # Active in webcam mode
            'ale': False,       # Disabled for now
            'screencap': False,
            'screen': False,
            'viewer': False,
        }
        
        self.audio_phase = 0.0
        self.envelope_phase = 0.0
        self.is_sounding = False
        
        self.current_pitch = 0.0
        self.current_amplitude = 0.0  # Preserves sign: positive/negative
        self.current_articulation = 0.0
        
        # Audio output
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sr,
            output=True,
            frames_per_buffer=1024
        )
        
        print(f"🎵 Pure Carrier Voice at worker {self.worker_idx}")
        print(f"   Starting phase: [0,0,0,0]")
        print(f"   Active in: {[m for m, a in self.mode_active.items() if a]}")
        print(f"   Harmonics: Value-responsive (positive=harmonic, negative=inharmonic)")

    def set_mode(self, mode):
        """Set current operating mode"""
        self.current_mode = mode
        self.is_sounding = self.mode_active.get(mode, False)

    def enable_for_mode(self, mode, enable=True):
        """Enable or disable voice for specific mode"""
        if mode in self.mode_active:
            self.mode_active[mode] = enable
            print(f"  🔊 Voice {'enabled' if enable else 'disabled'} for {mode} mode")

    def update_from_magi(self, magi_hive):
        """
        Extract parameters from MaGi hive.
        Returns True if voice should be active.
        """
        idx = self.worker_idx
        freq = magi_hive.freq[idx].item()      
        delay = magi_hive.delay[idx].item()    
        value = magi_hive.s_filtered[idx].item() 
        
        # Pitch (80-1500Hz logarithmic mapping)
        freq_norm = np.clip((freq - 0.5) / 69.5, 0.0, 1.0)
        self.current_pitch = 80.0 * (1500.0 / 80.0) ** freq_norm
        
        # Amplitude with sign preserved (positive/negative semantic)
        # tanh preserves sign while smoothing extreme values
        self.current_amplitude = np.tanh(value / 350.0) * 1.4 #was 0.7.. changed from 1000 to 350
        
        # Articulation rate (1-15Hz envelope modulation)
        delay_norm = np.clip((delay - 1.0) / 99.0, 0.0, 1.0)
        self.current_articulation = 1.0 + delay_norm * 14.0
        
        # Voice is active if value has meaningful magnitude
        return abs(value) > 20.0

    def generate_sound(self, frame_duration=0.02):
        """
        Generate harmonically rich carrier wave with value-responsive timbre.
        
        Semantic mapping:
        - Positive amplitude (>0.05): harmonic series (1x, 2x, 3x) - consonant, musical
        - Negative amplitude (<-0.05): inharmonic series (1x, 1.5x, 2.5x) - dissonant, tense
        - Near zero (-0.05 to 0.05): neutral/transitional state
        
        Harmonic count scales with amplitude magnitude for expressive intensity.
        """
        if not self.is_sounding or abs(self.current_amplitude) < 0.01:
            return None
        
        amp_mag = abs(self.current_amplitude)
        num_samples = int(self.sr * frame_duration)
        sample_indices = np.arange(num_samples)
        
        # Phase calculations for audio and envelope
        audio_phase_inc = 2.0 * np.pi * self.current_pitch / self.sr
        envelope_phase_inc = 2.0 * np.pi * self.current_articulation / self.sr
        
        audio_phase = self.audio_phase + audio_phase_inc * sample_indices
        envelope_phase = self.envelope_phase + envelope_phase_inc * sample_indices
        
        # --- VALUE-RESPONSIVE HARMONICS ---
        # Determine harmonic ratios based on amplitude sign with deadband
        if self.current_amplitude > 0.05:
            # POSITIVE: Harmonic series (stable, consonant)
            base_multipliers = [1, 2, 3]          # 1x, 2x, 3x
        elif self.current_amplitude < -0.05:
            # NEGATIVE: Inharmonic series (unstable, dissonant)
            base_multipliers = [1, 1.5, 2.5]      # 1x, 1.5x, 2.5x
        else:
            # NEUTRAL: Transitional state (simple octave)
            base_multipliers = [1, 2]             # 1x, 2x
        
        # Natural amplitude decay weights
        base_weights = [1.0, 0.4, 0.2]
        
        # Dynamic harmonic count based on amplitude magnitude
        # More intensity = richer harmonic spectrum
        harmonic_count = 1 + int(min(2, amp_mag / 0.25))  # 1-3 harmonics
        harmonic_count = min(harmonic_count, len(base_multipliers))
        
        multipliers = base_multipliers[:harmonic_count]
        weights = base_weights[:harmonic_count]
        
        # Generate harmonic stack
        wave = np.zeros(num_samples)
        for m, w in zip(multipliers, weights):
            # Each harmonic uses same phase multiplied by its frequency ratio
            wave += w * np.sin(audio_phase * m)
        
        # Normalize by sum of weights to maintain consistent amplitude
        wave = (wave / sum(weights)) * amp_mag
        
        # Apply articulation envelope (tremolo)
        wave *= (0.5 + 0.5 * np.sin(envelope_phase))
        # --- END HARMONICS ---
        
        # Update global phases for continuity
        self.audio_phase = (self.audio_phase + audio_phase_inc * num_samples) % (2.0 * np.pi)
        self.envelope_phase = (self.envelope_phase + envelope_phase_inc * num_samples) % (2.0 * np.pi)
        
        # Anti-click ramps at beginning and end of buffer
        if len(wave) > 100:
            wave[:50] *= np.linspace(0, 1, 50)
            wave[-50:] *= np.linspace(1, 0, 50)
        
        return wave.astype(np.float32)

    def generate_sound_simple(self, frame_duration=0.02):
        """
        Original simple carrier wave (for comparison/fallback).
        Maintained for backward compatibility.
        """
        if not self.is_sounding or abs(self.current_amplitude) < 0.01:
            return None
        
        num_samples = int(self.sr * frame_duration)
        sample_indices = np.arange(num_samples)
        
        audio_phase_inc = 2.0 * np.pi * self.current_pitch / self.sr
        envelope_phase_inc = 2.0 * np.pi * self.current_articulation / self.sr
        
        audio_phase = self.audio_phase + audio_phase_inc * sample_indices
        envelope_phase = self.envelope_phase + envelope_phase_inc * sample_indices
        
        # Simple single sine wave
        wave = self.current_amplitude * np.sin(audio_phase) * (0.5 + 0.5 * np.sin(envelope_phase))
        
        self.audio_phase = (self.audio_phase + audio_phase_inc * num_samples) % (2.0 * np.pi)
        self.envelope_phase = (self.envelope_phase + envelope_phase_inc * num_samples) % (2.0 * np.pi)
        
        if len(wave) > 100:
            wave[:50] *= np.linspace(0, 1, 50)
            wave[-50:] *= np.linspace(1, 0, 50)
        
        return wave.astype(np.float32)

    def speak(self, magi_hive):
        """Let MaGi speak through this carrier voice"""
        is_active = self.update_from_magi(magi_hive)
        if is_active and self.is_sounding:
            audio = self.generate_sound()
            if audio is not None:
                try: 
                    self.stream.write(audio.tobytes())
                except Exception:
                    # Silent fail on audio errors to avoid interrupting main loop
                    pass

    def cleanup(self):
        """Clean up audio resources"""
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
            print(f"📊 Stats: Mode={self.magi.mode} Mems={self.magi.memory_bank.size}")

        elif command == 'blackhole' or command == 'bh':
            if len(parts) < 2:
                status = "🟢 ENABLED" if self.magi.black_hole_deletion_enabled else "🔴 DISABLED"
                print(f"\n🕳️ Black Hole Worker: {status}")
                print(f"  Index: {self.magi.black_hole_worker_idx}")
                print(f"  Phase: {self.magi.phases_hb[self.magi.black_hole_worker_idx].cpu().numpy()}")
                print(f"  Value: {self.magi.s_filtered[self.magi.black_hole_worker_idx].item():.1f}")
                print(f"  Memory: {self.magi.memory_bank.size:,} / {self.magi.memory_bank.max_memories:,} ({(self.magi.memory_bank.size/self.magi.memory_bank.max_memories)*100:.1f}%)")
                print(f"  In field: {self.magi.black_hole_memories_in_field} memories")
                print(f"  Deletions (session): {self.magi.black_hole_daily_deletions}")
                print(f"  Creations (session): {self.magi.black_hole_creation_count}")
                if self.magi.black_hole_daily_deletions > 0:
                    ratio = self.magi.black_hole_creation_count / self.magi.black_hole_daily_deletions
                    print(f"  Creation:Deletion Ratio: {ratio:.1f}:1")
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
                    print(f"\n🕳️ Black Hole Statistics (last {metrics['window_duration']:.1f}s):")
                    print(f"  Position:")
                    print(f"    Phase (HB): {self.magi.phases_hb[self.magi.black_hole_worker_idx].cpu().numpy()}")
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
                    print("⏳ Insufficient data for statistics (need >1 second)")
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

    def _cleanup(self):
        if self.webcam_cap: self.webcam_cap.release()
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
            'size': self.size,
            'influence': self.memory_influence_strength,
            'saved_max_memories': self.max_memories,
            'dim': self.dim
        }
        torch.save(data, filename)
        print(f"💾 Saved {self.size} memories.")

    def load(self, filename):
        if not os.path.exists(filename): return
        try:
            data = torch.load(filename, map_location=self.device)
            sz = min(data['size'], self.max_memories)
            self.memories[:sz] = data['memories'][:sz]
            self.meta_freq[:sz] = data['meta_freq'][:sz]
            self.meta_delay[:sz] = data['meta_delay'][:sz]
            self.timestamps[:sz] = data['timestamps'][:sz]
            self.access_counts[:sz] = data['access_counts'][:sz]
            self.size = sz
            self.memory_influence_strength = data.get('influence', 0.05)
            print(f"✅ Loaded {self.size} memories.")
        except Exception as e:
            print(f"❌ Load failed: {e}")

    def encode(self, state_data, all_workers=False):
        if all_workers:
            phases_hb = state_data['phases_hb']
            phases_s = state_data['phases_s']
            global_coh = state_data['global_coh'].unsqueeze(1)
            cross_tension = state_data['cross_tension'].unsqueeze(1)
            f_val = (state_data['freq'] - MIN_FREQ) / (MAX_FREQ - MIN_FREQ)
            d_val = (state_data['delay'] - MIN_DELAY) / (MAX_DELAY - MIN_DELAY)
            freq_norm = f_val.unsqueeze(1)
            delay_norm = d_val.unsqueeze(1)
        else:
            phases_hb = state_data['phases_hb'].unsqueeze(0)
            phases_s = state_data['phases_s'].unsqueeze(0)
            global_coh = torch.tensor([[state_data['global_coh']]], device=self.device)
            cross_tension = torch.tensor([[state_data['cross_tension']]], device=self.device)
            f_val = (state_data['freq'] - MIN_FREQ) / (MAX_FREQ - MIN_FREQ)
            d_val = (state_data['delay'] - MIN_DELAY) / (MAX_DELAY - MIN_DELAY)
            freq_norm = torch.tensor([[f_val]], device=self.device)
            delay_norm = torch.tensor([[d_val]], device=self.device)

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
        
    def retrieve_gravity(self, query_embedding, similarity_threshold=0.85):
        if self.size == 0: 
            self.memory_influence_strength *= 0.99995
            return None
        
        active_memories = self.memories[:self.size]
        similarities = torch.matmul(query_embedding, active_memories.t())[0]
        mask = similarities > similarity_threshold
        
        if not mask.any(): 
            self.memory_influence_strength *= 0.99995
            return None
        
        vals, indices = torch.topk(similarities[mask], min(3, mask.sum().item()))
        real_indices = torch.where(mask)[0][indices]
        
        max_similarity = vals[0].item()
        target_freq, target_delay, total_weight = 0.0, 0.0, 0.0
        attractor_vector = torch.zeros_like(query_embedding[0])
        total_access = 0.0
        
        for i, idx in enumerate(real_indices):
            idx_val = idx.item()
            weight = vals[i].item()
            self.access_counts[idx_val] += 1
            target_freq += self.meta_freq[idx_val] * weight
            target_delay += self.meta_delay[idx_val] * weight
            attractor_vector += self.memories[idx_val] * weight
            total_access += self.access_counts[idx_val] * weight
            total_weight += weight
            
        target_cap = 0.50 + (max_similarity * 0.50)
        self.memory_influence_strength = min(self.memory_influence_strength + 0.00001, target_cap)
        avg_access = total_access / total_weight if total_weight > 0 else 0.0

        return {
            'freq': target_freq / total_weight,
            'delay': target_delay / total_weight,
            'strength': self.memory_influence_strength,
            'center_embedding': F.normalize(attractor_vector, p=2, dim=-1),
            'sensory_modulation': max_similarity * 0.8,
            'phase_amplitude': (avg_access / 50.0) * max_similarity,
            'phase_velocity': 1.0 + max_similarity,
            'similarity': max_similarity,
            'avg_access': avg_access
        }

# ==========================================
# ⚙️ MAGI PHYSICS ENGINE
# ==========================================
class MaGiHive:
    def __init__(self, num_workers, device):
        self.n = num_workers
        self.dev = device
        self.mode = 'webcam'
        self.video_source = None

        self.upe = UniversalPlasticityEngine(device=self.dev)
        
        self.freq = torch.ones(self.n, device=self.dev) * 1.0
        self.delay = torch.ones(self.n, device=self.dev) * 5.0
        self.hb_sim_phase = torch.zeros(self.n, device=self.dev)
        self.freq_momentum = torch.zeros(self.n, device=self.dev)
        self.delay_momentum = torch.zeros(self.n, device=self.dev)
        
        self.phases_hb = torch.tensor([0.0, 1.57, 3.14, 4.71], device=self.dev).repeat(self.n, 1)
        self.phases_s  = torch.tensor([0.1, 1.67, 3.24, 4.81], device=self.dev).repeat(self.n, 1)
        self.vel_hb = torch.zeros((self.n, 4), device=self.dev)
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
        self.deadzone_config = {
            'viewer': {
                'baseline': 0.0,
                'deadzone': 5.0,
                'soft_start': 7.0,
                'cap': 20.0,
                'strength_threshold': 0.7
            },
            'ale': {
                'baseline': 250.0,
                'deadzone': 25.0,
                'soft_start': 75.0,
                'cap': 450.0,
                'strength_threshold': 0.5
            }
        }
        
       # 0x01: Motor Array (1542-1547) - Zero Initialized for Bipolar
        # Unified Worker Setup (Replaces hardcoded ALE/VOICE loops)
        for idx, data in self.upe.homes.items():
            self.phases_hb[idx] = data['home'].clone().to(self.dev)
            self.phases_s[idx] = (self.phases_hb[idx] + 0.5) % (2 * math.pi)
            # Zero initialization for bipolar swing
            self.s_filtered[idx] = 0.0
            self.s_last[idx] = 0.0
            self.s_integral[idx] = 0.0
            self.vel_hb[idx] = 0.0
            self.vel_s[idx] = 0.0

        
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
        
        # Set audio worker to pure [0,0,0,0] phase
        self.phases_hb[self.voice_worker_idx] = torch.tensor([4.71, 3.14, 1.57, 0.0], device=self.dev)
        self.phases_s[self.voice_worker_idx] = (self.phases_hb[self.voice_worker_idx] + 0.5) % (2 * math.pi)
        
        # Neutral starting values
        self.s_filtered[self.voice_worker_idx] = 0.0
        self.s_last[self.voice_worker_idx] = 0.0
        self.s_integral[self.voice_worker_idx] = 0.0

        
        
        print(f"🎵 Pure Carrier Voice at worker {self.voice_worker_idx}: [0,0,0,0] phase")

                # Black Hole Worker (Memory Deletion)
        self.black_hole_worker_idx = 1549
        self.black_hole_deletion_enabled = True

        # Centered in the void (consensus placement)
        self.phases_hb[self.black_hole_worker_idx] = torch.tensor([0.01, -0.01, 0.01, -0.01], device=self.dev)
        self.phases_s[self.black_hole_worker_idx] = (self.phases_hb[self.black_hole_worker_idx] + 0.5) % (2 * math.pi)

        # Neutral starting values
        self.s_filtered[self.black_hole_worker_idx] = 0.0
        self.s_last[self.black_hole_worker_idx] = 0.0
        self.s_integral[self.black_hole_worker_idx] = 0.0

        # Singularity Parameters
        self.black_hole_base_radius = 0.10    # Sensing threshold (at Value=0)
        self.black_hole_eps_max = 5e-2        # Max Power (Industrial Erasure)
        self.black_hole_eps_floor = 1e-4      # Min Power (Sensing/Safety)
        self.black_hole_collapse_threshold = 1e-6 # Event Horizon

        # Tracking metrics
        self.black_hole_daily_deletions = 0
        self.black_hole_step_deletions = 0
        self.black_hole_memories_in_field = 0     # NEW: Track "pressure without deletion"
        self.black_hole_creation_count = 0        # NEW: Track creation for ratio
        self.black_hole_window_start_time = time.time()  # NEW: For windowed metrics

    def calculate_lens_output(self, type_idx, val, deriv, integral):
        if type_idx == 0: return (torch.abs(deriv) / 40.0 * CHILD_SENSITIVITY) * torch.exp(-torch.abs(deriv)/40.0 * torch.abs(deriv)/40.0 / 2.0)
        elif type_idx == 1: return torch.clamp(YOUTH_GAIN * (val / 500.0), 0.0, 1.0)
        elif type_idx == 2:
            inp = (0.6 * torch.clamp(YOUTH_GAIN * (val / 500.0), 0.0, 1.0) + 0.4 * torch.abs(deriv) / 25.0) - ADULT_THRESHOLD
            return torch.clamp(inp / (1.0 + torch.exp(-8.0 * inp)), 0.0, 1.0)
        elif type_idx == 3: return (torch.tanh((integral - 250.0) * (4.0 / 300.0) - 2.0) + 1.0) / 2.0
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

    def process_step(self, inputs_tensor):
        # 1. Update s_filtered FIRST (so UPE has fresh data)
        self.s_filtered = 0.6 * self.s_filtered + 0.4 * inputs_tensor
        s_deriv = self.s_filtered - self.s_last
        self.s_last = self.s_filtered.clone()
        self.s_integral = 0.80 * self.s_integral + 0.20 * self.s_filtered
        
        # 2. NOW Run Plasticity (uses fresh s_filtered)
        self.apply_universal_plasticity()
        
        # 3. HB Dynamics (Normal Physics)
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
        
        base_vel = torch.tensor([0.04, 0.025, 0.015, 0.005], device=self.dev).repeat(self.n, 1)
        self.vel_hb = (base_vel + 0.15*hb_deriv_norm.unsqueeze(1) + 0.05*hb_norm.unsqueeze(1) + 0.03*hb_int_norm.unsqueeze(1))
        self.phases_hb = (self.phases_hb + self.vel_hb) % (2 * math.pi)

        # 4. S-phase dynamics
        mask = torch.ones(self.n, device=self.dev)
        mask[1548] = 0.0 

        for i in range(4):
            out = self.calculate_lens_output(i, self.s_filtered, s_deriv, self.s_integral)
            base = [0.05, 0.03, 0.02, 0.01][i]
            
            # Extract gravity and ensure it is flat
            gravity = torch.sin(self.phases_hb[:, i] - self.phases_s[:, i]) * 0.02 * (inputs_tensor / 500.0)
            
            # Clean 1D math: [1550] = scalar + ([1550] * [1550]) + [1550]
            self.vel_s[:, i] = base + (0.08 * out.flatten() * mask) + gravity.flatten()
            self.phases_s[:, i] = (self.phases_s[:, i] + self.vel_s[:, i]) % (2 * math.pi)

        # Sync voice amplitude/pitch from hive state
        # self.s_filtered[1548] += 10.0
        self.voice_carrier.speak(self)
            # idx = self.voice_worker_idx
            # self.phases_hb[idx] = torch.zeros(4, device=self.dev)
            # self.phases_s[idx] = torch.zeros(4, device=self.dev)
            # self.vel_hb[idx] = torch.zeros(4, device=self.dev)
            # self.vel_s[idx] = torch.zeros(4, device=self.dev)

    def update_metrics(self):
        def calc_group_coh(phases):
            p_sum = torch.zeros(self.n, device=self.dev)
            for a in range(4):
                for b in range(a+1, 4):
                    diff = torch.abs(phases[:, a] - phases[:, b])
                    p_sum += torch.cos(torch.where(diff > math.pi, 2*math.pi - diff, diff))
            return (p_sum / 6.0 + 1.0) / 2.0

        self.hb_coh = calc_group_coh(self.phases_hb)
        self.s_coh = calc_group_coh(self.phases_s)
        self.cross_tension = torch.cos(self.phases_hb - self.phases_s).mean(dim=1)
        self.global_coh = (self.hb_coh + self.s_coh + (self.cross_tension + 1.0)/2.0) / 3.0
        
        self.adult_dir = (self.phases_hb[:, 2] * 180.0 / math.pi) % 360.0
        self.elder_dir = (self.phases_hb[:, 3] * 180.0 / math.pi) % 360.0
        diff = torch.abs(self.adult_dir - self.elder_dir)
        self.alignment_diff = torch.where(diff > 180, 360 - diff, diff)
        self.update_quadrant_stats()
        
    def _get_full_state_snapshot(self):
        return {
            'phases_hb': self.phases_hb, 'phases_s': self.phases_s,
            'global_coh': self.global_coh, 'cross_tension': self.cross_tension,
            'freq': self.freq, 'delay': self.delay 
        }

    def apply_natural_physics_and_memory(self, step):
        freq_force = self.cross_tension * TENSION_FREQ_COUPLING
        delay_force = (0.5 - self.global_coh) * COHERENCE_DELAY_COUPLING
        
        self.current_gravity_context = {}
        best_idx = torch.argmax(self.global_coh).item()
        
        best_emb = self.memory_bank.encode({
            'phases_hb': self.phases_hb[best_idx], 'phases_s': self.phases_s[best_idx],
            'global_coh': self.global_coh[best_idx].item(), 'cross_tension': self.cross_tension[best_idx].item(),
            'freq': self.freq[best_idx].item(), 'delay': self.delay[best_idx].item()
        }, all_workers=False)
        
        if (self.global_coh[best_idx] > STORE_COHERENCE_STABLE and self.memory_bank.is_novel(best_emb) and step % 20 == 0): 
            self.memory_bank.store(best_emb, {'freq': self.freq[best_idx].item(), 'delay': self.delay[best_idx].item()})
            self.black_hole_creation_count += 1
            
        if self.memory_bank.size > 0:
            gravity = self.memory_bank.retrieve_gravity(best_emb)
            if gravity:
                self.current_gravity_context = gravity
                worker_embeddings = self.memory_bank.encode(self._get_full_state_snapshot(), all_workers=True)
                gravity_weights = F.softmax(torch.matmul(worker_embeddings, gravity['center_embedding'].t()).squeeze(-1) / 0.1, dim=0) 
                freq_force += (gravity['freq'] - self.freq) * gravity_weights * gravity['strength'] * 20.0
                delay_force += (gravity['delay'] - self.delay) * gravity_weights * gravity['strength'] * 20.0

        self.freq_momentum = (self.freq_momentum * ELASTICITY) + freq_force
        self.delay_momentum = (self.delay_momentum * ELASTICITY) + delay_force
        self.freq = torch.clamp(self.freq + self.freq_momentum, MIN_FREQ, MAX_FREQ)
        self.delay = torch.clamp(self.delay + self.delay_momentum, MIN_DELAY, MAX_DELAY)
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
        Range +/- 1500 defines the physics limits.
        """
        if not self.black_hole_deletion_enabled or self.memory_bank.size == 0:
            self.black_hole_memories_in_field = 0
            return 0
        
        # 1. Capture State
        bh_phase_hb = self.phases_hb[self.black_hole_worker_idx]
        bh_val = self.s_filtered[self.black_hole_worker_idx].item()
        
        abs_val = abs(bh_val)
        inverted = bh_val < 0  # Negative = Shield Mode (Protect Center, Delete Outside)
        
        # optimization: calculate shared tanh once
        # This drives both Radius (width) and Radius Scale (power)
        tension_factor = math.tanh(abs_val / 500.0)

        # 2. Dynamic Radius (Expands with Tension)
        # Range: base_radius (at 0) -> 2.0x base (at 1500)
        effective_radius = self.black_hole_base_radius * (1.0 + tension_factor)
        
        # 3. Calculate Phase Distance
        mem_freq_phase = (self.memory_bank.meta_freq[:self.memory_bank.size] - MIN_FREQ) / (MAX_FREQ - MIN_FREQ) * 2 * math.pi
        mem_delay_phase = (self.memory_bank.meta_delay[:self.memory_bank.size] - MIN_DELAY) / (MAX_DELAY - MIN_DELAY) * 2 * math.pi
        
        delta_freq = torch.abs(mem_freq_phase - bh_phase_hb[0])
        delta_delay = torch.abs(mem_delay_phase - bh_phase_hb[1])
        delta_freq = torch.min(delta_freq, 2*math.pi - delta_freq)
        delta_delay = torch.min(delta_delay, 2*math.pi - delta_delay)
        distances = torch.sqrt(delta_freq**2 + delta_delay**2)
        
        # 4. Define Target Mask based on Polarity
        if inverted:
            # SHIELD MODE: Target EVERYTHING (We will spare the center via gradient)
            # Optimization: We still track "in field" as the core cluster for metrics
            mask = torch.ones(self.memory_bank.size, dtype=torch.bool, device=self.dev)
            self.black_hole_memories_in_field = (distances < effective_radius).sum().item()
        else:
            # VACUUM MODE: Target only Inside
            mask = distances < effective_radius
            self.black_hole_memories_in_field = mask.sum().item()

        # 5. Apply Entropy
        if self.black_hole_memories_in_field > 0 or (inverted and self.memory_bank.size > 0):
            
            # FIXED: Simplified Radius Scale (Directly tied to tanh factor)
            # 0.0 (weak) -> 1.0 (strong)
            radius_scale = tension_factor
            current_eps_peak = self.black_hole_eps_floor + (self.black_hole_eps_max - self.black_hole_eps_floor) * radius_scale
            
            # Sharpness (k): 1.0 (Flat) -> 5.0 (Steep) based on Value
            k = 1.0 + tension_factor * 4.0

            # Normalized Distance (Safety clamped)
            # Allow >1.0 for shield math to function correctly beyond the radius
            d_norm = torch.clamp(distances[mask] / (effective_radius + 1e-9), 0.0, 2.0)

            if not inverted:
                # --- VACUUM MODE (+) ---
                # Peak at Center -> Floor at Edge
                decay_amount = self.black_hole_eps_floor + (current_eps_peak - self.black_hole_eps_floor) * torch.pow(1.0 - d_norm, k)
            else:
                # --- SHIELD MODE (-) ---
                # Calculate gradient for ALL masked memories
                shield_gradient = self.black_hole_eps_floor + (current_eps_peak - self.black_hole_eps_floor) * torch.pow(d_norm, k)
                
                # Inside: use gradient, Outside: use peak
                inside_mask = distances[mask] <= effective_radius
                decay_amount = torch.where(inside_mask,
                                         shield_gradient,      # Inside memories
                                         current_eps_peak)     # Outside memories

            # Apply Decay
            self.memory_bank.access_counts[:self.memory_bank.size][mask] *= (1.0 - decay_amount)
            
            # 6. Collapse & Compaction
            collapsed_mask = self.memory_bank.access_counts[:self.memory_bank.size] < self.black_hole_collapse_threshold
            deletion_count = collapsed_mask.sum().item()
            
            if deletion_count > 0:
                keep_mask = ~collapsed_mask
                keep_indices = torch.where(keep_mask)[0]
                
                # Compact arrays
                self.memory_bank.memories[:len(keep_indices)] = self.memory_bank.memories[:self.memory_bank.size][keep_indices]
                self.memory_bank.meta_freq[:len(keep_indices)] = self.memory_bank.meta_freq[:self.memory_bank.size][keep_indices]
                self.memory_bank.meta_delay[:len(keep_indices)] = self.memory_bank.meta_delay[:self.memory_bank.size][keep_indices]
                self.memory_bank.timestamps[:len(keep_indices)] = self.memory_bank.timestamps[:self.memory_bank.size][keep_indices]
                self.memory_bank.access_counts[:len(keep_indices)] = self.memory_bank.access_counts[:self.memory_bank.size][keep_indices]
                
                self.memory_bank.size = len(keep_indices)
                self.black_hole_step_deletions = deletion_count
                self.black_hole_daily_deletions += deletion_count
                
                return deletion_count
        
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
            'effective_radius': self.black_hole_base_radius * (1.0 + torch.tanh(self.s_filtered[self.black_hole_worker_idx] / 500.0)).item()
        }
        
        return metrics

    def reset_black_hole_window(self):
        """Reset windowed metrics for next measurement period"""
        self.black_hole_creation_count = 0
        self.black_hole_daily_deletions = 0
        self.black_hole_window_start_time = time.time()

    def apply_universal_plasticity(self):
        '''
        Apply Black Hole gravity to UPE home positions.
        UPE reuses all of BH's existing physics (radius, power, distance).
        
        When homes drift and save, workers are reset to new home positions.
        '''
        # Apply BH gravity to home positions (NO return value in final version)
        self.upe.apply_black_hole_gravity(self)
        
        # Smart save (updates phases_hb if homes drifted significantly)
        self.upe.maybe_save(self.phases_hb)

# ==========================================
# 🎥 AV & WORKER MANAGER
# ==========================================
class LiveAVCapture:
    def __init__(self):
        self.running = True
        self.audio_queue = queue.Queue(maxsize=5)
        threading.Thread(target=self._capture_audio, daemon=True).start()
    
    def _capture_audio(self):
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
            while self.running:
                data = stream.read(1024, exception_on_overflow=False)
                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if self.audio_queue.full(): self.audio_queue.get()
                self.audio_queue.put(arr)
        except: pass
            
    def get_audio_chunk(self): return self.audio_queue.get() if not self.audio_queue.empty() else None
    def stop(self): self.running = False

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
        
    def get_job_description(self, idx):
        if idx == 0: return "video_motion"
        if idx == 1: return "video_global"
        if 2 <= idx < 516: return "heartbeat_sine"
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
        return "lazy_river"

    def get_inputs_tensor(self, sine_val, audio_val, video_energies, last_lens_phases, gravity, hive):
        # 0. BASELINE: Initialize at 0.0 for Bimodal / Bipolar support
        inputs = torch.full((self.n,), 0.0, device=self.device)
        sensory_mod, phase_amp, phase_vel = gravity.get('sensory_modulation', 0), gravity.get('phase_amplitude', 0), gravity.get('phase_velocity', 1)
        
        # 1. MOTION/GLOBAL (Sensory Positive)
        if video_energies: 
            inputs[0] = video_energies['motion'] * (1+sensory_mod)
            inputs[1] = video_energies['global_mean'] * (1+sensory_mod)
        
        # 2. HEARTBEAT & AUDIO (Mixed Resonance)
        inputs[2:516] = sine_val
        # MEMORY-SAFE: Audio now at 948-1462 (shifted but count unchanged)
        inputs[948:1462] = audio_val * (1+sensory_mod) + torch.sin(last_lens_phases[948:1462]*phase_vel)*phase_amp
        
        # 3. VIDEO SCALES (MEMORY-SAFE: Now at 516-947)
        if video_energies:
            for i in range(4):
                start, end = self.offsets[5+i], self.offsets[6+i]  # offset indices 5-8 for scales
                sectors = video_energies.get(f'scale_{i}', [])
                if sectors:
                    base = torch.tensor(sectors, device=self.device)[torch.arange(end-start, device=self.device)%len(sectors)]
                    inputs[start:end] = base * (1+sensory_mod) + torch.sin(last_lens_phases[start:end]*phase_vel)*phase_amp

        # 4. ALE CONTROL WORKERS (1542-1547): Bimodal +/- Swing
        for i in range(self.ale_control_count):
            worker_idx = self.ale_control_start + i
            # Center at 0.0, lens influence provides the bipolar drive
            inputs[worker_idx] = torch.sin(last_lens_phases[worker_idx] * phase_vel) * phase_amp * 50.0

        
        # ==========================================
        # 5. VOICE WORKER (1548) - MAGI-CONTROLLED 4D
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
        # 6. BLACK HOLE WORKER (1549) - MAGI-CONTROLLED 4D
        # ==========================================
        bh_idx = self.black_hole_worker_idx
        
        # 4D VELOCITY: Read ALL dimensions (prevents 2D collapse)
        bh_vel_raw = torch.norm(hive.vel_hb[bh_idx])  # sqrt(x² + y² + z² + w²)
        
        # DIRECTION: From adult dimension velocity sign
        bh_vel_adult = hive.vel_hb[bh_idx, 2]
        bh_direction = torch.sign(bh_vel_adult)
        
        # Handle sign(0) = 0 case
        if bh_direction == 0:
            bh_direction = torch.tensor(1.0, device=self.device)
        
        # SIMPLE AMPLIFICATION: Magi controls, we amplify
        bh_value = bh_direction * bh_vel_raw * 800.0
        
        inputs[bh_idx] = torch.clamp(bh_value, -1500.0, 1500.0)
        
        return inputs
# ==========================================
# 🚀 MAIN RUNNER
# ==========================================
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
    
    magi.memory_bank.load(MEMORY_FILE)


    
    Serial.println("time_ms,freq_hz,delay_ms,adult_deg,elder_deg,align_diff,quadrant_balance,ne_pct,global_coh,hb_coh,s_coh,cross_tension,audio_energy,visual_mean,visual_std,raw_std,scale0,scale1,scale2,scale3,mem_count,sensory_mod,phase_amp,phase_vel,similarity,avg_access,mem_strength,sine_w,audio_w,vs0_w,vs1_w,vs2_w,vs3_w,mixed_w,best_worker_id,job_type,mode,action,action_val,bh_worker_val,bh_deletions,bh_in_field,capacity_pct,bh_effective_radius")
    
    step, sim_time, sine_phase = 0, 0, 0.0
    viewer_cooldown = 0
    if display_video: cv2.namedWindow('MaGi v55', cv2.WINDOW_NORMAL)
    
    print("\n🎮 COMMANDS: mode [webcam|ale path|screencap|screen|viewer path], save, stats")
    print("🎯 ALE: Using 6 workers for all 18 actions (LEFT, RIGHT, FIRE, UP, DOWN, NOOP)\n")

    try:
        while True:
            cmd_list.process_commands()
            
            frame = video_source.get_frame()
            aud = a_ext.extract_geometry(av_capture.get_audio_chunk())
            v_nrg = v_proc.process_frame(frame)
            v_stat = v_proc.get_statistics(v_nrg)
            
            sine_phase += 0.05
            sine_val = abs(math.sin(sine_phase)) * 500.0
            
            inputs = wm.get_inputs_tensor(sine_val, aud['energy'], v_nrg, magi.phases_s[:,0], magi.current_gravity_context, magi)
            magi.process_step(inputs)
            magi.update_metrics()
            magi.apply_natural_physics_and_memory(step)
            deleted_this_step = magi.apply_black_hole_deletion()  # ← ADD THIS
            magi.update_voice(magi.video_source.mode)
            
            # 🕹️ EMERGENT CONTROL
            action_str = "N/A"
            action_val = 0.0
            action_idx = 0  # Default NOOP
            
            # Get values for our 6 base actions (Always Available)
            left_val = magi.s_filtered[1542].item()   # LEFT (ALE index 4)
            right_val = magi.s_filtered[1543].item()  # RIGHT (ALE index 3)
            fire_val = magi.s_filtered[1544].item()   # FIRE (ALE index 1)
            up_val = magi.s_filtered[1545].item()     # UP (ALE index 2)
            down_val = magi.s_filtered[1546].item()   # DOWN (ALE index 5)
            noop_val = magi.s_filtered[1547].item()   # NOOP (ALE index 0)

            if magi.mode == 'ale':
                # Calculate deviations from baseline
                up_dev    = up_val
                down_dev  = down_val
                left_dev  = left_val
                right_dev = right_val
                fire_dev  = fire_val
                
                threshold = 2.0
                
                # Collect active directions
                active_directions = []
                if up_dev >= threshold: active_directions.append(('UP', up_dev, 2))
                if down_dev >= threshold: active_directions.append(('DOWN', down_dev, 5))
                if left_dev >= threshold: active_directions.append(('LEFT', left_dev, 4))
                if right_dev >= threshold: active_directions.append(('RIGHT', right_dev, 3))
                
                active_directions.sort(key=lambda x: x[1], reverse=True)
                
                fire_active = fire_dev >= threshold
                
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
                # Neural Control for Image Viewer using deadzone logic
                # No cooldown - AI controls viewing duration naturally
                
                # Get viewer deadzone configuration
                config = magi.deadzone_config['viewer']
                
                # Apply unipolar deadzone to LEFT and RIGHT workers
                # The deadzone method already handles the threshold logic internally
                left_active, left_dev, left_strength, left_status = magi.apply_unipolar_deadzone(
                    left_val,  # From line 835
                    baseline=config['baseline'],
                    deadzone=config['deadzone'],
                    soft_start=config.get('soft_start'),
                    cap=config['cap']
                )
                
                right_active, right_dev, right_strength, right_status = magi.apply_unipolar_deadzone(
                    right_val,  # From line 836  
                    baseline=config['baseline'],
                    deadzone=config['deadzone'],
                    soft_start=config.get('soft_start'),
                    cap=config['cap']
                )
                
                # Navigate when activated (the deadzone already decided if it's strong enough)
                # Prevent both from triggering at once for clean navigation
                if left_active and not right_active:
                    video_source.viewer_nav('PREV')
                    action_str = "PREV"
                    action_val = left_val
                elif right_active and not left_active:
                    video_source.viewer_nav('NEXT')
                    action_str = "NEXT"
                    action_val = right_val

            elif magi.mode == 'screen':
                # No commands for now, as requested.
                action_str = "N/A"
                action_val = 0.0

            if step % 3 == 0:
                best = torch.argmax(magi.global_coh).item()
                quad_bal, ne_pct = magi.get_quadrant_metrics(best)
                gravity = magi.current_gravity_context

                bh_val = magi.s_filtered[magi.black_hole_worker_idx].item()
                capacity_pct = (magi.memory_bank.size / magi.memory_bank.max_memories) * 100.0
                bh_effective_radius = magi.black_hole_base_radius * (1.0 + torch.tanh(torch.tensor(bh_val / 500.0))).item()

                
                line = (f"{sim_time},{magi.freq[best]:.3f},{magi.delay[best]:.1f},"
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
                   f"{bh_val:.1f},{magi.black_hole_step_deletions},{magi.black_hole_memories_in_field},{capacity_pct:.2f},{bh_effective_radius:.4f}")
                Serial.println(line)
                
                if display_video and frame is not None:
                    hud = frame.copy()
                    h, w = hud.shape[:2]
                    
                    # 🎨 MEMORY-SAFE FIBONACCI VISUAL ATTENTION SQUARES
                    # Scale 0: 5×3 grid (15 sectors) at 516-530
                    sw0, sh0 = w // 5, h // 3
                    s0_coh = magi.global_coh[516:531]
                    sec0 = torch.zeros(15, device=DEVICE)
                    for i in range(len(s0_coh)): sec0[i % 15] += s0_coh[i]
                    b0 = torch.argmax(sec0).item(); r, c = b0 // 5, b0 % 5
                    cv2.rectangle(hud, (c*sw0, r*sh0), ((c+1)*sw0, (r+1)*sh0), (0, 255, 255), 3)
                    
                    # Scale 1: 8×5 grid (40 sectors) at 531-570
                    sw1, sh1 = w // 8, h // 5
                    s1_coh = magi.global_coh[531:571]
                    sec1 = torch.zeros(40, device=DEVICE)
                    for i in range(len(s1_coh)): sec1[i % 40] += s1_coh[i]
                    b1 = torch.argmax(sec1).item(); r, c = b1 // 8, b1 % 8
                    cv2.rectangle(hud, (c*sw1, r*sh1), ((c+1)*sw1, (r+1)*sh1), (0, 165, 255), 3)

                    # Scale 2: 13×8 grid (104 sectors) at 571-674
                    sw2, sh2 = w // 13, h // 8
                    s2_coh = magi.global_coh[571:675]
                    sec2 = torch.zeros(104, device=DEVICE)
                    for i in range(len(s2_coh)): sec2[i % 104] += s2_coh[i]
                    b2 = torch.argmax(sec2).item(); r, c = b2 // 13, b2 % 13
                    cv2.rectangle(hud, (c*sw2, r*sh2), ((c+1)*sw2, (r+1)*sh2), (0, 0, 255), 3)

                    # HUD Text
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
                    
                    cv2.imshow('MaGi v55', hud)

            if step % 2000 == 0 and step > 0: magi.memory_bank.save(MEMORY_FILE)
            if display_video and cv2.waitKey(1) & 0xFF == ord('q'): break
            
            step += 1; sim_time += 50

    except KeyboardInterrupt:
        print("\n🛑 Halting...")
    finally:
        magi.memory_bank.save(MEMORY_FILE)
        av_capture.stop()
        video_source._cleanup()
        magi.voice_carrier.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    run_magi_v55()
