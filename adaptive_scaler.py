# ==========================================
# 📊 ADAPTIVE SCALER v2 - Steps & Range Modes
# ==========================================

import time
import numpy as np

class AdaptiveScaler:
    """
    Adaptive range scaling with φ (Golden Ratio) spacing.
    
    TWO OPERATION MODES:
    
    1. STEPS MODE (like Voice):
       - Maps input to discrete brackets (0 to n_steps-1)
       - Each bracket has threshold values (φ or linear spaced)
       - Returns bracket index + continuous position within bracket
       
    2. RANGE MODE (like Black Hole):
       - Maps input to continuous output range
       - input_range_min/max define expected input range
       - output_range_min/max define target output range
       - Scaling can be linear or φ-spaced
    
    FEATURES:
    - Signal tracking with decay (like voice tracker)
    - Configurable dead zones (φ or linear)
    - No damping - pure scaling (MaGi provides movement)
    """
    
    def __init__(self,
                 name="scaler",
                 
                 # Mode Selection
                 mode='steps',              # 'steps' or 'range'
                 
                 # STEPS MODE Parameters (like Voice)
                 n_steps=16,                 # Number of discrete steps/brackets
                 steps_are_bipolar=False,    # If True, steps are -N/2 to +N/2
                 
                 # RANGE MODE Parameters (like Black Hole)
                 input_range_min=0.0,        # Min expected input (absolute)
                 input_range_max=100.0,       # Max expected input (absolute)
                 output_range_min=0.0,        # Min output value
                 output_range_max=1500.0,     # Max output value
                 
                 # Spacing Configuration
                 spacing='phi_inverse',       # 'linear', 'phi', 'phi_inverse'
                 
                 # Range Tracking
                 min_threshold=1.5,           # Below this = inactive
                 initial_max=12.0,             # Starting assumption for max
                 track_signal=True,            # Enable signal tracking with decay
                 track_window_seconds=60.0,    # How long to remember (seconds)
                 track_decay_rate=0.9995,      # Per-step decay when signal drops
                 track_baseline=12.0,          # Baseline to decay toward
                 
                 # Dead Zone Configuration
                 dead_zone_mode='none',        # 'none', 'phi', 'linear'
                 dead_zone_value=0.382,        # For linear: 0-1 proportion, for phi: multiplier
                 
                 # Output Configuration
                 bipolar=True,                  # If True, output preserves sign
                 clip_output=True):              # Clip output to defined ranges
        
        self.name = name
        self.mode = mode
        self.spacing = spacing
        self.bipolar = bipolar
        self.clip_output = clip_output
        self.phi = 1.618033988749895
        
        # ==========================================
        # STEPS MODE Configuration
        # ==========================================
        self.n_steps = n_steps
        self.steps_are_bipolar = steps_are_bipolar
        
        # For steps mode, generate bracket names if needed
        if steps_are_bipolar:
            half = n_steps // 2
            self.step_names = [f"step_{i-half}" for i in range(n_steps)]
        else:
            self.step_names = [f"step_{i}" for i in range(n_steps)]
        
        # ==========================================
        # RANGE MODE Configuration
        # ==========================================
        self.input_range_min = input_range_min
        self.input_range_max = input_range_max
        self.output_range_min = output_range_min
        self.output_range_max = output_range_max
        
        # ==========================================
        # RANGE TRACKING
        # ==========================================
        self.min_threshold = min_threshold
        self.observed_max = initial_max
        self.absolute_max = initial_max
        
        # ==========================================
        # SIGNAL TRACKING (Like Voice Tracker)
        # ==========================================
        self.track_signal = track_signal
        self.track_window_seconds = track_window_seconds
        self.track_decay_rate = track_decay_rate
        self.track_baseline = track_baseline
        
        self.signal_history = []          # (timestamp, magnitude)
        self.tracked_max = initial_max
        self.last_signal_time = time.time()
        self.signal_strength = 0.0
        
        # ==========================================
        # DEAD ZONE
        # ==========================================
        self.dead_zone_mode = dead_zone_mode
        self.dead_zone_value = dead_zone_value
        self._dead_zone_threshold = self._calculate_dead_zone()
        
        # ==========================================
        # THRESHOLDS (for steps mode)
        # ==========================================
        self._thresholds = []
        self._update_thresholds()
        
        # ==========================================
        # MINIMAL STATS (just enough for monitoring)
        # ==========================================
        self.expansion_count = 0      # How many times range expanded
        self.decay_count = 0          # How many decay events occurred
        self.active_count = 0         # How many frames were active
        
        self._print_init()
    
    def _print_init(self):
        """Print initialization info"""
        print(f"\n📊 AdaptiveScaler[{self.name}]: {self.mode.upper()} mode")
        
        if self.mode == 'steps':
            print(f"   Steps: {self.n_steps} ({'bipolar' if self.steps_are_bipolar else 'unipolar'})")
            print(f"   Spacing: {self.spacing}")
        else:
            print(f"   Range: [{self.input_range_min:.1f}→{self.input_range_max:.1f}] → "
                  f"[{self.output_range_min:.1f}→{self.output_range_max:.1f}]")
            print(f"   Scaling: {self.spacing}")
        
        print(f"   Signal Tracking: {'ON' if self.track_signal else 'OFF'} "
              f"(decay={self.track_decay_rate})")
        print(f"   Dead Zone: {self.dead_zone_mode} (threshold: {self._dead_zone_threshold:.3f})")
    
    # ==========================================
    # PRIVATE METHODS
    # ==========================================
    
    def _calculate_dead_zone(self):
        """Calculate absolute dead zone threshold"""
        if self.dead_zone_mode == 'none':
            return self.min_threshold
        
        range_size = self.observed_max - self.min_threshold
        
        if self.dead_zone_mode == 'linear':
            return self.min_threshold + (range_size * self.dead_zone_value)
        
        elif self.dead_zone_mode == 'phi':
            # First bracket scaled by dead_zone_value
            total_ratio = self.observed_max / self.min_threshold
            phi_scaled = self.observed_max / (total_ratio ** self.dead_zone_value)
            return max(self.min_threshold, phi_scaled)
        
        elif self.dead_zone_mode == 'phi_inverse':
            # Large first bracket
            return self.min_threshold + (range_size * self.dead_zone_value)
        
        return self.min_threshold
    
    def _update_thresholds(self):
        """Generate thresholds for steps mode"""
        if self.mode != 'steps':
            return
        
        a = self.min_threshold
        b = self.observed_max
        n = self.n_steps
        
        if n <= 1 or b <= a:
            self._thresholds = [a, b] if n == 2 else [a]
            return
        
        if self.spacing == 'linear':
            # Linear spacing
            self._thresholds = [a + (b - a) * i / (n - 1) for i in range(n)]
            
        elif self.spacing == 'phi':
            # φ spacing: small gaps at start, large at end
            total_ratio = b / a
            ratio = total_ratio ** (1.0 / (n - 1))
            
            thresholds = [a]
            current = a
            for i in range(n - 1):
                current = min(b, current * ratio)
                thresholds.append(current)
            self._thresholds = thresholds
            
        elif self.spacing == 'phi_inverse':
            # Inverse φ: large gaps at start, small at end (voice-like)
            total_range = b - a
            total_ratio = b / a
            r = 1.0 / (total_ratio ** (1.0 / (n - 1)))
            
            if abs(r - 1.0) < 1e-10:
                gaps = [total_range / (n - 1)] * (n - 1)
            else:
                sum_series = (1 - r ** (n - 1)) / (1 - r)
                first_gap = total_range / sum_series
                
                gaps = []
                current_gap = first_gap
                for i in range(n - 1):
                    gaps.append(current_gap)
                    current_gap *= r
            
            thresholds = [a]
            current = a
            for gap in gaps:
                current += gap
                thresholds.append(min(current, b))
            
            thresholds[-1] = b
            self._thresholds = thresholds
    
    def _update_signal_tracking(self, magnitude, current_time=None):
        """Update signal tracking with decay (like voice tracker)"""
        if not self.track_signal:
            return
        
        if current_time is None:
            current_time = time.time()
        
        # Add to history
        self.signal_history.append((current_time, magnitude))
        
        # Prune old history
        cutoff = current_time - self.track_window_seconds
        self.signal_history = [(t, m) for t, m in self.signal_history if t > cutoff]
        
        # Calculate recent max
        recent_max = max([m for _, m in self.signal_history]) if self.signal_history else self.track_baseline
        
        # Apply decay if signal dropped
        if magnitude < self.tracked_max:
            target = max(recent_max, self.track_baseline)
            self.tracked_max = self.tracked_max * self.track_decay_rate + target * (1 - self.track_decay_rate)
            self.decay_count += 1
        else:
            self.tracked_max = max(self.tracked_max, magnitude)
        
        # Update observed_max for threshold calculations
        self.observed_max = self.tracked_max
        
        # Calculate signal strength
        if self.tracked_max > 0:
            self.signal_strength = magnitude / self.tracked_max
        else:
            self.signal_strength = 0.0
    
    # ==========================================
    # PUBLIC METHODS
    # ==========================================
    
    def update_range(self, value, current_time=None):
        """Update observed maximum"""
        magnitude = abs(value)
        
        old_max = self.observed_max
        expanded = False
        
        if self.track_signal:
            self._update_signal_tracking(magnitude, current_time)
            if magnitude > old_max:
                expanded = True
                self.expansion_count += 1
                self.absolute_max = max(self.absolute_max, magnitude)
        else:
            if magnitude > self.observed_max:
                self.observed_max = magnitude
                self.absolute_max = max(self.absolute_max, magnitude)
                expanded = True
                self.expansion_count += 1
        
        # Recalculate if expanded
        if expanded:
            self._update_thresholds()
            self._dead_zone_threshold = self._calculate_dead_zone()
        
        return expanded, old_max, self.observed_max
    
    def apply_dead_zone(self, value):
        """Apply dead zone, return (processed, is_active, status)"""
        magnitude = abs(value)
        
        if magnitude < self.min_threshold:
            return 0.0, False, 'silent'
        
        if magnitude < self._dead_zone_threshold:
            if self.dead_zone_mode == 'none':
                return value, True, 'active'
            return 0.0, False, 'dead_zone'
        
        return value, True, 'active'
    
    def process_steps_mode(self, value, is_active):
        """Process in STEPS mode (like Voice)"""
        if not is_active:
            return {
                'step_index': 0,
                'step_name': self.step_names[0],
                'position': 0.0,
                'continuous_value': 0.0
            }
        
        magnitude = abs(value)
        sign = 1.0 if value >= 0 else -1.0
        
        # Clamp to range
        v = max(self.min_threshold, min(magnitude, self.observed_max))
        
        # Find step
        for i in range(len(self._thresholds) - 1):
            t_start = self._thresholds[i]
            t_end = self._thresholds[i + 1]
            
            if t_start <= v < t_end:
                # Continuous position within step (0-1)
                if t_end > t_start:
                    pos_in_step = (v - t_start) / (t_end - t_start)
                else:
                    pos_in_step = 0.0
                
                # Step index (0 to n_steps-1)
                step_index = i
                
                # For bipolar steps, adjust index
                if self.steps_are_bipolar:
                    half = self.n_steps // 2
                    display_index = step_index - half
                else:
                    display_index = step_index
                
                step_name = self.step_names[step_index] if step_index < len(self.step_names) else f"step_{step_index}"
                
                # Continuous value preserves sign
                continuous = (step_index + pos_in_step) / (self.n_steps - 1)
                if self.bipolar:
                    continuous = continuous * 2 - 1  # Map 0-1 to -1 to 1
                    continuous *= sign
                
                return {
                    'step_index': step_index,
                    'display_index': display_index if self.steps_are_bipolar else step_index,
                    'step_name': step_name,
                    'position_in_step': pos_in_step,
                    'position': (step_index + pos_in_step) / (self.n_steps - 1),
                    'continuous_value': continuous
                }
        
        # At max
        last_idx = self.n_steps - 1
        return {
            'step_index': last_idx,
            'display_index': last_idx - (self.n_steps // 2) if self.steps_are_bipolar else last_idx,
            'step_name': self.step_names[last_idx],
            'position_in_step': 1.0,
            'position': 1.0,
            'continuous_value': sign if self.bipolar else 1.0
        }
    
    def process_range_mode(self, value, is_active):
        """Process in RANGE mode (like Black Hole)"""
        if not is_active:
            return 0.0
        
        magnitude = abs(value)
        sign = 1.0 if value >= 0 else -1.0
        
        # Normalize input to 0-1 based on configured range
        input_range = self.input_range_max - self.input_range_min
        if input_range > 0:
            t = (magnitude - self.input_range_min) / input_range
            t = max(0.0, min(1.0, t))
        else:
            t = 1.0
        
        # Apply spacing transformation
        if self.spacing == 'linear':
            scaled_t = t
        elif self.spacing == 'phi':
            # φ spacing: compress low, expand high
            scaled_t = t ** self.phi
        elif self.spacing == 'phi_inverse':
            # Inverse φ: expand low, compress high (voice-like)
            scaled_t = t ** (1 / self.phi)
        else:
            scaled_t = t
        
        # Map to output range
        output_range = self.output_range_max - self.output_range_min
        output = self.output_range_min + (scaled_t * output_range)
        
        # Apply sign for bipolar output
        if self.bipolar:
            output *= sign
        
        # Clip if requested
        if self.clip_output:
            if self.bipolar:
                output = max(-self.output_range_max, min(self.output_range_max, output))
            else:
                output = max(self.output_range_min, min(self.output_range_max, output))
        
        return output
    
    def process(self, value, current_time=None):
        """
        Main entry point: Process value through scaler.
        
        Returns appropriate output based on mode:
        - steps mode: dict with step info
        - range mode: scaled value
        """
        # Update range tracking
        expanded, old_max, current_max = self.update_range(value, current_time)
        
        # Apply dead zone
        processed, is_active, zone_status = self.apply_dead_zone(value)
        
        if is_active:
            self.active_count += 1
        
        # Process based on mode
        if self.mode == 'steps':
            output = self.process_steps_mode(processed, is_active)
        else:  # range mode
            output = self.process_range_mode(processed, is_active)
        
        # Build result dict
        result = {
            'input': value,
            'output': output,
            'is_active': is_active,
            'zone_status': zone_status,
            'expanded': expanded,
            'observed_max': self.observed_max,
            'tracked_max': self.tracked_max if self.track_signal else self.observed_max,
            'signal_strength': self.signal_strength,
            'dead_zone_threshold': self._dead_zone_threshold,
            'mode': self.mode
        }
        
        # Add mode-specific info
        if self.mode == 'steps' and isinstance(output, dict):
            result.update(output)
        
        return result
    
    def get_info(self):
        """Get current configuration and state (minimal stats)"""
        info = {
            'name': self.name,
            'mode': self.mode,
            'spacing': self.spacing,
            'bipolar': self.bipolar,
            'range': {
                'min': self.min_threshold,
                'max': self.observed_max,
                'absolute_max': self.absolute_max
            },
            'dead_zone': {
                'mode': self.dead_zone_mode,
                'threshold': self._dead_zone_threshold,
                'value': self.dead_zone_value
            },
            'signal_tracking': {
                'enabled': self.track_signal,
                'window': self.track_window_seconds,
                'decay_rate': self.track_decay_rate,
                'current_strength': self.signal_strength
            },
            'stats': {
                'expansions': self.expansion_count,
                'decays': self.decay_count,
                'active_frames': self.active_count
            }
        }
        
        if self.mode == 'steps':
            info['steps'] = {
                'n_steps': self.n_steps,
                'bipolar': self.steps_are_bipolar,
                'thresholds': self._thresholds.copy()
            }
        else:
            info['range_mapping'] = {
                'input': [self.input_range_min, self.input_range_max],
                'output': [self.output_range_min, self.output_range_max]
            }
        
        return info


# # ==========================================
# # 🎯 WORKER INTEGRATION EXAMPLES
# # ==========================================

# def demonstrate_adaptive_scaler():
#     """Demonstrate both steps and range modes"""
    
#     print("\n" + "="*70)
#     print("ADAPTIVE SCALER v2 - STEPS & RANGE MODES")
#     print("="*70)
    
#     # ==========================================
#     # EXAMPLE 1: Voice (Steps Mode, φ_inverse)
#     # ==========================================
#     print("\n🔊 VOICE WORKER (Steps Mode, φ_inverse)")
#     voice = AdaptiveScaler(
#         name="voice",
#         mode='steps',
#         n_steps=16,
#         steps_are_bipolar=False,
#         spacing='phi_inverse',
#         min_threshold=1.5,
#         initial_max=12.0,
#         track_signal=True,
#         track_decay_rate=0.9995,
#         dead_zone_mode='phi_inverse',
#         dead_zone_value=0.382
#     )
    
#     test_values = [0.5, 2.0, 4.0, 6.0, 8.0, 12.0, 15.0, 10.0, 3.0]
#     for val in test_values:
#         result = voice.process(val)
#         if result['is_active']:
#             print(f"  Input {val:5.1f} → Step {result['step_index']:2d} "
#                   f"({result['step_name']}) pos={result['position_in_step']:.2f}")
#         else:
#             print(f"  Input {val:5.1f} → {result['zone_status'].upper()}")
    
#     # ==========================================
#     # EXAMPLE 2: Black Hole (Range Mode, φ)
#     # ==========================================
#     print("\n🕳️ BLACK HOLE (Range Mode, φ scaling)")
#     bh = AdaptiveScaler(
#         name="black_hole",
#         mode='range',
#         input_range_min=10.0,
#         input_range_max=1500.0,
#         output_range_min=1e-4,
#         output_range_max=5e-2,
#         spacing='phi',
#         bipolar=False,  # Magnitude only
#         min_threshold=10.0,
#         initial_max=100.0,
#         track_signal=True,
#         track_decay_rate=0.999,
#         dead_zone_mode='phi',
#         dead_zone_value=0.382
#     )
    
#     for val in [5, 50, 200, 500, 1000, 1500, 2000]:
#         result = bh.process(val)
#         if result['is_active']:
#             eps = result['output']
#             print(f"  Input {val:5.0f} → ε={eps:.6f} "
#                   f"[{result['zone_status']}]")
#         else:
#             print(f"  Input {val:5.0f} → {result['zone_status'].upper()}")
    
#     # ==========================================
#     # EXAMPLE 3: Bridge Worker (Range Mode, φ_inverse)
#     # ==========================================
#     print("\n🌉 BRIDGE WORKER (Range Mode, φ_inverse)")
#     bridge = AdaptiveScaler(
#         name="bridge",
#         mode='range',
#         input_range_min=5.0,
#         input_range_max=100.0,
#         output_range_min=-1.0,
#         output_range_max=1.0,
#         spacing='phi_inverse',
#         bipolar=True,
#         min_threshold=5.0,
#         initial_max=50.0,
#         track_signal=True,
#         track_window_seconds=30.0,
#         track_decay_rate=0.995
#     )
    
#     for val in [-2, -10, -30, -80, -50, -20, -5, 0, 5, 20, 50, 80, 100, 120]:
#         result = bridge.process(val)
#         print(f"  Input {val:4.0f} → Output {result['output']:6.3f} "
#               f"[sig={result['signal_strength']:.2f}]")
    
#     # ==========================================
#     # EXAMPLE 4: ALE Fire (Steps Mode, Linear)
#     # ==========================================
#     print("\n🔥 ALE FIRE (Steps Mode, Linear)")
#     fire = AdaptiveScaler(
#         name="ale_fire",
#         mode='steps',
#         n_steps=5,
#         steps_are_bipolar=False,
#         spacing='linear',
#         min_threshold=100.0,
#         initial_max=500.0,
#         dead_zone_mode='linear',
#         dead_zone_value=0.20,  # 20% dead zone
#         bipolar=False  # Unipolar output
#     )
    
#     for val in [50, 150, 250, 350, 450, 500, 600]:
#         result = fire.process(val)
#         if result['is_active']:
#             strength = (result['step_index'] / 4) * 100
#             print(f"  Input {val:4.0f} → Strength {strength:3.0f}% "
#                   f"(step {result['step_index']})")
#         else:
#             print(f"  Input {val:4.0f} → {result['zone_status'].upper()}")
    
#     # Show stats for bridge worker
#     print("\n📊 BRIDGE STATISTICS:")
#     info = bridge.get_info()
#     print(f"  Updates: {info['stats']['updates']}")
#     print(f"  Expansions: {info['stats']['expansions']}")
#     print(f"  Decay events: {info['stats']['decay_events']}")
#     print(f"  Dead zone hits: {info['stats']['dead_zone_hits']}")
#     print(f"  Active frames: {info['stats']['active_frames']}")


# if __name__ == "__main__":
#     demonstrate_adaptive_scaler()