"""
SO-101 Real Robot Control via UDP
=================================
Drop-in replacement for the Genesis simulation script.

Commands (JSON over UDP to 127.0.0.1:5005):
  {"mode": "pos", "pos": [x, y, z]}
  {"mode": "pos", "pos": [x, y, z], "wrist_roll": 0.5, "wrist_flex": -0.3}
  {"mode": "gripper", "state": "open"}
  {"mode": "gripper", "state": "close"}
  {"mode": "stop"}              <- emergency stop
  {"mode": "reset"}             <- re-enable after cutoff
"""

import socket
import json
import threading
import time
import math
import sys
import pathlib
import numpy as np
import ikpy.chain

# ---------------------------------------------------------------------------
# lerobot imports
# ---------------------------------------------------------------------------
try:
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    FollowerClass = SOFollower
    ConfigClass = SOFollowerRobotConfig
except ImportError:
    try:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        FollowerClass = SO101Follower
        ConfigClass = SO101FollowerConfig
    except ImportError:
        try:
            from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig
            FollowerClass = SO101Follower
            ConfigClass = SO101FollowerConfig
        except ImportError:
            print("ERROR: Could not import lerobot SO follower classes.")
            sys.exit(1)

# ===========================================================================
# Configuration
# ===========================================================================
ROBOT_PORT = "COM10"
ROBOT_ID   = None           # matches calibration filename (None → None.json)

UDP_IP     = "127.0.0.1"
UDP_PORT   = 5005
BUFFER_SIZE = 1024
CONTROL_HZ  = 50

# Workspace scale — MaGi ±0.5m → real arm ~±0.25m
WORKSPACE_SCALE = 0.5

# Arm speed limit (degrees per second, 0 = no limit / full servo speed)
ARM_SPEED = 100

# Gripper uses 0-100 scale (lerobot percentage, NOT degrees)
GRIPPER_OPEN  = 100.0
GRIPPER_CLOSE = 0.0

# Load/temp safety
LOAD_THRESHOLD  = 850
LOAD_TIME_LIMIT = 3.0
LOAD_CHECK_HZ   = 10
TEMP_WARN       = 75
TEMP_PAUSE      = 90
COOLDOWN_TIME   = 2.0

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

# Per-motor enable — set False to disable
MOTOR_ENABLED = {
    "shoulder_pan":  True,
    "shoulder_lift": True,
    "elbow_flex":    True,
    "wrist_flex":    True,
    "wrist_roll":    True,
    "gripper":       True,
}

# ===========================================================================
# IK — ikpy numerical solver using actual URDF geometry
# ===========================================================================
#
# Chain: base_link → shoulder_pan → shoulder_link → shoulder_lift →
#        upper_arm_link → elbow_flex → lower_arm_link → wrist_flex →
#        wrist_link → wrist_roll → gripper_link → gripper_frame_joint(fixed) →
#        gripper_frame_link  (the true end-effector tip)
#
# active_links_mask: 7 entries, one per link in the chain.
# [base, pan, lift, elbow, wflex, wroll, gripper_frame(fixed)]
# [False, True, True, True, True, True, False]
#
URDF_PATH = "so101_new_calib.urdf"

# Joint limits from URDF (radians), used to clamp solver output
_LIMITS = [
    None,                    # index 0: base_link (inactive)
    (-1.91986,  1.91986),   # index 1: shoulder_pan
    (-1.74533,  1.74533),   # index 2: shoulder_lift
    (-1.69,     1.69),      # index 3: elbow_flex
    (-1.65806,  1.65806),   # index 4: wrist_flex
    (-2.74385,  2.84121),   # index 5: wrist_roll
    None,                    # index 6: gripper_frame (fixed, always 0)
]

def _load_chain(urdf_path):
    """Load the ikpy chain once. Tries the given path and a few fallbacks."""
    candidates = [
        urdf_path,
        "so101/so101_new_calib.urdf",
        "SO101/so101_new_calib.urdf",
    ]
    for p in candidates:
        if pathlib.Path(p).exists():
            chain = ikpy.chain.Chain.from_urdf_file(
                p,
                base_elements=["base_link"],
                active_links_mask=[False, True, True, True, True, True, False],
            )
            print(f"IK chain loaded from: {p}")
            return chain
    print("ERROR: URDF not found for IK chain. Tried:")
    for p in candidates:
        print(f"  {p}")
    sys.exit(1)

_chain = _load_chain(URDF_PATH)


def solve_ik(x, y, z, current_obs=None):
    """
    Cartesian position (metres, already workspace-scaled) →
    joint angles dict (degrees, ready for lerobot send_action).

    current_obs: dict from robot.get_observation(), keys like 'shoulder_pan.pos'
                 in degrees. Used to seed the solver so it stays on the same
                 solution branch and avoids elbow-flip discontinuities.

    Solves all 5 arm joints (pan + lift + elbow + wrist_flex + wrist_roll)
    using the actual URDF geometry — no hand-coded link lengths needed.
    """
    # 4×4 target: position only, no orientation constraint (matches Genesis)
    target = np.eye(4)
    target[:3, 3] = [x, y, z]

    # Seed vector: [0, pan, lift, elbow, wflex, wroll, 0] in radians
    seed = np.zeros(7)
    if current_obs is not None:
        seed[1] = math.radians(current_obs.get("shoulder_pan.pos",  0.0))
        seed[2] = math.radians(current_obs.get("shoulder_lift.pos", 0.0))
        seed[3] = math.radians(current_obs.get("elbow_flex.pos",    0.0))
        seed[4] = math.radians(current_obs.get("wrist_flex.pos",    0.0))
        seed[5] = math.radians(current_obs.get("wrist_roll.pos",    0.0))

    angles = _chain.inverse_kinematics(
        target,
        initial_position=seed,
        orientation_mode=None,  # position-only
    )

    # Clamp to URDF joint limits
    for i, lim in enumerate(_LIMITS):
        if lim is not None:
            angles[i] = max(lim[0], min(lim[1], angles[i]))

    return {
        "shoulder_pan.pos":  math.degrees(angles[1]),
        "shoulder_lift.pos": math.degrees(angles[2]),
        "elbow_flex.pos":    math.degrees(angles[3]),
        "wrist_flex.pos":    math.degrees(angles[4]),
        "wrist_roll.pos":    math.degrees(angles[5]),
    }


# ===========================================================================
# Print calibration data at startup (for reference / debugging)
# ===========================================================================
def print_calibration():
    cal_path = pathlib.Path.home() / ".cache" / "huggingface" / "lerobot" / \
               "calibration" / "robots" / "so_follower" / f"{ROBOT_ID}.json"

    if not cal_path.exists():
        print(f"Calibration file not found: {cal_path}")
        return

    print(f"Calibration: {cal_path}")
    try:
        with open(cal_path) as f:
            cal = json.load(f)

        print(f"  {'JOINT':<18s} {'ID':>3s} {'RANGE_MIN':>10s} {'RANGE_MAX':>10s} {'HOMING':>8s} {'DRIVE':>6s} {'TICKS':>7s}")
        print(f"  {'─'*18} {'─'*3} {'─'*10} {'─'*10} {'─'*8} {'─'*6} {'─'*7}")
        for jn in JOINT_NAMES:
            if jn not in cal:
                print(f"  {jn:<18s} (not in calibration)")
                continue
            jc = cal[jn]
            rmin = jc.get("range_min", "?")
            rmax = jc.get("range_max", "?")
            home = jc.get("homing_offset", "?")
            drv  = jc.get("drive_mode", "?")
            mid  = jc.get("id", "?")
            ticks = abs(rmax - rmin) if isinstance(rmin, int) and isinstance(rmax, int) else "?"
            print(f"  {jn:<18s} {str(mid):>3s} {str(rmin):>10s} {str(rmax):>10s} {str(home):>8s} {str(drv):>6s} {str(ticks):>7s}")
        print()
    except Exception as e:
        print(f"Error reading calibration: {e}\n")


# ===========================================================================
# Load monitor
# ===========================================================================
class LoadMonitor:
    def __init__(self, threshold, time_limit, cooldown):
        self.threshold = threshold
        self.time_limit = time_limit
        self.cooldown = cooldown
        self.overload_start = {}
        self.paused = False
        self.pause_start = 0.0
        self._last_warn = {}

    def update(self, load_dict):
        now = time.time()
        for jn, raw in load_dict.items():
            mag = abs(raw) if isinstance(raw, (int, float)) else 0
            if mag >= self.threshold:
                if jn not in self.overload_start:
                    self.overload_start[jn] = now
                    print(f"⚠️  {jn} load={mag}")
                elif now - self.overload_start[jn] >= self.time_limit:
                    print(f"⏸️  {jn} overloaded (load={mag}) for {self.time_limit}s — pausing {self.cooldown}s")
                    self.paused = True
                    self.pause_start = now
                    return
            else:
                self.overload_start.pop(jn, None)

    def check_temp(self, temp_dict):
        now = time.time()
        for jn, temp in temp_dict.items():
            if not isinstance(temp, (int, float)):
                continue
            if temp >= TEMP_PAUSE:
                print(f"⏸️  {jn} temp={temp}°C — pausing {self.cooldown}s")
                self.paused = True
                self.pause_start = now
                return
            elif temp >= TEMP_WARN:
                last = self._last_warn.get(jn, 0)
                if now - last > 10.0:
                    print(f"🌡️  {jn} temp={temp}°C")
                    self._last_warn[jn] = now

    def check_cooldown(self):
        if not self.paused:
            return True
        if time.time() - self.pause_start >= self.cooldown:
            self.paused = False
            self.overload_start.clear()
            print(f"▶️  Cooldown complete — resuming")
            return True
        return False

    def reset(self):
        self.paused = False
        self.overload_start.clear()
        self._last_warn.clear()
        print("✅ Safety monitor reset")


# ===========================================================================
# UDP listener
# ===========================================================================
class UDPServer:
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.sock.setblocking(False)
        self.latest_pos_cmd     = None
        self.latest_gripper_cmd = None
        self.latest_wrist_cmd   = None
        self.request_stop       = False
        self.request_reset      = False
        self.running = True

    def listen(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(BUFFER_SIZE)
                try:
                    cmd = json.loads(data.decode())

                    if cmd.get("mode") == "pos" and "pos" in cmd:
                        pos = cmd["pos"]
                        if len(pos) >= 3:
                            self.latest_pos_cmd = [float(pos[0]), float(pos[1]), float(pos[2])]
                        # Wrist overrides — wrist joints are now solved by IK,
                        # but explicit overrides from the client still take priority.
                        wrist_roll = cmd.get("wrist_roll", cmd.get("wrist_rot", None))
                        wrist_flex = cmd.get("wrist_flex", cmd.get("wrist_tilt", None))
                        if wrist_roll is not None or wrist_flex is not None:
                            self.latest_wrist_cmd = (
                                float(wrist_roll) if wrist_roll is not None else None,
                                float(wrist_flex) if wrist_flex is not None else None,
                            )

                    elif cmd.get("mode") == "gripper" and "state" in cmd:
                        state = cmd["state"].lower()
                        if state in ("open", "close"):
                            self.latest_gripper_cmd = state
                            print(f"Gripper command: {state}")

                    elif cmd.get("mode") == "stop":
                        self.request_stop = True
                        print("🛑 Stop command received")

                    elif cmd.get("mode") == "reset":
                        self.request_reset = True
                        print("Reset command received")

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Invalid command: {e}")
            except BlockingIOError:
                pass
            time.sleep(0.001)

    def stop(self):
        self.running = False
        self.sock.close()


# ===========================================================================
# Main
# ===========================================================================
def main():
    print_calibration()

    print(f"Connecting to SO-101 on {ROBOT_PORT} ...")
    config = ConfigClass(port=ROBOT_PORT, id=ROBOT_ID, use_degrees=True)
    try:
        config.robot_type = "so101_follower"
    except Exception:
        pass

    robot = FollowerClass(config)
    robot.connect(calibrate=False)

    if not robot.is_connected:
        print("ERROR: Could not connect.")
        sys.exit(1)
    print("SO-101 connected!")

    # Disable torque on disabled motors
    enabled  = [jn for jn in JOINT_NAMES if MOTOR_ENABLED.get(jn, True)]
    disabled = [jn for jn in JOINT_NAMES if not MOTOR_ENABLED.get(jn, True)]
    if disabled:
        print(f"Motors DISABLED: {', '.join(disabled)}")
        for jn in disabled:
            try:
                robot.bus.write("Torque_Enable", jn, 0)
            except Exception:
                pass
    print(f"Motors ENABLED:  {', '.join(enabled)}")

    # Read initial positions
    obs = robot.get_observation()
    target = {f"{jn}.pos": obs.get(f"{jn}.pos", 0.0) for jn in JOINT_NAMES}
    print(f"Initial positions: {target}")
    print(f"Gripper: open={GRIPPER_OPEN} close={GRIPPER_CLOSE} (0-100 scale)")

    monitor = LoadMonitor(LOAD_THRESHOLD, LOAD_TIME_LIMIT, COOLDOWN_TIME)

    udp = UDPServer(UDP_IP, UDP_PORT)
    threading.Thread(target=udp.listen, daemon=True).start()
    print(f"\nUDP listening on {UDP_IP}:{UDP_PORT}")
    print(f"Safety: load>{LOAD_THRESHOLD} for {LOAD_TIME_LIMIT}s → {COOLDOWN_TIME}s cooldown")
    print(f"        temp warn>{TEMP_WARN}°C, pause>{TEMP_PAUSE}°C → {COOLDOWN_TIME}s cooldown")
    print(f"Workspace scale: {WORKSPACE_SCALE}")
    speed_str = f"{ARM_SPEED}°/s ({ARM_SPEED/CONTROL_HZ:.1f}°/cycle)" if ARM_SPEED > 0 else "unlimited"
    print(f"Arm speed: {speed_str}")
    print("IK: ikpy numerical solver, seeded from current pose (no branch-flipping)")
    print("Ctrl+C to quit.\n")

    loop_dt = 1.0 / CONTROL_HZ
    load_interval = 1.0 / LOAD_CHECK_HZ
    last_load_check = 0.0
    emergency_stopped = False
    next_tick = time.perf_counter()

    try:
        while True:

            # ---- Emergency stop ----
            if udp.request_stop:
                if not emergency_stopped:
                    print("🛑 Emergency stop (send 'reset' to recover)")
                    try:
                        robot.bus.disable_torque()
                    except Exception:
                        pass
                    emergency_stopped = True
                udp.request_stop = False

            # ---- Manual reset ----
            if udp.request_reset:
                if emergency_stopped:
                    print("✅ Re-enabling torque")
                    try:
                        robot.bus.enable_torque()
                    except Exception:
                        pass
                    emergency_stopped = False
                monitor.reset()
                obs = robot.get_observation()
                target = {f"{jn}.pos": obs.get(f"{jn}.pos", 0.0) for jn in JOINT_NAMES}
                udp.request_reset = False

            if emergency_stopped:
                next_tick += loop_dt
                sleep_t = next_tick - time.perf_counter()
                if sleep_t > 0:
                    time.sleep(sleep_t)
                continue

            # ---- Load & temp monitoring ----
            now_time = time.time()
            if now_time - last_load_check >= load_interval:
                last_load_check = now_time
                try:
                    loads = robot.bus.sync_read("Present_Load")
                    monitor.update(loads)
                except Exception:
                    pass
                try:
                    temps = robot.bus.sync_read("Present_Temperature")
                    monitor.check_temp(temps)
                except Exception:
                    pass

            if monitor.paused:
                if not monitor.check_cooldown():
                    next_tick += loop_dt
                    sleep_t = next_tick - time.perf_counter()
                    if sleep_t > 0:
                        time.sleep(sleep_t)
                    continue
                else:
                    obs = robot.get_observation()
                    target = {f"{jn}.pos": obs.get(f"{jn}.pos", 0.0) for jn in JOINT_NAMES}

            # ============================================================
            # Command processing
            # ============================================================

            # 1. Position → scale → IK (all 5 arm joints, seeded from current pose)
            if udp.latest_pos_cmd is not None:
                pos = udp.latest_pos_cmd
                obs = robot.get_observation()
                ik = solve_ik(
                    pos[0] * WORKSPACE_SCALE,
                    pos[1] * WORKSPACE_SCALE,
                    pos[2] * WORKSPACE_SCALE,
                    current_obs=obs,    # seed prevents elbow-flip branch switching
                )
                target.update(ik)
                udp.latest_pos_cmd = None

            # 2. Gripper (0-100 scale: 0=closed, 100=open)
            if udp.latest_gripper_cmd is not None:
                if udp.latest_gripper_cmd == "open":
                    target["gripper.pos"] = GRIPPER_OPEN
                else:
                    target["gripper.pos"] = GRIPPER_CLOSE
                udp.latest_gripper_cmd = None

            # 3. Explicit wrist overrides (radians from sender → degrees for lerobot).
            #    These override whatever IK computed for wrist joints, same as before.
            if udp.latest_wrist_cmd is not None:
                roll_rad, flex_rad = udp.latest_wrist_cmd
                if roll_rad is not None:
                    target["wrist_roll.pos"] = math.degrees(roll_rad)
                if flex_rad is not None:
                    target["wrist_flex.pos"] = math.degrees(flex_rad)
                udp.latest_wrist_cmd = None

            # 4. Send — only enabled motors, with speed limiting
            action = {k: v for k, v in target.items()
                      if MOTOR_ENABLED.get(k.replace(".pos", ""), True)}

            if ARM_SPEED > 0:
                max_step = ARM_SPEED / CONTROL_HZ  # degrees per cycle
                obs = robot.get_observation()
                for key in action:
                    current = obs.get(key, action[key])
                    goal    = action[key]
                    delta   = goal - current
                    if abs(delta) > max_step:
                        action[key] = current + max_step * (1.0 if delta > 0 else -1.0)

            robot.send_action(action)

            # ---- Rate control ----
            next_tick += loop_dt
            sleep_t = next_tick - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        udp.stop()
        try:
            robot.bus.disable_torque()
            robot.disconnect()
        except Exception:
            pass
        print("Done.")


if __name__ == "__main__":
    main()