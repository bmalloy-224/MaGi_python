"""
SO-101 Real Robot Control via UDP
=================================
Commands (JSON over UDP):
  {"mode": "pos", "pos": [x, y, z]}
  {"mode": "pos", "pos": [x, y, z], "wrist_roll": 0.5, "wrist_flex": -0.3}
  {"mode": "gripper", "state": "open"}
  {"mode": "gripper", "state": "close"}
  {"mode": "stop"}    <- emergency stop (disables torque)
  {"mode": "reset"}   <- re-enable after stop/cooldown
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
    ConfigClass   = SOFollowerRobotConfig
except ImportError:
    try:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        FollowerClass = SO101Follower
        ConfigClass   = SO101FollowerConfig
    except ImportError:
        try:
            from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig
            FollowerClass = SO101Follower
            ConfigClass   = SO101FollowerConfig
        except ImportError:
            print("ERROR: Could not import lerobot SO follower classes.")
            sys.exit(1)

# ===========================================================================
# Configuration
# ===========================================================================
ROBOT_PORT = "/dev/ttyACM4"
ROBOT_ID   = "so101_arm"

UDP_IP      = "0.0.0.0"
UDP_PORT    = 5005
BUFFER_SIZE = 1024
CONTROL_HZ  = 50

# Workspace scale: sim +-1 m -> real arm ~+-0.5 m reach
WORKSPACE_SCALE = 1.0

# Arm speed limit in degrees/second (0 = unlimited)
ARM_SPEED = 100

# Watchdog: if no command arrives in this many seconds, hold position
CMD_TIMEOUT = 2.0

# Gripper: mapped from Genesis radian values via URDF limits (-0.174533..1.74533)
# open=0.5rad → 35.14%,  close=-0.1rad → 3.88%
GRIPPER_OPEN  = 35.14
GRIPPER_CLOSE = 3.88

# Load/temp safety
LOAD_THRESHOLD  = 850
LOAD_TIME_LIMIT = 3.0
LOAD_CHECK_HZ   = 10
TEMP_WARN       = 75
TEMP_PAUSE      = 90
COOLDOWN_TIME   = 2.0

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

# Log IK output angles each cycle (useful for checking workspace limits)
IK_LOG = False

# Per-motor enable -- set False to skip sending to that motor
MOTOR_ENABLED = {
    "shoulder_pan":  True,
    "shoulder_lift": True,
    "elbow_flex":    True,
    "wrist_flex":    True,
    "wrist_roll":    True,
    "gripper":       True,
}

# ===========================================================================
# IK -- ikpy numerical solver using actual URDF geometry
# ===========================================================================
# Chain (7 links):
#   base_link(fixed) -> shoulder_pan -> shoulder_lift -> elbow_flex ->
#   wrist_flex -> wrist_roll -> gripper_frame(fixed)
# active_links_mask: [False, True, True, True, True, True, False]
#
URDF_PATH = "so101_new_calib.urdf"

# Joint limits from URDF (radians) -- indices match chain links 0..6
_LIMITS = [
    None,                    # 0: base_link (inactive)
    (-1.91986,  1.91986),   # 1: shoulder_pan
    (-1.74533,  1.74533),   # 2: shoulder_lift
    (-1.69,     1.69),      # 3: elbow_flex
    (-1.65806,  1.65806),   # 4: wrist_flex
    (-2.74385,  2.84121),   # 5: wrist_roll
    None,                    # 6: gripper_frame (fixed)
]

def _load_chain(urdf_path):
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
            if len(chain.links) != len(_LIMITS):
                print(f"WARNING: URDF chain has {len(chain.links)} links, "
                      f"expected {len(_LIMITS)}. active_links_mask may be wrong.")
            print(f"IK chain loaded from: {p}  ({len(chain.links)} links)")
            return chain
    print("ERROR: URDF not found for IK chain. Tried:")
    for p in candidates:
        print(f"  {p}")
    sys.exit(1)

_chain = _load_chain(URDF_PATH)


def solve_ik(x, y, z, current_obs=None):
    """
    Cartesian position (metres, already workspace-scaled) ->
    joint angle dict (degrees) for lerobot send_action, or None on failure.

    Seeds from current_obs so the solver stays on the same solution branch
    and avoids elbow-flip discontinuities.
    """
    seed = np.zeros(7)
    if current_obs is not None:
        seed[1] = math.radians(current_obs.get("shoulder_pan.pos",  0.0))
        seed[2] = math.radians(current_obs.get("shoulder_lift.pos", 0.0))
        seed[3] = math.radians(current_obs.get("elbow_flex.pos",    0.0))
        seed[4] = math.radians(current_obs.get("wrist_flex.pos",    0.0))
        seed[5] = math.radians(current_obs.get("wrist_roll.pos",    0.0))

    # Clamp seed -- ikpy rejects out-of-bounds initial guesses
    for i, lim in enumerate(_LIMITS):
        if lim is not None:
            seed[i] = max(lim[0], min(lim[1], seed[i]))

    # Match Genesis: identity orientation at target position.
    # ikpy accepts a 4x4 frame when orientation_mode is left at default.
    target_frame = np.eye(4)
    target_frame[:3, 3] = [x, y, z]

    angles = None
    try:
        angles = _chain.inverse_kinematics(target_frame, initial_position=seed)
    except Exception as e:
        print(f"IK frame error: {e}")

    if angles is None:
        print("IK frame failed -- retrying position-only")
        try:
            angles = _chain.inverse_kinematics([x, y, z], initial_position=seed)
        except Exception as e:
            print(f"IK position-only error: {e}")
            return None

    if angles is None:
        print("IK failed to converge -- holding previous target")
        return None

    # Clamp output to URDF limits
    for i, lim in enumerate(_LIMITS):
        if lim is not None:
            angles[i] = max(lim[0], min(lim[1], angles[i]))

    result = {
        "shoulder_pan.pos":  math.degrees(angles[1]),
        "shoulder_lift.pos": math.degrees(angles[2]),
        "elbow_flex.pos":    math.degrees(angles[3]),
        "wrist_flex.pos":    math.degrees(angles[4]),
        "wrist_roll.pos":    math.degrees(angles[5]),
    }

    if IK_LOG:
        print("IK -> " + "  ".join(f"{k.split('.')[0]}={v:+.1f}" for k, v in result.items()))

    return result


def _shortest_angle_delta(current_deg, target_deg):
    """Shortest signed delta for a continuous-rotation joint (wrist_roll)."""
    delta = (target_deg - current_deg + 180.0) % 360.0 - 180.0
    return delta


# ===========================================================================
# Calibration display
# ===========================================================================
def print_calibration():
    cal_path = (pathlib.Path.home() / ".cache" / "huggingface" / "lerobot" /
                "calibration" / "robots" / "so_follower" / f"{ROBOT_ID}.json")
    if not cal_path.exists():
        print(f"Calibration file not found: {cal_path}")
        return None
    print(f"Calibration: {cal_path}")
    try:
        with open(cal_path) as f:
            cal = json.load(f)
        print(f"  {'JOINT':<18s} {'ID':>3s} {'RANGE_MIN':>10s} {'RANGE_MAX':>10s} "
              f"{'HOMING':>8s} {'DRIVE':>6s} {'TICKS':>7s}")
        print(f"  {'--'*18} {'--'*3} {'--'*10} {'--'*10} {'--'*8} {'--'*6} {'--'*7}")
        for jn in JOINT_NAMES:
            if jn not in cal:
                print(f"  {jn:<18s} (not in calibration)")
                continue
            jc    = cal[jn]
            rmin  = jc.get("range_min", "?")
            rmax  = jc.get("range_max", "?")
            home  = jc.get("homing_offset", "?")
            drv   = jc.get("drive_mode", "?")
            mid   = jc.get("id", "?")
            ticks = abs(rmax - rmin) if isinstance(rmin, int) and isinstance(rmax, int) else "?"
            print(f"  {jn:<18s} {str(mid):>3s} {str(rmin):>10s} {str(rmax):>10s} "
                  f"{str(home):>8s} {str(drv):>6s} {str(ticks):>7s}")
        print()
        return cal
    except Exception as e:
        print(f"Error reading calibration: {e}\n")
        return None
    return cal


def _gripper_limits_from_cal(cal):
    """
    Determine gripper open/close values to send via send_action.

    lerobot with use_degrees=True maps raw ticks through calibration into
    a 0-100 percentage space.  Current values match Genesis:
        Genesis open  =  0.5 rad → 35.14% (via URDF limits -0.174533..1.74533)
        Genesis close = -0.1 rad →  3.88%

    If gripper direction is reversed on your unit, swap GRIPPER_OPEN/CLOSE.
    """
    if cal and "gripper" in cal:
        g = cal["gripper"]
        rmin = g.get("range_min", "?")
        rmax = g.get("range_max", "?")
        print(f"Gripper cal: range_min={rmin}  range_max={rmax}  ticks span="
              f"{abs(rmax-rmin) if isinstance(rmin,int) else '?'}")
        print(f"Gripper sending: open={GRIPPER_OPEN}  close={GRIPPER_CLOSE}  "
              f"(0-100 pct scale — change if gripper barely moves or slams)")
    return GRIPPER_OPEN, GRIPPER_CLOSE


# ===========================================================================
# Load / temp safety monitor
# ===========================================================================
class LoadMonitor:
    def __init__(self, threshold, time_limit, cooldown):
        self.threshold      = threshold
        self.time_limit     = time_limit
        self.cooldown       = cooldown
        self.overload_start = {}
        self.paused         = False
        self.pause_start    = 0.0
        self._last_warn     = {}

    def update(self, load_dict):
        now = time.perf_counter()
        for jn, raw in load_dict.items():
            mag = abs(raw) if isinstance(raw, (int, float)) else 0
            if mag >= self.threshold:
                if jn not in self.overload_start:
                    self.overload_start[jn] = now
                    print(f"Warning: {jn} load={mag}")
                elif now - self.overload_start[jn] >= self.time_limit:
                    print(f"Pause: {jn} overload={mag} for {self.time_limit}s -> cooldown {self.cooldown}s")
                    self.paused      = True
                    self.pause_start = now
                    return
            else:
                self.overload_start.pop(jn, None)

    def check_temp(self, temp_dict):
        now = time.perf_counter()
        for jn, temp in temp_dict.items():
            if not isinstance(temp, (int, float)):
                continue
            if temp >= TEMP_PAUSE:
                print(f"Pause: {jn} temp={temp}C -> cooldown {self.cooldown}s")
                self.paused      = True
                self.pause_start = now
                return
            elif temp >= TEMP_WARN:
                last = self._last_warn.get(jn, 0)
                if now - last > 10.0:
                    print(f"Warn: {jn} temp={temp}C")
                    self._last_warn[jn] = now

    def check_cooldown(self):
        if not self.paused:
            return True
        if time.perf_counter() - self.pause_start >= self.cooldown:
            self.paused = False
            self.overload_start.clear()
            print("Cooldown complete -- resuming")
            return True
        return False

    def reset(self):
        self.paused = False
        self.overload_start.clear()
        self._last_warn.clear()
        print("Safety monitor reset")


# ===========================================================================
# UDP listener  (thread-safe via lock + atomic pop methods)
# ===========================================================================
class UDPServer:
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((ip, port))
        self.sock.setblocking(False)

        self._lock        = threading.Lock()
        self._pos_cmd     = None   # [x, y, z]
        self._gripper_cmd = None   # "open" | "close"
        self._wrist_cmd   = None   # (roll_rad|None, flex_rad|None)
        self._stop        = False
        self._reset       = False
        self.running      = True

    # --- atomic getters (check-and-clear in one lock) ---

    def pop_pos(self):
        with self._lock:
            v, self._pos_cmd = self._pos_cmd, None
            return v

    def pop_gripper(self):
        with self._lock:
            v, self._gripper_cmd = self._gripper_cmd, None
            return v

    def pop_wrist(self):
        with self._lock:
            v, self._wrist_cmd = self._wrist_cmd, None
            return v

    def pop_stop(self):
        with self._lock:
            v, self._stop = self._stop, False
            return v

    def pop_reset(self):
        with self._lock:
            v, self._reset = self._reset, False
            return v

    # --- listener thread ---

    def listen(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(BUFFER_SIZE)
                try:
                    cmd = json.loads(data.decode("utf-8"))
                    self._handle(cmd)
                except (json.JSONDecodeError, KeyError, ValueError,
                        UnicodeDecodeError) as e:
                    print(f"Invalid command: {e}")
            except BlockingIOError:
                pass
            time.sleep(0.001)

    def _handle(self, cmd):
        mode = cmd.get("mode")

        if mode == "pos" and "pos" in cmd:
            pos = cmd["pos"]
            if len(pos) == 3:
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                if not all(math.isfinite(v) for v in (x, y, z)):
                    print(f"Rejected pos command: non-finite values {pos}")
                    return
                with self._lock:
                    self._pos_cmd = [x, y, z]
            roll = cmd.get("wrist_roll", cmd.get("wrist_rot"))
            flex = cmd.get("wrist_flex", cmd.get("wrist_tilt"))
            if roll is not None or flex is not None:
                with self._lock:
                    self._wrist_cmd = (
                        float(roll) if roll is not None else None,
                        float(flex) if flex is not None else None,
                    )

        elif mode == "gripper" and "state" in cmd:
            state = cmd["state"].lower()
            if state in ("open", "close"):
                with self._lock:
                    self._gripper_cmd = state
                print(f"Gripper: {state}")

        elif mode == "stop":
            with self._lock:
                self._stop = True
            print("STOP command received")

        elif mode == "reset":
            with self._lock:
                self._reset = True
            print("RESET command received")

    def stop(self):
        self.running = False
        self.sock.close()


# ===========================================================================
# Main loop
# ===========================================================================
def main():
    cal = print_calibration()
    g_open, g_close = _gripper_limits_from_cal(cal)

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

    obs    = robot.get_observation()
    target = {f"{jn}.pos": obs.get(f"{jn}.pos", 0.0) for jn in JOINT_NAMES}
    print(f"Initial positions: { {k: round(v, 1) for k, v in target.items()} }")
    print(f"Gripper: open={GRIPPER_OPEN}  close={GRIPPER_CLOSE}  (Genesis-mapped %)")

    monitor = LoadMonitor(LOAD_THRESHOLD, LOAD_TIME_LIMIT, COOLDOWN_TIME)

    udp = UDPServer(UDP_IP, UDP_PORT)
    threading.Thread(target=udp.listen, daemon=True).start()

    print(f"\nUDP on {UDP_IP}:{UDP_PORT}")
    print(f"Safety: load>{LOAD_THRESHOLD} for {LOAD_TIME_LIMIT}s -> {COOLDOWN_TIME}s cooldown  "
          f"| temp warn>{TEMP_WARN}C  pause>{TEMP_PAUSE}C")
    speed_str = (f"{ARM_SPEED}deg/s ({ARM_SPEED/CONTROL_HZ:.1f}deg/cycle)"
                 if ARM_SPEED > 0 else "unlimited")
    print(f"Speed: {speed_str}  |  Workspace scale: {WORKSPACE_SCALE}  |  Watchdog: {CMD_TIMEOUT}s")
    print("IK: ikpy numerical (URDF geometry, seeded from current pose)")
    print("Ctrl+C to quit.\n")

    loop_dt           = 1.0 / CONTROL_HZ
    load_interval     = 1.0 / LOAD_CHECK_HZ
    last_load_check   = 0.0
    last_cmd_time     = time.perf_counter()
    emergency_stopped = False
    next_tick         = time.perf_counter()

    try:
        while True:

            # ----------------------------------------------------------------
            # Emergency stop
            # ----------------------------------------------------------------
            if udp.pop_stop():
                if not emergency_stopped:
                    print("Emergency stop -- disabling torque  (send 'reset' to recover)")
                    try:
                        robot.bus.disable_torque()
                        emergency_stopped = True
                    except Exception as e:
                        print(f"Failed to disable torque: {e}")

            # ----------------------------------------------------------------
            # Manual reset
            # ----------------------------------------------------------------
            if udp.pop_reset():
                if emergency_stopped:
                    print("Re-enabling torque")
                    try:
                        robot.bus.enable_torque()
                        emergency_stopped = False
                    except Exception as e:
                        print(f"Failed to enable torque: {e}")
                monitor.reset()
                obs    = robot.get_observation()
                target = {f"{jn}.pos": obs.get(f"{jn}.pos", 0.0) for jn in JOINT_NAMES}
                last_cmd_time = time.perf_counter()

            if emergency_stopped:
                next_tick += loop_dt
                sleep_t = next_tick - time.perf_counter()
                if sleep_t > 0:
                    time.sleep(sleep_t)
                continue

            # ----------------------------------------------------------------
            # Load & temp monitoring (throttled)
            # ----------------------------------------------------------------
            now_pc = time.perf_counter()
            if now_pc - last_load_check >= load_interval:
                last_load_check = now_pc
                try:
                    monitor.update(robot.bus.sync_read("Present_Load"))
                except Exception:
                    pass
                try:
                    monitor.check_temp(robot.bus.sync_read("Present_Temperature"))
                except Exception:
                    pass

            if monitor.paused:
                if not monitor.check_cooldown():
                    next_tick += loop_dt
                    sleep_t = next_tick - time.perf_counter()
                    if sleep_t > 0:
                        time.sleep(sleep_t)
                    continue
                # Resumed -- resync target to actual position
                obs    = robot.get_observation()
                target = {f"{jn}.pos": obs.get(f"{jn}.pos", 0.0) for jn in JOINT_NAMES}

            # ----------------------------------------------------------------
            # Single observation read per cycle (IK seed + speed limit)
            # ----------------------------------------------------------------
            try:
                obs = robot.get_observation()
            except Exception as e:
                print(f"get_observation failed: {e} -- skipping cycle")
                next_tick += loop_dt
                sleep_t = next_tick - time.perf_counter()
                if sleep_t > 0:
                    time.sleep(sleep_t)
                continue

            # ----------------------------------------------------------------
            # Command processing
            # ----------------------------------------------------------------
            got_cmd = False

            # 1. Position -> IK (all 5 arm joints, seeded from current obs)
            pos_cmd = udp.pop_pos()
            if pos_cmd is not None:
                ik = solve_ik(
                    pos_cmd[0] * WORKSPACE_SCALE,
                    pos_cmd[1] * WORKSPACE_SCALE,
                    pos_cmd[2] * WORKSPACE_SCALE,
                    current_obs=obs,
                )
                if ik is not None:
                    target.update(ik)
                got_cmd = True

            # 2. Gripper (0-100 scale)
            grip_cmd = udp.pop_gripper()
            if grip_cmd is not None:
                target["gripper.pos"] = g_open if grip_cmd == "open" else g_close
                got_cmd = True

            # 3. Explicit wrist overrides (radians -> degrees, override IK result)
            wrist_cmd = udp.pop_wrist()
            if wrist_cmd is not None:
                roll_rad, flex_rad = wrist_cmd
                # Match Genesis: missing wrist joint is forced to 0.0, not preserved
                target["wrist_roll.pos"] = math.degrees(roll_rad) if roll_rad is not None else 0.0
                target["wrist_flex.pos"] = math.degrees(flex_rad) if flex_rad is not None else 0.0
                got_cmd = True

            if got_cmd:
                last_cmd_time = time.perf_counter()
            elif CMD_TIMEOUT > 0 and (time.perf_counter() - last_cmd_time) > CMD_TIMEOUT:
                pass  # watchdog: arm holds last target naturally

            # ----------------------------------------------------------------
            # Speed limiting + send
            # ----------------------------------------------------------------
            action = {k: v for k, v in target.items()
                      if MOTOR_ENABLED.get(k.replace(".pos", ""), True)}

            if ARM_SPEED > 0:
                max_step = ARM_SPEED / CONTROL_HZ
                for key in list(action.keys()):
                    current = obs.get(key)
                    if current is None:
                        print(f"Warning: observation missing {key}")
                        continue
                    if key == "wrist_roll.pos":
                        # Continuous rotation joint — take shortest angular path
                        delta = _shortest_angle_delta(current, action[key])
                        if abs(delta) > max_step:
                            action[key] = current + math.copysign(max_step, delta)
                        else:
                            action[key] = current + delta
                    else:
                        delta = action[key] - current
                        if abs(delta) > max_step:
                            action[key] = current + math.copysign(max_step, delta)

            try:
                robot.send_action(action)
            except Exception as e:
                print(f"send_action failed: {e}")

            # ----------------------------------------------------------------
            # Rate control (phase-locked)
            # ----------------------------------------------------------------
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
