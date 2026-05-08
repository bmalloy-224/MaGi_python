import genesis as gs
import numpy as np
import socket
import json
import threading
import time
import os
import random

# UDP settings
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
BUFFER_SIZE = 1024

# Gripper joint limits
GRIPPER_OPEN = 0.5
GRIPPER_CLOSE = -0.1

# Respawn interval (seconds)
RESPAWN_INTERVAL = 500.0

# Object sizes
CUBE_SIZE = 0.05
LARGE_SIZE = 0.15
CUBE_HEIGHT = 0.05

# Fixed positions for stacks (closer to robot)
STACK_POSITIONS = [
    (0.35,  0.25, CUBE_HEIGHT),
    (0.35, -0.25, CUBE_HEIGHT),
    (-0.25, 0.35, CUBE_HEIGHT),
    (-0.25, -0.35, CUBE_HEIGHT),
]

# Other objects moved closer
LARGE_BOX_POS = (0.20, 0.20, CUBE_HEIGHT)
LARGE_SPHERE_POS = (0.20, -0.20, CUBE_HEIGHT)
SMALL_SPHERE_POS = (-0.20, 0.20, CUBE_HEIGHT)
LARGE_CYLINDER_POS = (-0.20, -0.20, CUBE_HEIGHT)

# Colors for non‑cube objects
LARGE_BOX_COLOR = (0.9, 0.3, 0.1, 1.0)
LARGE_SPHERE_COLOR = (0.2, 0.6, 0.9, 1.0)
SMALL_SPHERE_COLOR = (0.95, 0.85, 0.1, 1.0)
LARGE_CYLINDER_COLOR = (0.8, 0.1, 0.8, 1.0)

def random_color_no_green():
    """Generate a random RGBA color that avoids green tones."""
    while True:
        r = random.random()
        g = random.random()
        b = random.random()
        if not (g > 0.5 and g > r and g > b):
            return (r, g, b, 1.0)

class UDPServer:
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.sock.setblocking(False)
        self.latest_pos_cmd = None
        self.latest_gripper_cmd = None
        self.latest_wrist_cmd = None
        self.running = True

    def listen(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(BUFFER_SIZE)
                try:
                    cmd = json.loads(data.decode())
                    if cmd.get("mode") == "pos" and "pos" in cmd:
                        pos = cmd["pos"]
                        if len(pos) == 3:
                            self.latest_pos_cmd = np.array(pos, dtype=np.float32)
                            print(f"Position command: {self.latest_pos_cmd}")
                        wrist_roll = cmd.get("wrist_roll", None)
                        wrist_flex = cmd.get("wrist_flex", None)
                        if wrist_roll is None and "wrist_rot" in cmd:
                            wrist_roll = cmd["wrist_rot"]
                        if wrist_flex is None and "wrist_tilt" in cmd:
                            wrist_flex = cmd["wrist_tilt"]
                        if wrist_roll is not None or wrist_flex is not None:
                            self.latest_wrist_cmd = (
                                float(wrist_roll) if wrist_roll is not None else 0.0,
                                float(wrist_flex) if wrist_flex is not None else 0.0,
                            )
                            print(f"Wrist command: roll={self.latest_wrist_cmd[0]:.3f} flex={self.latest_wrist_cmd[1]:.3f}")
                    elif cmd.get("mode") == "gripper" and "state" in cmd:
                        state = cmd["state"].lower()
                        if state in ["open", "close"]:
                            self.latest_gripper_cmd = state
                            print(f"Gripper command: {state}")
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Invalid command: {e}")
            except BlockingIOError:
                pass
            time.sleep(0.001)

    def stop(self):
        self.running = False
        self.sock.close()

def respawn_objects(all_objects_with_pos):
    """Reset all objects to their initial positions."""
    for obj, init_pos in all_objects_with_pos:
        obj.set_pos(init_pos)
    print(f"🔄 Objects respawned ({len(all_objects_with_pos)} objects)")

def main():
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.006,          # slightly larger for faster simulation
            substeps=1,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -1.0, 0.8),
            camera_lookat=(0.0, 0.0, 0.3),
        ),
        show_viewer=True,
    )

    # # Ground plane
    # ground = scene.add_entity(
    #     gs.morphs.Box(
    #         pos=(0.0, 0.0, -0.05),
    #         size=(5.0, 5.0, 0.1),
    #         fixed=True,
    #     ),
    #     material=gs.materials.Rigid(),
    #     surface=gs.surfaces.Plastic(),
    # )

    ground = scene.add_entity(
    gs.morphs.Plane(
        pos=(0.0, 0.0, 0.00),   # same Z position as your box
        fixed=True,
    ),
    surface=gs.surfaces.Plastic(
        diffuse_texture=gs.textures.ImageTexture(
            image_path="CustomUVChecker_byValle_1K.png",
            encoding='srgb'
        )
    ),
    vis_mode='visual',
)




    all_objects_with_pos = []

    # --- 4 stacks of 3 cubes each, random colors ---
    for sx, sy, sz_base in STACK_POSITIONS:
        for level in range(3):
            pos = (sx, sy, sz_base + level * CUBE_SIZE)
            color = random_color_no_green()
            cube = scene.add_entity(
                gs.morphs.Box(
                    pos=pos,
                    size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                    fixed=False,
                ),
                material=gs.materials.Rigid(),
                surface=gs.surfaces.Rough(color=color),
            )
            all_objects_with_pos.append((cube, pos))

    # --- Large box ---
    large_box = scene.add_entity(
        gs.morphs.Box(
            pos=LARGE_BOX_POS,
            size=(LARGE_SIZE, LARGE_SIZE, LARGE_SIZE),
            fixed=False,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Plastic(color=LARGE_BOX_COLOR),
    )
    all_objects_with_pos.append((large_box, LARGE_BOX_POS))

    # --- Large sphere ---
    large_sphere = scene.add_entity(
        gs.morphs.Sphere(
            pos=LARGE_SPHERE_POS,
            radius=LARGE_SIZE / 2.0,
            fixed=False,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Plastic(color=LARGE_SPHERE_COLOR),
    )
    all_objects_with_pos.append((large_sphere, LARGE_SPHERE_POS))

    # --- Small sphere ---
    small_sphere = scene.add_entity(
        gs.morphs.Sphere(
            pos=SMALL_SPHERE_POS,
            radius=CUBE_SIZE / 2.0,
            fixed=False,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Plastic(color=SMALL_SPHERE_COLOR),
    )
    all_objects_with_pos.append((small_sphere, SMALL_SPHERE_POS))

    # --- Large cylinder ---
    large_cylinder = scene.add_entity(
        gs.morphs.Cylinder(
            pos=LARGE_CYLINDER_POS,
            radius=LARGE_SIZE / 2.0,
            height=LARGE_SIZE,
            fixed=False,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Plastic(color=LARGE_CYLINDER_COLOR),
    )
    all_objects_with_pos.append((large_cylinder, LARGE_CYLINDER_POS))

    print(f"🎲 Created {len(all_objects_with_pos)} objects: 4 stacks of 3 cubes, large box, large sphere, small sphere, large cylinder")
    print(f"   Small cube size: {CUBE_SIZE}, large size: {LARGE_SIZE}")
    print(f"   Objects will respawn to original positions every {RESPAWN_INTERVAL}s")

    # Load robot — GREEN arm
    urdf_path = "so101/so101_new_calib.urdf"
    if not os.path.exists(urdf_path):
        urdf_path = "SO101/so101_new_calib.urdf"
        if not os.path.exists(urdf_path):
            print(f"Error: URDF not found.")
            return

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            pos=(0.0, 0.0, 0.0),
            fixed=True,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Plastic(color=(0.1, 0.75, 0.2, 1.0)),
    )
    scene.build()

    # Joint info
    joints = robot.joints
    joint_names = [j.name for j in joints]
    print("Joint names (actuated):", joint_names)
    print("Number of DOFs:", robot.n_qs)

    gripper_idx = joint_names.index("gripper") if "gripper" in joint_names else -1
    wrist_roll_idx = joint_names.index("wrist_roll") if "wrist_roll" in joint_names else -1
    wrist_flex_idx = joint_names.index("wrist_flex") if "wrist_flex" in joint_names else -1

    # Control gains (slightly increased for better response)
    robot.set_dofs_kp(90.0)
    robot.set_dofs_kv(18.0)

    # Initial joint positions
    current_qpos = robot.get_dofs_position()
    target_qpos = current_qpos.clone()
    print("Initial joint positions:", current_qpos)

    # Warm-up: let objects settle
    print("Warming up simulation for 500 steps...")
    for _ in range(500):
        scene.step()
    print("Warm-up complete.")

    # ------------------------------------------------------------------
    # TEST MOVEMENT: Move shoulder_pan to 0.5 rad for 2 seconds
    print("Test movement: setting shoulder_pan to 0.5 rad")
    test_qpos = target_qpos.clone()
    test_qpos[0] = 0.5
    robot.control_dofs_position(test_qpos)
    for _ in range(120):
        scene.step()
    print("Test movement complete.")
    robot.control_dofs_position(current_qpos)
    for _ in range(60):
        scene.step()
    target_qpos = robot.get_dofs_position().clone()
    # ------------------------------------------------------------------

    # End-effector link and fixed orientation
    eef_link = robot.get_link("gripper_link")
    target_quat = np.array([1, 0, 0, 0])

    # Start UDP server
    udp_server = UDPServer(UDP_IP, UDP_PORT)
    server_thread = threading.Thread(target=udp_server.listen, daemon=True)
    server_thread.start()
    print(f"UDP server listening on {UDP_IP}:{UDP_PORT}")

    print("Simulation running. Send JSON commands:")
    print('  {"mode": "pos", "pos": [0.3, 0.2, 0.2]}')
    print('  {"mode": "pos", "pos": [0.3, 0.2, 0.2], "wrist_roll": 0.5, "wrist_flex": 0.3}')
    print('  (or use "wrist_rot" and "wrist_tilt" instead)')
    print('  {"mode": "gripper", "state": "open"}')
    print('  {"mode": "gripper", "state": "close"}')

    # Timer for respawning
    last_respawn_time = time.time()

    try:
        while True:
            # Respawn all objects to original positions periodically
            if time.time() - last_respawn_time >= RESPAWN_INTERVAL:
                respawn_objects(all_objects_with_pos)
                last_respawn_time = time.time()

            # Position command (IK)
            if udp_server.latest_pos_cmd is not None:
                target_pos = udp_server.latest_pos_cmd
                q_ik = robot.inverse_kinematics(
                    link=eef_link,
                    pos=target_pos,
                    quat=target_quat,
                )
                if q_ik is not None:
                    target_qpos[:] = q_ik[:]
                    print(f"IK success: {q_ik}")
                else:
                    print("IK failed for target:", target_pos)
                udp_server.latest_pos_cmd = None

            # Gripper command
            if udp_server.latest_gripper_cmd is not None:
                if gripper_idx != -1:
                    if udp_server.latest_gripper_cmd == "open":
                        target_qpos[gripper_idx] = GRIPPER_OPEN
                    else:
                        target_qpos[gripper_idx] = GRIPPER_CLOSE
                    print(f"Gripper set to {udp_server.latest_gripper_cmd}")
                else:
                    print("Gripper joint not found")
                udp_server.latest_gripper_cmd = None

            # Wrist command
            if udp_server.latest_wrist_cmd is not None:
                wrist_roll_val, wrist_flex_val = udp_server.latest_wrist_cmd
                if wrist_roll_idx != -1:
                    target_qpos[wrist_roll_idx] = wrist_roll_val
                if wrist_flex_idx != -1:
                    target_qpos[wrist_flex_idx] = wrist_flex_val
                print(f"Wrist set: roll={wrist_roll_val:.3f} flex={wrist_flex_val:.3f}")
                udp_server.latest_wrist_cmd = None

            # Apply target positions
            robot.control_dofs_position(target_qpos)
            scene.step()

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        udp_server.stop()
        print("Exited.")

if __name__ == "__main__":
    main()
