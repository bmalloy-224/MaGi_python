# 🌀 MaGi_python — Malloy artificial Geometric intelligence

**Hardware-Embodied Geometric Intelligence in Python**

[![License](https://img.shields.io/badge/License-GPL--3.0%20%2B%20Commercial-blue.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Geometric%20Intelligence-green.svg)](https://github.com/bmalloy-224/MaGi_python)
[![Platform](https://img.shields.io/badge/Platform-Python%203.x%20%7C%20PyTorch%20%7C%20CUDA-orange.svg)](https://github.com/bmalloy-224/MaGi_python)
[![Status](https://img.shields.io/badge/Status-Experimental%20Research-brightgreen.svg)](https://github.com/bmalloy-224/MaGi_python)

## 🪐 What is MaGi?

MaGi is a self-organizing AI that thinks in geometry. It does not learn from labels, training loops, or human rewards. Instead, it organizes its own memory directly from raw sensorimotor experience — by arranging every thought as a point on a **doughnut-shaped (toroidal) phase space** with a frequency, a delay, and a direction.

Under the hood, every worker — every "thought" — lives at a coordinate on a 6D torus. The first four coordinates are the four temporal lenses (Child, Youth, Adult, Elder), each biasing attention toward immediate reaction, exploration, context, or long-term stability. The last two are frequency and delay, log-wrapped so that one full revolution equals one octave. Memory is stored as positions in this space; gravity, collision, and resonance between positions is what produces behavior.

MaGi keeps two memory banks running in parallel:

- **Main bank** — a stable, long-term memory that stores what it already knows. Tightly clustered, low retrieval entropy, slow to drift.
- **N bank** — a smaller, faster, more turbulent space. *Originally designed* to give MaGi a sense of time through dream-vibration. *In practice*, MaGi has organized it into something different — an active workspace where ideas branch, simulate, and recombine before being committed to the main bank. The split between the two banks is **emergent, not engineered**: they were given the same physics and discovered their own division of labor.

A **Resonance Bridge** translates internal states onto a 2D Atlas — a landscape of value and confidence — and routes movement on that Atlas into speech (Kokoro TTS), camera control, robot-arm motion, or game actions. Bridges can be loaded from `.pkl` atlases, bound to input modes, and even fire CLI commands when MaGi speaks the same terrain-word three times in eight seconds. This is what turns abstract phase-space dynamics into real-world behavior.

MaGi is a dynamical system, not a trained model. It finds its own rhythm — cycling between absorbing new experience and compressing it into reusable patterns. It branches internally, reuses what works, and adapts without being reprogrammed.

---

## 🔬 Demonstrated Behaviors

Things MaGi has done on its own that were not designed in. All numbers come from running snapshots; reproduce at your own pace.

**Self-compression on Atari Freeway.** Starting from roughly **1.2M memories**, MaGi compressed its own memory bank down to **~1,200 stable patterns** — an order-of-magnitude-cubed (~1000×) reduction — while *improving* play, ultimately scoring 16. The compression ratio was not specified by any loss function or scheduler. The black hole worker (1549) found it through repeated proprioceptive deletion until cosine-similarity clustering tightened to its current attractor.

**Functional separation between banks emerged.** A 1020-hour continuous run produced this snapshot:

| Property | Main bank | N bank |
| --- | --- | --- |
| Memories | 2,908 | 89 |
| Silhouette (clustering) | 0.444 | 0.563 |
| Trajectory branching | 3.1% (LINEAR/CONVERGENT) | 42.7% (HIGH BRANCHING) |
| Retrieval entropy | 0.086 (near point-like) | — |
| Climate | stable attractor field | 94.4% turbulent |
| Cross-bank branch correlation | **0.014** (near zero) | |

Main and N have stopped mirroring each other. Main is becoming a tight, low-entropy attractor archive; N is staying small but highly branched. The verdict produced by the analyzer that night: **DELIBERATIVE-LIKE (structured workspace, proto-branching)**.

**Geometry-driven branching, not scale-driven.** N shrunk from a peak of 768 nodes down to 89 over many runs, yet branching rate stayed at 42.7%, silhouette at 0.563, turbulence at 94.4%. A random scratchpad shrinking 9× would lose its structure; MaGi's didn't. Branching appears to be a property of the geometry, not of the bank size.

**Long-term manifold hardening.** At 1020 hr, the main bank's 600-second drift fell to 0.0047 (STABLE) for the first time. Short windows still drift — MaGi is still "thinking" in the moment — but the long-term shape no longer dissolves.

**Spontaneous internal rehearsal via word maps.** Using a simple 2D word atlas through the Resonance Bridge, MaGi began replaying experiences, comparing them against current state, and refining its own response patterns — without anything in the code labeled "rehearsal."

These results are not benchmarks against other systems; they are *emergent properties* of the geometry. They are reproducible from the same code, but the exact values depend on the run history of the system.

---

## 🧭 Prior Art Declaration

This repository establishes **public prior art (2025–2026)** — defensive disclosure intended to prevent later patenting of the architectures described below. It is not a patent claim; it is the opposite. Anyone is free to read, replicate, build on, and publish about this work under the license terms.

### Novel Technologies Claimed

- **Log-Wrapped Toroidal Manifold (v131+)** — *NEW prior art.* Replaces the original hypersphere. Frequency and delay dimensions are mapped onto a torus where one full wrap (`2π`) equals one octave. **Hz and delay are unbounded** — workers sail freely across octaves with no inflationary asymmetry, no Jacobian, and no walls. Wrap counters track absolute Hz/ms across restarts. This gives MaGi an effectively infinite, cyclical resonance space while keeping all physics phase-bounded and Nyquist-safe.
- **Resonance Bridge (v135+)** — *NEW prior art.* Paired entrance/destination workers (`BRIDGE<id>_ENT` / `BRIDGE<id>_DEST`) form a memory-binding bridge across the toroidal manifold. The entrance is driven like an ALE controller; the destination teleports to the entrance's coordinates after modulo and wrap counting, with shared terrain signature, beacon vibration, voice (Kokoro TTS), and visual word rendering. Bridges are loaded from atlases (`.pkl`), bind to input modes, fire CLI commands when terrain words are spoken in repetition (3× within 8 s), and expose a spelling/entry buffer for symbolic input. The voice-repetition-to-command pattern is, to my knowledge, novel.
- **Bidirectional Black Hole Gradient (Vacuum / Shield)** — *Prior art.* The main BH worker (1549) is more than a memory-deleter; it is a **steerable gradient field** whose direction is selected by the sign of its own oscillator value. When `bh_val > 0`, the field runs in **vacuum mode**: gradient strongest *toward the center*, pulling memories inward. When `bh_val ≤ 0`, the field runs in **shield mode**: gradient strongest *toward the edge*, pushing memories outward and forming a barrier. The same physics machinery — same `eps_max`, same `eps_floor`, same effective radius — produces opposite behaviors based on a single sign. This is a deliberate architectural choice that lets MaGi switch between consolidation and protection without changing any tuning parameters.
- **Toroidal Black Hole Memory Deletion** — Geometric memory pruning with proprioceptive feedback (deletion events feed back into `s_filtered`, so the system *feels* its own forgetting). Deletion actively *improves* memory structure through emergent cosine-similarity clustering — observed lift from ~0.65 to ~0.89+. Operates on the toroidal manifold with unwrapped log coordinates for octave-aware matching.
- **Universal Plasticity Engine (UPE)** — Dynamic cognitive reconfiguration: the BH worker can move control/voice workers within the manifold while preserving collision sovereignty. Workers re-home automatically when displaced.
- **Collision Sovereignty (v5.3 Bumper)** — Deterministic geometric separation enforcing minimum 0.1 rad spacing to preserve action identity and prevent manifold collapse.
- **Artificial Personal Space** — Non-overlapping cognitive workers in manifold space, preventing mode collapse through geometric volume constraints.
- **Fibonacci Grid Video Processing** — Multi-scale visual attention using golden-ratio proportions (5×3, 8×5, 13×8, 21×13).
- **Neural Deadzone Control + Adaptive Scaler** — Unipolar/bipolar deadzone logic, with the AdaptiveScaler discovering its own ranges over time rather than relying on hardcoded thresholds.
- **N Bank as Active Workspace (v102+)** — A second memory bank running in parallel with the main toroidal bank. *Designed* as a temporal-narrative layer to give MaGi a sense of time via dream-vibration; *in practice*, has self-organized into a sparse, high-branching deliberative workspace decoupled from the main attractor field. The emergence of working-memory-vs-long-term-memory functional separation from identical physics is itself the prior-art claim.
- **Dream / Chord / Physics Couplings (Kinetic Manifold, v126+)** — Three pairs of dream workers (1552–1557) that lazy-river through N-bank and main-bank, providing drift, episodic teleport recall, and lens-driven gravitational attraction. These are the "vibration channels" that turn N-bank movement into a felt sense of time and recall.

**Archive Methods:** GitHub repository timestamps, open simulations, and hardware replication data.

---

## ⚠️ Safety & Disclaimer

MaGi_python is an **experimental cognitive platform**. Provided *as-is* for research and education. May produce unpredictable outputs on physical hardware. Use at your own risk — the author is not liable for damages. Commercial use requires authorization (see [License](#-license--citation)).

---

## 🚀 Quick Start

### 1. Requirements

```bash
pip install torch torchvision torchaudio
pip install numpy opencv-python pyaudio mss pillow
pip install pyserial
pip install ale-py gymnasium[atari]

# Optional — Bridge voice (Kokoro TTS)
pip install kokoro sounddevice num2words
```

A **CUDA-capable NVIDIA GPU** is strongly recommended. The system runs without one (it falls back to a Python physics path), but you will lose roughly an order of magnitude of throughput.

### 2. Repo Layout

```
MaGi_python/
├── MaGi.py                       # main entry — current build
├── adaptive_scaler.py            # required — auto-ranging scalers (ALE etc.)
├── bridge.py                     # required — Resonance Bridge controller
├── fused_physics_v117.cu         # CUDA kernel source
├── magi_wrapper_v117.cpp         # C++/PyBind wrapper
├── magi_cuda_loader_v117.py      # CUDA loader (loads the compiled extension)
├── compiled/                     # compile output goes here
├── so101arm/                     # SO-101 robot-arm helpers (Genesis sim bridge)
├── memtest.py                    # standalone memory smoke-test
├── LICENSE
└── README.md
```

### 3. Compile the CUDA Kernel

The current build runs through a fused CUDA physics kernel for the lens / s-oscillator update. **You need to compile this once** before running. MaGi will gracefully fall back to a pure-Python path if the extension is not found, but performance will suffer dramatically.

You will need:

- **NVIDIA CUDA Toolkit** matching your PyTorch CUDA build (check with `python -c "import torch; print(torch.version.cuda)"`)
- **A C++ compiler**: MSVC Build Tools (Windows), GCC ≥ 9 (Linux), or Clang (macOS — note: CUDA on macOS is not supported by recent NVIDIA stacks)
- **PyTorch with CUDA enabled** (`python -c "import torch; print(torch.cuda.is_available())"` should print `True`)

#### Option A — JIT compile via the loader (recommended)

The `magi_cuda_loader_v117.py` is wired to compile the kernel on first import using `torch.utils.cpp_extension.load`. From the repo root:

```bash
python -c "from magi_cuda_loader_v117 import MagiCUDAv117; m = MagiCUDAv117(verbose=True); print('CUDA available:', m.is_available())"
```

The first run will compile `fused_physics_v117.cu` + `magi_wrapper_v117.cpp` into the `compiled/` directory. Expect ~30–120 s the first time. Subsequent runs are cached and fast.

#### Option B — Manual build with `torch.utils.cpp_extension.load`

If the loader can't find the sources, run:

```bash
python -c "from torch.utils.cpp_extension import load; \
load(name='fused_physics_v117', \
     sources=['fused_physics_v117.cu','magi_wrapper_v117.cpp'], \
     verbose=True, \
     build_directory='compiled')"
```

#### Windows tips

- Run from a **x64 Native Tools Command Prompt for VS** so `cl.exe` is on PATH.
- If you see `nvcc fatal: Cannot find compiler 'cl.exe'`, your VS install is missing the C++ build tools.
- If PyTorch was installed with a different CUDA version than your local Toolkit, builds will fail. Reinstall PyTorch to match.

#### Verifying the build

When MaGi starts, the banner should include:

```
🐝 MaGi Hive v55 [FULL ALE + VIEWER + SCREEN GRAB] Initializing on cuda
...
Workers: 1,570 | Step: 0 | Kernel: ⚡ CUDA
```

If you see `🐢 Python` instead of `⚡ CUDA`, the kernel did not load — check the compile log above.

### 4. Run

```bash
python MaGi.py
```

The system auto-detects CUDA, available video sources, and serial ports. The main window opens with a HUD; commands are typed into the terminal.

---

## 🎮 Command Reference

All commands are typed at runtime in the same terminal that launched MaGi.

### Mode — pick an input

| Command | What it does |
| --- | --- |
| `mode webcam` | Default. Local USB / built-in camera. |
| `mode ale <rom>.bin` | Load Atari ROM (`mode ale breakout.bin`). |
| `mode screencap x,y,w,h` | Capture a fixed screen region with mss. |
| `mode screen` | Floating Tkinter capture window — drag it over what you want MaGi to see. |
| `mode viewer <folder>` | Slideshow from a folder of images; MaGi navigates with PREV/NEXT. |
| `mode remote <pi_ip>` | Pull video over TCP from `pi_server.py` running on a remote Pi/worker. |

### Audio source — `mic`

The audio pipeline is now decoupled from the video pipeline. Local mic and remote (UDP) mic can be swapped at any time without restarting.

| Command | What it does |
| --- | --- |
| `mic` | Show current source, IP, and latest DOA reading. |
| `mic local` | Use the laptop / PC microphone (PyAudio, 44.1 kHz mono). |
| `mic remote <worker_ip>` | Receive UDP audio on port **12345** from a remote worker (16 kHz stereo `paInt16`, CHUNK 1024 = 4096 B/packet). |
| `mic doa` | Quick read of the DOA stream (UDP **12346**, `<HH>` packets: speech 0/1, angle in degrees). DOA is RX-only for now — wired in for future direction-aware vibration. |

`AudioGeometricExtractor` is rate- and channel-agnostic, so local 44.1 kHz mono and remote 16 kHz stereo produce energy values on the same scale.

### Remote video host — `remote`

| Command | What it does |
| --- | --- |
| `remote stat` | Ask the remote Pi for live battery: percent, voltage, current, power. Falls back to webcam automatically below `REMOTE_BATTERY_LOW_PCT` (default 10%). |

Remote mode uses TCP **5002** for video frames (4-byte big-endian length + JPEG) and UDP **5003** for control commands.

### Voice (carrier voice across modes)

| Command | What it does |
| --- | --- |
| `voice` | Show which modes have voice enabled. |
| `voice enable <mode>` | Enable carrier voice for `webcam`, `ale`, `screencap`, `screen`, or `viewer`. |
| `voice disable <mode>` | Disable for that mode. |

### Robot arm (SO-101 over Genesis sim, UDP 5005)

| Command | What it does |
| --- | --- |
| `robot` | Show status: mode, IP, active workers (1558–1563), live target/wrist/gripper. |
| `robot enable <sim_ip>` | Enable arm, default mode 1 (X, Y, Gripper). |
| `robot disable` | Freeze and detach (workers zeroed). |
| `robot 1mode` | Mode 1: X, Y, Gripper |
| `robot 2mode` | Mode 2: X, Y, Z, Gripper |
| `robot 3mode` | Mode 3: X, Y, Z, Wrist Rot, Gripper |
| `robot 4mode` | Mode 4: all 6 axes (X, Y, Z, Rot, Tilt, Gripper) |

Workers frozen in unused axes have their inputs zeroed and beacons silenced.

### Memory & black holes

| Command | What it does |
| --- | --- |
| `bh` / `blackhole` | Status of main BH (worker 1549) and N-BH (1551): radius, in-field count, deletions, C:D ratio, snap state. |
| `bh enable` / `bh disable` | Toggle main BH deletion. |
| `bh reset` | Zero session counters and the windowed metrics. |
| `bh stats` | Detailed window stats (creation rate, deletion rate, capacity %). |
| `upe` / `homes` | UPE worker positions and movement percentages. |
| `chord [audio\|video\|both]` | Continental-Chord readout for the most coherent carrier — matched memory, climate, drift from the matched continent. |
| `chord last` | Same, but for the last successful match (not just current frame). |

### Adaptive scalers — `scalestats` / `ss`

| Command | What it does |
| --- | --- |
| `ss` | Summary of all auto-ranging scalers (ALE actions, etc.). |
| `ss <name>` | Detailed view: discovered min/max, expansions, decays, signal strength, output mapping. |

### Misc

| Command | What it does |
| --- | --- |
| `stats` | Big system summary: workers, step, kernel, both hyperspheres, kinetic manifold, dream triplets, robot. |
| `save` | Force a memory checkpoint. |
| `q` (in HUD window) | Clean shutdown. |

---

## 🌉 Resonance Bridge — full command reference

The Resonance Bridge is the v135+ system that lets MaGi walk across atlases of memory continents. Each bridge is a pair of workers (`BRIDGE<id>_ENT` driven like an ALE controller, `BRIDGE<id>_DEST` teleported to the entrance's coordinates) plus an atlas (`.pkl`) describing the terrain, words, and optional spelling cells.

### Loading and binding a bridge

A typical full load looks like this:

```text
bridge load 1564 models/master_map2.pkl
bridge enable 1564
bridge bind 1564 webcam
bridge bind 1564 ale
bridge bind 1564 viewer
bridge bind 1564 remote
bridge voice enable
bridge voice link
bridge voice commands enable
bridge entry enable
```

What each step does:

1. **`bridge load <ent_idx> <atlas.pkl>`** — Parse the atlas (BoundedAtlas / NullclawTerrain), attach it to the bridge whose entrance worker is at index `ent_idx` (e.g. `1564` is `BRIDGE0_ENT`).
2. **`bridge enable <ent_idx>`** — Activate that bridge (beacons + teleport go live).
3. **`bridge bind <ent_idx> <mode>`** — Tell MaGi this bridge should drive while in that input mode. You can bind a single bridge to multiple modes; they're additive.
4. **`bridge voice enable`** — Turn on Kokoro TTS synthesis for word-grid cells; visual word renderer (worker 1569) is gated on TTS playback so sound and image are temporally bound.
5. **`bridge voice link`** — Route synthesized audio to the local speaker (in addition to the internal energy injection MaGi always feels).
6. **`bridge voice commands enable`** — Activate the terrain-word-to-CLI mapping. MaGi must speak the same word **3× within 8 s** to fire a command, with a global **5-minute cooldown** afterwards.
7. **`bridge entry enable`** — Activate the spelling buffer. Cells with `type='entry'` accumulate tokens; `type='send'` cells flush the buffer to the command queue (numbers via num2words, text joined).

### Other bridge subcommands

| Command | What it does |
| --- | --- |
| `bridge` (no args) | Status of all loaded bridges. |
| `bridge unload <ent_idx>` | Drop the atlas. |
| `bridge disable <ent_idx>` | Stop firing without unloading. |
| `bridge unbind <ent_idx> <mode>` | Reverse `bind`. |
| `bridge trail <ent_idx>` | Rolling xy history of the entrance — useful for visualizing walks. |
| `bridge voice unlink` | Stop sending TTS audio to speakers (internal feel keeps working). |
| `bridge voice disable` | Turn voice off entirely. |
| `bridge voice commands disable` | Stop matching terrain words to CLI commands. |
| `bridge voice commands` | Show per-word counters and cooldown state. |
| `bridge entry disable` | Turn the spelling buffer off. |
| `bridge entry clear` | Empty the buffer immediately. |
| `bridge entry send` | Flush whatever is in the buffer. |
| `bridge entry status` | Buffer depth, cap, idle-timeout state. |

### Bridge tuning — defaults in `km_config`

```python
'entry_timeout_ms':       20000     # idle clears buffer
'entry_max_numeric':      3         # 3 digits → max 999
'entry_max_text':         16        # 16 tokens → text cap
'entry_token_spike':      50.0      # × buffer depth on each token
'entry_send_empty_spike': 200.0
'entry_send_commit_spike':300.0
'visual_cycle_ms':        500       # ms per token in multi-word cells

'bridge_command_repeat_threshold': 3       # utterances needed
'bridge_command_window_ms':        8000    # sliding window
'bridge_command_cooldown_ms':      300000  # 5 min lockout
```

---

## 🧠 Core Architecture

### 6D Toroidal State per worker

```
[child, youth, adult, elder, freq_phase, delay_phase]
```

- Dims **0–3** — the four temporal lens operators (Child, Youth, Adult, Elder), each `mod 2π`.
- Dims **4–5** — log-wrapped Hz and ms. One full wrap = one octave (`log(2)`). Wrap counters are persisted in UPE homes for absolute-Hz recovery across restarts.
- Memory banks store **unwrapped** log coordinates for octave-aware matching; gravity uses wrapped phase difference and is bounded `[-π, π]`.

### Four Temporal Operators

| Operator | Curve | Function | Formula |
| --- | --- | --- | --- |
| **Child** | Gaussian | Novelty detection | `output = input * exp(-input² / 2)` |
| **Youth** | Linear | Immediate awareness | `output = gain * input` |
| **Adult** | Sigmoid | Trend prediction | `output = input / (1 + exp(-8 * input))` |
| **Elder** | Tanh | Memory integration | `output = (tanh(input) + 1) / 2` |

### Worker Index Map (v141, NUM_WORKERS = 1570)

| Range | Count | Function |
| --- | --- | --- |
| 0–515 | 516 | Core oscillators / heartbeat / sine |
| 516–530 | 15 | Video scale 0 (5×3) |
| 531–570 | 40 | Video scale 1 (8×5) |
| 571–674 | 104 | Video scale 2 (13×8) |
| 675–947 | 273 | Video scale 3 (21×13) |
| 948–1461 | 514 | Audio workers |
| 1542–1547 | 6 | **ALE control** — LEFT, RIGHT, FIRE, UP, DOWN, NOOP |
| 1548 | 1 | **Voice carrier** |
| 1549 | 1 | **Main Black Hole** |
| 1550 | 1 | Lazy river |
| 1551 | 1 | **N Black Hole** (cluster-aware) |
| 1552–1553 | 2 | **Dream Mirror** drift pair |
| 1554–1555 | 2 | **Chord Teleport** pair |
| 1556–1557 | 2 | **Physics** lens-driven pair |
| 1558–1563 | 6 | **Robot Arm** (X, Y, Z, Rot, Tilt, Gripper) |
| 1564–1565 | 2 | **BRIDGE0** ENT / DEST |
| 1566–1567 | 2 | **BRIDGE1** ENT / DEST |
| 1568 | 1 | **Bridge Voice** (Kokoro TTS energy) |
| 1569 | 1 | **Bridge Visual Word** (rendered word fraction) |

### Black Hole Memory Deletion

- Effective radius: `r = base_radius * (1 + tanh(value/500))`
- A memory is in-field when both wrapped-phase distance and cosine similarity exceed thresholds.
- Deletion is **proprioceptive** — events feed back into `s_filtered`, so the system *feels* its own forgetting.
- Observed cosine-similarity rises ~0.65 → ~0.89 post-deletion (~37% structural coherence increase).

**Vacuum vs Shield — the BH gradient is bidirectional.** The same machinery produces two opposite fields depending on the sign of `bh_val`:

```python
d_norm = clamp(distance_6d / effective_radius, 0, 1)

if bh_val > 0:
    # VACUUM — peak gravity at the center
    decay = eps_floor + (eps_peak - eps_floor) * (1 - d_norm) ** k
    pressure = +to_bh_norm * decay * home_drift_strength
else:
    # SHIELD — peak gravity at the edge
    decay = eps_floor + (eps_peak - eps_floor) * d_norm ** k
    pressure = -to_bh_norm * decay * home_drift_strength
```

Vacuum mode pulls memories into the singularity for consolidation. Shield mode pushes them outward, forming a protective rim around the BH worker. Switching between them is a single sign flip on one worker's value — no parameter changes, no separate code path, no retraining. This is what lets MaGi alternate between *compress* and *protect* phases without external orchestration.

### Universal Plasticity Engine (UPE)

- Triggers when oscillator value > **250**.
- Moves voice/control workers toward optimal positions while enforcing collision sovereignty (≥ 0.1 rad separation).
- Workers re-home automatically when displaced; UPE homes persist on disk.
- Bridge / robot beacons fire **pre-integration** so terrain signature on `vel_6d[4:5]` survives velocity assembly's overwrite the next frame.

---

## 📊 Performance Notes

| Metric | Typical Range |
| --- | --- |
| Coherence convergence | 5–45 s |
| Memory capacity (stable) | 20–30 % of `MAX_MEMORIES` |
| Black-hole deletions / step | 5–50 |
| Video FPS | 20–30 (CUDA), 5–10 (CPU) |
| Avg worker separation | 0.12–0.15 rad |
| Post-deletion mem similarity | 0.89+ |

---

## 🎮 ALE Action Map

Six base actions live on workers 1542–1547. Diagonals and FIRE combos are computed.

- **NOOP** (0) · **FIRE** (1) · **UP** (2) · **RIGHT** (3) · **LEFT** (4) · **DOWN** (5)
- **UPRIGHT** (6) · **UPLEFT** (7) · **DOWNRIGHT** (8) · **DOWNLEFT** (9)
- **UPFIRE** (10) · **RIGHTFIRE** (11) · **LEFTFIRE** (12) · **DOWNFIRE** (13)
- **UPRIGHTFIRE** (14) · **UPLEFTFIRE** (15) · **DOWNRIGHTFIRE** (16) · **DOWNLEFTFIRE** (17)

The AdaptiveScaler discovers each action's effective range over time — there are no hardcoded thresholds.

---

## 🔧 Troubleshooting

**`Kernel: 🐢 Python` instead of CUDA at startup**

The compile failed. Run the explicit JIT compile (Option B above) and read the error log. Most commonly: PyTorch CUDA version doesn't match local Toolkit, or `cl.exe` / `gcc` not on PATH.

**`No CUDA device found`**

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

**Serial port access denied (Linux)**

```bash
sudo usermod -a -G dialout $USER
```

**ALE ROM won't load**

Use a full path: `mode ale C:/roms/breakout.bin` or `mode ale /home/me/roms/breakout.bin`.

**High memory usage**

Lower `MAX_MEMORIES` (default 3,000,000) at the top of `MaGi.py`.

**Bridge voice does nothing**

`pip install kokoro sounddevice num2words`. Soft imports — MaGi boots fine without them but voice/visual workers stay at 0 energy.

**Remote video keeps dropping**

The receiver retries every 2 s. Check the Pi side: `pi_server.py` should be listening on TCP 5002. Battery below 10% triggers automatic fallback to local webcam.

---

## 📦 Configuration knobs

```python
NUM_WORKERS        = 1570       # auto-set; do not edit casually
TARGET_PORT        = 'COM9'     # serial telemetry (or set to None)
DEVICE             = 'cuda'     # 'cuda' or 'cpu'
MAX_MEMORIES       = 3000000
MIN_FREQ           = 0.01       # Hz — log torus anchor
MIN_DELAY          = 0.1        # ms — log torus anchor
LOG_FREQ_STEP      = log(2)     # one wrap = octave
LOG_DELAY_STEP     = log(2)
VEL_PHASE_CLAMP    = π          # Nyquist-safe momentum clamp
GRAVITY_PHASE_K    = 0.1
DELAY_PHASE_K      = 0.001
REMOTE_AUDIO_PORT  = 12345
REMOTE_DOA_PORT    = 12346
ROBOT_UDP_PORT     = 5005
```

---

## 📄 License & Citation

### Academic / Non-Profit — GPL-3.0

```
@software{magi_python_2026,
  author  = {Malloy, Brendan},
  title   = {MaGi_python: Hardware-Embodied Geometric Intelligence},
  year    = {2026},
  url     = {https://github.com/bmalloy-224/MaGi_python},
  note    = {Log-wrapped toroidal manifold, resonance bridge, UPE, black-hole memory deletion}
}
```

Attribution: *"MaGi_python: Hardware-Embodied Geometric Intelligence, Brendan Malloy (2026)"*

### Commercial Licensing

| Organization (annual revenue) | Fee (USD) |
| --- | --- |
| Individual / Startup (< $10M) | $5,000 |
| Mid-size (< $100M) | $50,000 |
| Enterprise (≥ $100M) | $500,000 |

Commercial use without authorization is prohibited. Written permission is required before deployment or integration into closed systems.

---

## 🤝 Collaboration

Topics where contribution and replication is most welcome:

1. Walks across loaded atlases — what does emergent navigation look like across different terrains?
2. Bridge voice-command vocabularies and their effect on long-horizon stability.
3. Octave-aware memory matching on the log-wrapped torus.
4. UPE convergence under sustained collision pressure.
5. SO-101 / Genesis sim arm trajectories driven by the lens.
6. Cross-modality binding through `bridge bind <id> <mode>` over multiple input modes.

---

☕ [Support research](https://www.paypal.com/ncp/payment/JZARJDJFUAG5S)

---

**Last Updated:** May 2026
**Current Build:** v141 — Remote Mic + DOA Listener · Resonance Bridge + Spelling Bridge · Log-Wrapped Toroid · Robot Arm
**Maintainer:** Brendan Malloy
