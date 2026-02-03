```markdown
# 🌀 MaGi_python — Malloy artificial Geometric intelligence

**Hardware-Embodied Systems using python**  
> Exploring how curve-based operators produce emergent cognition through geometric constraints. Uses fibonacci-scale visual attention, 4D hypersphere manifolds, hypersphere black hole memory management, and dynamic worker movement (UPE).

[![License](https://img.shields.io/badge/License-GPL--3.0%20%2B%20Commercial-blue.svg)](https://github.com/bmalloy-224/MaGi_python/blob/main/LICENSE)
[![Research](https://img.shields.io/badge/Research-Geometric%20Intelligence-green.svg)](https://github.com/bmalloy-224/MaGi_python)
[![Platform](https://img.shields.io/badge/Platform-Python%203.x%20%7C%20PyTorch%20%7C%20CUDA-orange.svg)](https://github.com/bmalloy-224/MaGi_python)
[![Status](https://img.shields.io/badge/Status-Experimental%20Research-brightgreen.svg)](https://github.com/bmalloy-224/MaGi_python)

---

## 🧭 Prior Art Declaration

This repository establishes **public prior art (2026)** for hardware-embodied geometric intelligence with advanced neural control systems.  
Specifically, it documents that 4 lens curves, prime timing, timing directionality, **hypersphere black hole memory deletion**, **Universal Plasticity Engine (UPE)** control systems, and **geometric self-preservation through collision sovereignty** **determine AI cognitive architecture**.

### Novel Technologies Claimed:
- **Hypersphere Black Hole Memory Deletion Worker**: Geometric memory management using black hole physics principles for intelligent memory pruning in hypersphere space with **proprioceptive feedback anchoring** — deletion actively **improves memory structure through emergent cosine similarity clustering**
- **Universal Plasticity Engine (UPE)**: Allows dynamic movement of control/voice workers within the hypersphere, enabling cognitive reconfiguration **while maintaining collision sovereignty**
- **Collision Sovereignty (v5.3 Bumper)**: Deterministic geometric separation enforcing minimum 0.1 radian spacing to preserve action identity and prevent manifold collapse
- **Artificial Personal Space**: Implementation of non-overlapping cognitive workers in hypersphere manifolds, preventing mode collapse through geometric volume constraints
- **Fibonacci Grid Video Processing**: Multi-scale visual attention using golden ratio proportions (5×3, 8×5, 13×8, 21×13)
- **Neural Deadzone Control**: Unipolar and bipolar deadzone logic for stable AI-to-system control

**Archive Methods**
* GitHub repository timestamp (2026)
* Open simulations and hardware replication data

---

## ⚠️ Safety & Disclaimer

MaGi_python is an **experimental cognitive platform**.

* Provided *as-is* for research and education
* May produce unpredictable outputs on physical hardware
* Use at your own risk — the author is not liable for damages
* Commercial use requires authorization (see [License](#license--citation))

---

## 🎯 Discovery Overview

MaGi_python demonstrates that geometric intelligence expresses through:

* **Prime number timing bases**
* **Memory field dynamics**
* **Control system configuration**
* **Geometric volume (worker density)**

| Configuration | Worker Density | Emergent Style |
| --- | --- | --- |
| **CUDA (GPU)** | Sparse (0.15 rad avg) | "Parallel Harmonizer" — massive-scale coherent discovery |
| **CPU (Python)** | Medium (0.12 rad avg) | "Adaptive Explorer" — flexible, noise-resilient cognition |
| **Dense Mode** | Dense (0.10 rad min) | "Strategic Balancer" — optimal resource allocation |

> 0.05 radian spacing differences produce emergent behavioral patterns observable as "anxious/reactive" or "deliberate/stable" cognitive modes.

---

## 🔬 Research Basis & Emergent Properties

MaGi implements **minimal geometric rules** that produce **complex emergent cognition**:

### 🧩 The Emergence Principle
Intelligence emerges from constrained geometry, not from explicit programming. MaGi's simple rules produce behaviors that can be described in anthropomorphic terms, but arise purely from systemic interactions.

### 🏆 Novel Contribution Statement

While individual components build on established research, MaGi's **architectural synthesis** establishes prior art for:

| **Prior Art Claim** | **Distinction from Existing Work** |
|---------------------|-----------------------------------|
| **Dynamic Geometric Sovereignty** | Real-time worker movement with collision prevention in learning systems |
| **Memory-as-Proprioceptive-Feedback** | Deletion events directly influencing immediate action selection |
| **Density-Determined Cognition** | Worker spacing as primary determinant of behavioral mode |
| **Embodied Fibonacci Attention** | Multi-scale vision directly coupled to motor control loops |

**Key Insight**: Intelligence is what happens *between* the lines of code — in the geometric relationships and dynamic interactions of the system.

---

## 🚀 Quick Start

### Requirements

```bash
pip install torch torchvision torchaudio
pip install numpy opencv-python pyaudio
pip install pyserial mss pillow
pip install ale-py gymnasium[atari]
```

### Hardware (Optional)
* Webcam or video input device
* Audio input device
* Serial device (Arduino/Teensy) for telemetry output
* CUDA-capable GPU (recommended for real-time performance)

### Basic Usage

```bash
# Run with webcam input
python MaGi_vp01.py

# The system will auto-detect:
# - Available video sources (webcam, ALE, screen, viewer)
# - CUDA availability
# - Serial ports
```

---

## 🎮 Command Reference

MaGi_python supports multiple operational modes and commands:

### Mode Selection
Type these commands at runtime:
- `mode webcam` - Use local webcam as video source
- `mode robot lowrez` - Robot camera: 640×480 @ 120 FPS, H.264
- `mode robot highrez` - Robot camera: 1920×1440 @ 30 FPS, H.265
- `mode robot` - Default to lowrez mode
- `mode ale <game>.bin` - Load Atari game (e.g., `mode ale breakout.bin`)
- `mode screen` - Capture screen content
- `mode viewer` - Neural-controlled image viewer (expects `images/` subfolder)

### Audio Control
- `mic local` - PC microphone (pyaudio)
- `mic robot` - Pi microphone (UDP 5004)
- `mic both` - Mix local + robot (pressure superposition)
- `mic off` - Silence input

### Runtime Commands
During execution:
- `BH` - Display black hole worker information:
  - Worker index, effective radius, memories in field
  - Step deletions, capacity percentage
  - Bumper activity, collision events
  - Memory structure observations
- `UPE` - Universal Plasticity Engine status:
  - Worker positions and movement percentages
- `stats` - System coherence and memory count
- `save` - Checkpoint memory bank
- `reset` - Zero motor workers (1542-1547)
- `estop` - Hardware emergency stop

### Example Sessions

```bash
# Play Breakout with ALE
$ python MaGi_vp01.py
> mode ale breakout.bin

# View black hole status
> BH
Black Hole Worker: 1549
Effective Radius: 0.2847
Memories in Field: 142
Step Deletions: 8
Capacity: 23.45%
BH Value: 347.2
Bumper Activity: 0.847
Collision Events: 0
Memory Structure: 0.892
```

---

## 🧠 Core Technologies

### Four Temporal Operators

| Operator | Curve | Function | Formula |
| --- | --- | --- | --- |
| **Child** | Gaussian | Novelty detection | `output = input * exp(-input²/2)` |
| **Youth** | Linear | Immediate awareness | `output = gain * input` |
| **Adult** | Sigmoid | Trend prediction | `output = input / (1 + exp(-8*input))` |
| **Elder** | Tanh | Memory integration | `output = (tanh(input) + 1)/2` |

---

### 🛡️ Collision Sovereignty (v5.3 Bumper)

**Geometric Self-Preservation System**

Workers maintain **minimum 0.1 radian separation** to preserve action identity.

```python
MIN_SEPARATION = 0.1  # radians

def enforce_collision_sovereignty(workers):
    for i, worker_i in enumerate(workers):
        for j, worker_j in enumerate(workers[i+1:], start=i+1):
            distance = torch.acos(torch.clamp(
                torch.dot(worker_i.position, worker_j.position), -1, 1
            ))
            
            if distance < MIN_SEPARATION:
                repulsion = (worker_i.position - worker_j.position)
                repulsion = repulsion / torch.norm(repulsion)
                violation = (MIN_SEPARATION - distance) / MIN_SEPARATION
                force = repulsion * violation * 0.1
                
                worker_i.position += force
                worker_j.position -= force
                worker_i.position /= torch.norm(worker_i.position)
                worker_j.position /= torch.norm(worker_j.position)
    
    return workers
```

| Worker Density | Min Separation | Emergent Style |
|----------------|----------------|----------------|
| Sparse (0.20+ rad) | 0.10 rad | Deliberate/Stable |
| Medium (0.12-0.20 rad) | 0.10 rad | Balanced/Adaptive |
| Dense (0.10-0.12 rad) | 0.10 rad | Anxious/Reactive |
| Critical (<0.10 rad) | ENFORCED | System prevents |

---

### Hypersphere Black Hole Memory Deletion

- **Location**: Worker 1549 (geometrically locked)
- **Function**: Geometric memory pruning
- **Radius**: `effective_radius = base_radius * (1 + tanh(value/500))`
- **Proprioceptive Feedback**: Deletion events influence system state
- **Emergent Structure**: Deletion produces tighter cosine similarity clustering

```python
similarity = torch.nn.functional.cosine_similarity(memory_vector, current_state)
distance = torch.norm(memory_position - bh_position)
in_field = (distance < effective_radius) and (similarity > threshold)

if in_field: 
    delete_memory()
```

**Observed Effects**:
- Pre-deletion similarity: ~0.65
- Post-deletion similarity: ~0.89+
- Improvement: ~37% structural coherence increase

---

### Universal Plasticity Engine (UPE)

Dynamic cognitive reconfiguration with geometric sovereignty.

- **Trigger**: Oscillator value > 250
- **Method**: Gradient-based movement
- **Preservation**: Workers 1542-1549 maintain learned positions
- **File**: `motor_voice_map.pth`

```python
if control_value > 250:
    target_position = find_optimal_hypersphere_position()
    proposed_position = lerp(current_pos, target_pos, plasticity_rate)
    safe_position = enforce_collision_sovereignty([proposed_position], all_workers)
    voice_worker_position = safe_position
```

---

### Fibonacci Grid Video Processing

| Scale | Grid | Workers | Range |
| --- | --- | --- | --- |
| 0 | 5×3 | 15 | 516-530 |
| 1 | 8×5 | 40 | 531-570 |
| 2 | 13×8 | 104 | 571-674 |
| 3 | 21×13 | 273 | 675-947 |

**Total**: 432 video workers

---

## 📊 Worker Index Map

| Range | Count | Function | Status |
|-------|-------|----------|--------|
| 0-515 | 516 | Core oscillators | Protected |
| 516-530 | 15 | Video scale 0 | Protected |
| 531-570 | 40 | Video scale 1 | Protected |
| 571-674 | 104 | Video scale 2 | Protected |
| 675-947 | 273 | Video scale 3 | Protected |
| 948-1461 | 514 | Audio workers | Protected |
| **1542-1547** | **6** | **ALE control** | **Geometrically Locked** |
| **1548** | **1** | **Voice carrier** | **Geometrically Locked** |
| **1549** | **1** | **Black hole** | **Geometrically Locked** |
| 1550+ | ~6549 | Lazy river | Protected |

---

## 📈 Performance Observations

### Emergent Behavioral Modes

| Density | Behavior | Description |
|---------|----------|-------------|
| Dense (0.10-0.12 rad) | Anxious/Reactive | High-frequency oscillation |
| Medium (0.12-0.20 rad) | Balanced/Adaptive | Flexible exploration |
| Sparse (0.20+ rad) | Deliberate/Stable | Coherent pattern recognition |

### Telemetry

| Metric | Range | Notes |
|--------|-------|-------|
| Coherence convergence | 5-45 seconds | Self-organization time |
| Memory capacity (stable) | 20-30% of max | Equilibrium point |
| Black hole deletions/step | 5-50 | Dynamic pruning |
| Video processing | 20-30 FPS | Real-time |
| Collision events/hour | 0-5 | Constraint effectiveness |
| Worker separation (avg) | 0.12-0.15 rad | Emergent distribution |
| Memory similarity (post-deletion) | 0.89+ | Emergent clustering |

---

## 🎮 ALE Game Control

### Base Actions (Workers 1542-1547, Geometrically Locked)
- **NOOP** (0), **FIRE** (1), **UP** (2), **RIGHT** (3), **LEFT** (4), **DOWN** (5)

### Computed Actions
- Diagonals: **UPRIGHT** (6), **UPLEFT** (7), **DOWNRIGHT** (8), **DOWNLEFT** (9)
- Fire combos: **UPFIRE** (10), **RIGHTFIRE** (11), **LEFTFIRE** (12), **DOWNFIRE** (13), **UPRIGHTFIRE** (14), **UPLEFTFIRE** (15), **DOWNRIGHTFIRE** (16), **DOWNLEFTFIRE** (17)

---

## 📄 License & Citation

### Academic Use — GPL-3.0

**Attribution:** "*MaGi_python: Hardware-Embodied Geometric Intelligence*, Brendan Malloy (2026)"

```bibtex
@software{magi_python_2026,
  author = {Malloy, Brendan},
  title = {MaGi_python: Hardware-Embodied Geometric Intelligence},
  year = {2026},
  url = {https://github.com/bmalloy-224/MaGi_python},
  note = {Geometric Intelligence Platform with Emergent Behavioral Modes}
}
```

### Commercial Licensing

| Organization | Fee |
| --- | --- |
| Startup | $5,000 |
| Mid-size | $50,000 |
| Enterprise | $500,000 |

Commercial use without authorization is prohibited.

---

## 🤝 Collaboration Invitation

1. Document emergent behavioral patterns across density ranges
2. Examine UPE convergence with geometric constraints
3. Develop game strategies using geometrically constrained workers
4. Observe black hole effects on memory structure emergence
5. Validate density-cognition relationship across tasks
6. Study geometric constraint effectiveness
7. Investigate artificial personal space in other architectures
8. Document emergent similarity clustering dynamics
9. Analyze proprioceptive feedback in memory systems

---

## 🛠 Technical Architecture

### System Requirements

- **Python**: 3.8+
- **PyTorch**: 1.10+ (CUDA recommended)
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: NVIDIA with 4GB+ VRAM (optional)
- **Storage**: 500MB code, up to 10GB memory files

### Configuration

```python
NUM_WORKERS = 8100
TARGET_PORT = 'COM9'        # Serial output (or None)
DEVICE = 'cuda'             # or 'cpu'
MAX_MEMORIES = 3000000
MIN_SEPARATION = 0.1        # v5.3 Geometric constraint (radians)
```

### Performance Modes

| Mode | Device | FPS | Worker Density |
|------|--------|-----|----------------|
| Real-time | CUDA | 20-30 | Medium (0.12 rad) |
| Balanced | CUDA | 10-20 | Sparse (0.15 rad) |
| Analysis | CPU | 1-5 | Dense (0.10 rad) |

---

## 🔬 Research Contributions

- **Emergent Cognitive Modes**: Density-dependent behavioral phenotypes
- **Geometric Identity Preservation**: Action distinctness through enforced spacing
- **Proprioceptive Memory Management**: Deletion influencing ongoing cognition
- **Constrained Emergence**: Simple rules producing complex behaviors
- **Fibonacci Harmonic Attention**: Multi-scale perception through golden ratios
- **Dynamic Geometric Sovereignty**: Real-time collision prevention

---

## 🔧 Troubleshooting

**No CUDA device found**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Serial port access denied**
```bash
sudo usermod -a -G dialout $USER  # Linux
```

**ALE game won't load**
```bash
python MaGi_vp01.py
> mode ale /path/to/breakout.bin
```

**High memory usage**
```python
MAX_MEMORIES = 1000000  # Reduce from 3000000
```

**Worker constraint warnings**
```bash
# Check BH command for statistics
# Adjust MIN_SEPARATION (0.08-0.15 rad range)
```

---

## 🎓 Educational Use

- Emergent intelligence in geometrically constrained systems
- Memory management with proprioceptive feedback
- Neural control interfaces with geometric guarantees
- Multi-scale visual attention through harmonic ratios
- Prime number resonance in timing systems
- Geometric identity preservation
- Artificial personal space for embodied cognition
- Density-dependent behavioral emergence
- Constraint-based self-organization
- Emergent structure through selective forgetting

---



☕ [Support research](https://www.paypal.com/ncp/payment/JZARJDJFUAG5S)

---

## 📦 Repository Contents

- `MaGi_vp01.py` - Main implementation (v5.3 Geometric Constraints)
- `README.md` - This file
- `LICENSE` - GPL-3.0 with commercial terms
- `motor_voice_map.pth` - UPE mappings (generated)
- `magi_memory.pt` - Memory checkpoint (generated)

---

**Last Updated**: February 2026  
**Version**: p01 (Memory-Safe Fibonacci Grids + v5.3 Geometric Sovereignty)  
**Maintainer**: Brendan Malloy
```
