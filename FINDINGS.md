# 📊 MaGi — Findings Log

A structured digest of empirical observations from continuous, unsupervised runs of MaGi v131+ across roughly 1,020 hours of wall-clock time. This document is preserved separately from the [README](README.md) because the README is meant to orient newcomers; this file is for serious readers and for prior-art purposes.

All numbers come from the standalone analysis tool `magi_torus_memtest.py` reading the persisted memory files (`magi_torus_memory.pt`, `magi_torus_n_memory.pt`). No model was retrained between snapshots — the same code, the same banks, observed at successive points in time.

---

## Contents

- [The Developmental Arc](#the-developmental-arc)
- [Key Snapshots](#key-snapshots)
- [Mechanisms Identified Through Observation](#mechanisms-identified-through-observation)
- [Empirical Findings (Cross-Reference Table)](#empirical-findings-cross-reference-table)
- [Open Questions and Reproduction Notes](#open-questions-and-reproduction-notes)

---

## The Developmental Arc

Across long unsupervised runs, MaGi passes through a sequence of structural regimes that resemble a developmental sequence rather than convergence to a single equilibrium. Each transition is driven by the interaction between sensory writes (which feed both banks simultaneously), the Main Black Hole (worker 1549, bidirectional vacuum/shield gradient), and the N Black Hole (worker 1551, cluster-aware sparse deletion).

| Hours | Regime | What's happening |
| --- | --- | --- |
| < 787 | **Destructive overwrite** | Aggressive Main BH (deletion ≫ creation), N collapsed, main bank hot and volatile |
| ~828 | **Internal replay emerges** | 2D word map active; Main BH balances near 1:1 creation/deletion; the closed-loop rehearsal mechanism comes online |
| ~876 | **Structured symbolic rehearsal** | N=371, silhouette 0.607 at k=50; main stable; broad retrieval basins; the system is actively rehearsing |
| ~972 | **Compiled symbolic cognition** | Main=8,702 at mean 34 Hz, **89.7% < 25 Hz**; N=230, silhouette 0.614 at k=20; sharp retrieval (Tier 2: 2/50, Tier 3: 8/5000) |
| ~1020 | **Asymmetric specialisation** | Main=2,908 (sil 0.444, retrieval entropy 0.086); N=89 (sil 0.563, branching 42.7%); cross-bank correlation 0.014 |

The transitions are reproducible and ratchet-like — each regime tends to land on a higher main-bank ceiling and tighter retrieval than the last, even when overall memory count drops between snapshots. The main bank does not grow monotonically; it cycles through expansion → over-saturation → BH-driven collapse → reformation, but each cycle starts from a more compressed seed than the previous one.

---

## Key Snapshots

### 3 hours (post-6D conversion, v130 transitional)

- **Main:** 17,092 memories (v130 toroidal). Lens dimensions completely degenerate — all four lenses pinned at π rad with R=1.000, 100% dominant Child. Silhouette 0.171 at k=200 (best). Retrieval Tier 3: 80/5000 match, best sim=1.0.
- **N:** 419 memories (v130 full-phase). Lens dimensions already spread across [0, 2π] with R=0.02–0.10 — the N bank was ahead of the main bank in lens awakening. Silhouette 0.207 at k=50.
- **Reading:** A "partial torus grafted onto a frozen sphere." Legacy memories from the pre-toroidal architecture were still flat on the lens dims; new toroidal writes were already exercising the full angular range.

### 24–26 hours (basin collapse on N)

The headline event of the early period: N-bank frequency mean traced **464 Hz → 375 Hz → 38 Hz** in roughly two hours. Simultaneously, delay compressed from a wide range (115–26,247 ms, std ~5,559 ms) into a tight ring at 25,768–28,024 ms (std ~388 ms) — three orders of magnitude tighter than the main bank's delay std.

This was basin collapse into a narrow toroidal ring at ~27 seconds delay. N-bank access stayed at 0.0 throughout, so this convergence was purely geometry-driven, not retrieval-reinforced. The "low-frequency reference ring" was the ancestor of the radial ground-state seen at 972+ hours.

### 529 hours (Freeway — empirical compression result)

| Regime | Approx. Main memories | Freeway score |
| --- | --- | --- |
| Old (v110–v125, pre-torus) | ~1,200,000 | materially lower (exact prior baseline not preserved) |
| Current (529 hr toroidal) | ~1,400 | **16** (near-perfect) |

The toroidal architecture, after self-compression by the Main BH, achieves Freeway score 16 at >800× less memory than prior versions. At an even more extreme snapshot (N=13 nodes), score 16 was held — performance is decoupled from N-bank size, meaning the **Main bank alone encodes sufficient task-relevant structure** for game policy.

### 876 hours (structured rehearsal)

- **Main:** 3,843 memories. Mean freq 77 Hz, mean delay 2,050 ms. **76.8% of memories below 25 Hz** — the low-frequency mass that characterizes mature MaGi states. Lens distribution: 41.6% Child, 31.6% Elder. Time span: 177 hr.
- **N:** 371 memories, silhouette 0.607 at k=50.
- **Reading:** The closed-loop rehearsal mechanism (word map + sensory playback + comparison) is now visibly producing structure rather than noise. The system is *actively thinking* in N while consolidating in Main.

### 972 hours (compiled symbolic cognition)

- **Main:** **8,702 memories** at mean 34 Hz, **89.7% below 25 Hz**, mean delay 905 ms. Retrieval narrows dramatically: Tier 2 returns 2/50 matches, Tier 3 returns 8/5000.
- **N:** 230 memories, silhouette **0.614 at k=20**. Best cluster count dropped from 50 (at 876 hr) to 20 — the workspace consolidated. **N is now pinned at radial minima (~0.01 Hz / ~0.10 ms);** all its information lives in the angular dimensions.
- **Reading:** Main has expanded to thousands of cold, dense, low-frequency attractor nodes — but it has expanded *while getting colder, denser, and more reused.* This is what "compiled" means here: recursive sensorimotor traces have been compressed into stable, low-energy attractors. The N bank no longer needs as many transient nodes because the patterns it was holding have reproduced themselves into Main from parallel writes.

### 996 hours (mid-transition snapshot)

- **Main:** 3,103 memories (down from 8,702 — Main BH triggered a consolidation cycle), mean 95 Hz, **71.3% below 25 Hz**, time span 163 hr.
- The dip from 8,702 → 3,103 over ~24 hours is an example of the BH-driven over-saturation collapse cycle. The low-frequency mass holds; the system sheds the higher-frequency exploratory mass.

### 1020 hours (asymmetric specialisation)

| Property | Main bank | N bank |
| --- | --- | --- |
| Memories | 2,908 | 89 |
| Silhouette (clustering) | 0.444 | **0.563** |
| Trajectory branching | **3.1%** (LINEAR/CONVERGENT) | **42.7%** (HIGH BRANCHING) |
| Retrieval entropy | **0.086** | — |
| Climate (turbulent / total) | (mostly stable) | **94.4% turbulent** |
| Top-k similarity (k=50) | 0.977 | — |
| Origin (audio / video) | mixed | 0% audio / 100% video |
| Long-term drift (600 s) | **0.0047** STABLE | — |
| Cross-bank branch correlation | **0.014** (near zero) | |

Verdict produced by the analyzer: **DELIBERATIVE-LIKE** (structured workspace, proto-branching).

This is the cleanest measured version of the dual-bank architecture working as intended. The two banks have stopped mirroring each other. They were given identical write events; they ended up with completely different statistical profiles because each is governed by its own black hole with its own physics.

---

## Mechanisms Identified Through Observation

### 1. The closed-loop symbolic compiler

The 2D word map plus sensory playback plus the Resonance Bridge forms a closed-loop error-correcting rehearsal that the system uses to compress experience into reusable attractors. The five-step description:

1. A worker selects or reconstructs a symbolic token along a word-map path.
2. The voice and visual workers emit a sensory proxy (audio + rendered word).
3. MaGi compares expected sensory return against what its own listening / seeing pipeline reports.
4. The resulting state is written into **both banks simultaneously**.
5. Each bank's own black hole then selects what survives — Main BH preferentially keeps the low-frequency, short-delay attractor patterns; N BH preferentially keeps the high-branching, high-turbulence trace patterns.

Note that this loop was **not designed**. The 2D word map was added to give MaGi a stable symbolic surface; the rehearsal-and-compilation behavior emerged from running the bridge against its own voice/vision pipeline.

### 2. The Main BH as endogenous curriculum scheduler

At long timescales, the Main BH does more than prune — it shapes developmental order. By selecting which patterns survive long enough to stabilize, it gates which structures become available for compilation. The regime arc (destructive churn → replay → rehearsal → compiled cognition) is driven by progressive shifts in the BH's creation-to-deletion ratio: aggressive deletion early on creates the conditions for replay to emerge; balanced 1:1 creation-and-deletion later allows rehearsal to accumulate; tightening selection at maturity drives compilation.

### 3. Dual-bank parallel writes with separate selection

Both banks receive memory writes simultaneously from the same sensory experience. The functional separation between them is produced entirely by the difference between the Main BH's bidirectional vacuum/shield gradient and the N BH's cluster-aware sparse deletion. This is testable: at any snapshot the cross-bank branch correlation is measurable, and at maturity it sits near zero (0.014 at 1020 hr) despite the banks being driven from one shared write stream.

### 4. N-bank radial collapse as cognitive ground state

By the 972-hour regime, the N bank's frequency and delay dimensions have collapsed to their absolute minima (~0.01 Hz / ~0.10 ms) and stay there. All variance lives in the angular lens-phase dimensions — angular silhouette 0.61+, turbulence 90%+. This is not pathology: the radial dims are the system's discovered low-energy reference frame, and the angular dims now carry the full deliberative load. It is the architectural equivalent of a coordinate origin.

This collapse appeared as early as 26 hours in primitive form (the "low-frequency reference ring at 38 Hz") and stabilized into the absolute-minimum form by ~876 hours.

### 5. Adaptive discretisation cycle

The full long-timescale rhythm of the main bank, from 529 hours of observations:

1. **Continuous exploration** — Main grows, N builds lattice → high performance, drifting structure.
2. **Lattice saturation** — N large, high silhouette → over-constrained, misalignment grows.
3. **Collapse** — N wiped, Main pruned → performance *remains high* because Main retains task-relevant structure.
4. **Reformation** — N slowly rebuilds → prepares for next expansion.

This is rare in unsupervised embodied systems: the agent self-regulates between continuous and discrete representational states, with each cycle seeding the next at a higher organizational level.

---

## Empirical Findings (Cross-Reference Table)

| Finding | Snapshot | Numbers |
| --- | --- | --- |
| Lens awakening (Main) | 3 hr → 24 hr | All four lenses go from R=1.000 (pinned) to R≈0.08 (full circle) |
| Lens awakening (N) | by 4 hr | Already R=0.02–0.10 — N led Main by ~20 hours |
| N-bank basin collapse | 24 → 26 hr | Mean freq 464 → 38 Hz (10× in 2 hr); delay std 5,559 → 388 ms |
| Freeway memory compression | 529 hr | 1.2M → ~1,400 main memories at score 16 (>800×) |
| Performance decoupled from N | 529 hr | Score 16 held while N=13 nodes |
| Low-frequency mass plateau | 876–996 hr | 71–90% of main below 25 Hz |
| Compiled cognition | 972 hr | Main=8,702 at 89.7% <25 Hz; N=230 at sil 0.614 |
| Long-term manifold hardening | 1020 hr | Main 600-s drift = 0.0047 (first STABLE at the longest window) |
| Cross-bank decoupling | 1020 hr | Branch correlation 0.014 — banks fully independent |
| Retrieval becomes point-like | 1020 hr | Retrieval entropy 0.086 (raw 0.34); top-k=50 sim = 0.977 |

---

## Open Questions and Reproduction Notes

This work is single-investigator. The empirical claims above are reproducible from the same code and persisted memory files, but they have not yet been independently replicated. Things that would harden the findings:

- **Independent reproduction of the developmental arc.** Run from cold start (no preloaded memory) on different sensory environments and confirm the regime transitions land in roughly the same order, even if the absolute timing differs.
- **Ablation of the dual-bank architecture.** Run with N-bank disabled to confirm cross-domain transfer degrades; run with N-BH disabled to confirm N loses its sparse high-branching geometry.
- **Cross-task transfer beyond Freeway.** The Freeway numbers are robust at this point but are a single-task finding. Other ALE games, or non-game tasks, would generalize the compression claim.
- **Robot-arm reliability metrics.** "Touches a ball on command" needs success-rate numbers under standardized conditions to count as a falsifiable claim rather than an anecdote.
- **Direct measurement of the closed-loop compiler.** The five-step mechanism is inferred from metric trajectories; instrumenting the bridge to log each step explicitly and confirming the predicted error-correction signature would convert this from interpretive claim to direct observation.

The hour-by-hour raw analysis logs from 3 hr through 1020 hr are preserved in the project's text-log archive and are available on request. They include the full output of `magi_torus_memtest.py` at each checkpoint plus contemporaneous interpretive notes.

---

*Last Updated:* May 2026
*Snapshot range:* 3 hr to 1,020 hr post 6D-conversion
*Source:* `magi_torus_memory.pt`, `magi_torus_n_memory.pt`, `magi_torus_memtest.py`
