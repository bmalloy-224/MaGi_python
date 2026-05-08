"""
MaGi v131 Memory Analysis — Dual Bank Edition (Log-Wrapped Torus)
Refined: vectorized cross-bank binning, reused cluster labels in N bank,
extracted execution-topology helper, cleaner epsilon in load balance.

Compatibility: reads files saved by MaGi v140 main bank (mapping_version=1)
and N bank. All field names match (`freq`/`delay` plus `metadata_*` aliases
both handled by `_safe_get`). Constants match the main program:
  MIN_FREQ=0.01, MIN_DELAY=0.1, LOG_FREQ_STEP=LOG_DELAY_STEP=log(2).
"""

import argparse
import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# -- Constants ----------------------------------------------------------------
MIN_FREQ       = 0.01
MIN_DELAY      = 0.1
LOG_FREQ_STEP  = math.log(2)
LOG_DELAY_STEP = math.log(2)
TWO_PI = 2 * math.pi

V130_MIN_FREQ,  V130_MAX_FREQ  = 0.01, 500.0
V130_MIN_DELAY, V130_MAX_DELAY = 0.1,  20000.0

DEFAULT_MAIN_FILE = "magi_torus_memory.pt"
DEFAULT_N_FILE    = "magi_torus_n_memory.pt"

LENS_NAMES  = ['child', 'youth', 'adult', 'elder']
LENS_COLORS = ['#4fc3f7', '#81c784', '#ffb74d', '#e57373']


# ----------------------------------------------------------------------------
# Helpers (with local RNG)
# ----------------------------------------------------------------------------

def _resolve_file(primary, legacy):
    if os.path.exists(primary):
        return primary
    if os.path.exists(legacy):
        print(f"  info: Using legacy file: {legacy}")
        return legacy
    return primary

def _norm(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norms + 1e-8)

def _silhouette(arr_norm, rng, step_sizes=(10, 20, 50, 100, 200)):
    results = []
    for k in step_sizes:
        if k >= len(arr_norm) // 5:
            break
        km = MiniBatchKMeans(n_clusters=k, random_state=int(rng.integers(2**32)), n_init=5, batch_size=4096)
        labels = km.fit_predict(arr_norm)
        score  = silhouette_score(arr_norm, labels, sample_size=min(5000, len(arr_norm)))
        results.append((k, score))
    return results

def _temporal_stability(arr_norm, timestamps, rng, windows=(60, 300, 600)):
    tmax = timestamps.max()
    rows = []
    for w in windows:
        mask = timestamps > (tmax - w)
        n    = mask.sum()
        if n < 10:
            rows.append((w, n, None, None, None))
            continue
        rec = arr_norm[mask]
        n_old = int((~mask).sum())
        if n_old == 0:
            # No historical data outside the window (e.g. all timestamps equal,
            # or window covers the whole bank). "Stable" is meaningless here.
            rows.append((w, n, None, None, None))
            continue
        old = arr_norm[~mask]
        # sample recent-recent pairs
        n_rr = min(5000, n * (n-1) // 2)
        if n_rr > 0:
            rr_idx = rng.integers(0, n, size=(n_rr, 2))
            # ensure i != j
            rr_same = rr_idx[:,0] == rr_idx[:,1]
            while rr_same.any():
                rr_idx[rr_same,1] = rng.integers(0, n, size=rr_same.sum())
                rr_same = rr_idx[:,0] == rr_idx[:,1]
            sim_rr = (rec[rr_idx[:,0]] * rec[rr_idx[:,1]]).sum(axis=1).mean()
        else:
            sim_rr = 0.0

        # sample recent-old pairs
        n_ro = min(5000, n * len(old))
        ro_idx_i = rng.integers(0, n, size=n_ro)
        ro_idx_j = rng.integers(0, len(old), size=n_ro)
        sim_ro = (rec[ro_idx_i] * old[ro_idx_j]).sum(axis=1).mean()

        drift = abs(sim_rr - sim_ro)
        rows.append((w, n, sim_rr, sim_ro, drift))
    return rows

def _detect_branching(arr_norm, timestamps, rng, time_window=600, sim_threshold=0.75, div_threshold=0.4):
    order = np.argsort(timestamps)
    arr = arr_norm[order]
    ts = timestamps[order]

    branches = 0
    step = max(1, len(arr) // 2000)
    start = rng.integers(0, step)
    sampled_indices = range(start, len(arr), step)

    for i in sampled_indices:
        f_mask = (ts > ts[i]) & (ts <= ts[i] + time_window)
        f_idx = np.where(f_mask)[0]
        if len(f_idx) < 2:
            continue
        f_arr = arr[f_idx]
        sims = f_arr @ arr[i]
        conn_mask = sims > sim_threshold
        if conn_mask.sum() < 2:
            continue
        conn_arr = f_arr[conn_mask]
        # cap connected set size to avoid explosion
        if len(conn_arr) > 128:
            sel = rng.choice(len(conn_arr), 128, replace=False)
            conn_arr = conn_arr[sel]
        sim_mat = conn_arr @ conn_arr.T
        triu = np.triu_indices_from(sim_mat, k=1)
        if len(triu[0]) == 0:
            continue
        min_sim = sim_mat[triu].min()
        if min_sim < div_threshold:
            branches += 1

    rate = branches / len(sampled_indices) if len(sampled_indices) > 0 else 0
    return branches, rate

def _safe_get(data, *keys, fallback=None, size=None):
    if not isinstance(data, dict):
        print("  warning: data is not a dict, using fallback")
        return fallback if fallback is not None else torch.zeros(size or 0)
    for k in keys:
        if k in data:
            return data[k]
    if fallback is not None:
        return fallback
    return torch.zeros(size or 0)


# ----------------------------------------------------------------------------
# Execution topology classification (heuristic, shared by analyze_n + verdict)
# ----------------------------------------------------------------------------

def _execution_topology(best_partition, branch_rate, bsr):
    """
    Classify execution topology from branching diagnostics.

    branch_score = partition density (capped 1.0 at k>=50)
                 + fork rate * 10   (typical max ~5.0 if every sample forks)
                 + branch separation (capped 1.0 at bsr>=2)

    Labels:
        BRANCHED-LIKE     score>=2.0 AND fork_rate>0.03 AND bsr>1.5
        DELIBERATIVE-LIKE score>=1.2 OR  (k>15 AND fork_rate>0.01 AND bsr>1.2)
        UNITARY-LIKE      otherwise

    Returns: (label, score)
    """
    partition_term  = best_partition / 50.0 if best_partition < 50 else 1.0
    rate_term       = branch_rate * 10.0
    separation_term = bsr / 2.0 if bsr < 2 else 1.0
    score = partition_term + rate_term + separation_term

    if score >= 2.0 and branch_rate > 0.03 and bsr > 1.5:
        return 'BRANCHED-LIKE', score
    if score >= 1.2 or (best_partition > 15 and branch_rate > 0.01 and bsr > 1.2):
        return 'DELIBERATIVE-LIKE', score
    return 'UNITARY-LIKE', score


# ----------------------------------------------------------------------------
# v131 mapping & circular stats (with validation)
# ----------------------------------------------------------------------------

def _log_coord_to_freq(log_coord):
    return MIN_FREQ * np.exp(log_coord * LOG_FREQ_STEP)

def _log_coord_to_delay(log_coord):
    return MIN_DELAY * np.exp(log_coord * LOG_DELAY_STEP)

def _v130_phase_to_freq(phase):
    return (phase / TWO_PI) * (V130_MAX_FREQ - V130_MIN_FREQ) + V130_MIN_FREQ

def _v130_phase_to_delay(phase):
    return (phase / TWO_PI) * (V130_MAX_DELAY - V130_MIN_DELAY) + V130_MIN_DELAY

def _circ_mean(phases):
    return np.arctan2(np.sin(phases).mean(), np.cos(phases).mean()) % TWO_PI

def _circ_var(phases):
    R = np.sqrt(np.sin(phases).mean()**2 + np.cos(phases).mean()**2)
    return 1.0 - R

def _circ_concentration(phases):
    return np.sqrt(np.sin(phases).mean()**2 + np.cos(phases).mean()**2)


# ----------------------------------------------------------------------------
# Console bar charts (log-spaced bins, guarded)
# ----------------------------------------------------------------------------

def _safe_geomspace(start, stop, num):
    # Always returns strictly monotonic edges so np.histogram / ax.hist never crash.
    # np.nextafter guarantees a distinct next-representable float at any scale,
    # avoiding any concern about start*1e-6 underflowing into start.
    if start <= 0:
        start = 1e-6
    if stop <= start or abs(stop - start) < 1e-6 * max(1.0, abs(start)):
        return np.array([start, np.nextafter(start, np.inf)])
    return np.geomspace(start, stop, num)

def _hz_bar_chart(hz, size, bins=20):
    low = max(hz.min(), 1e-6)
    hi = hz.max()
    if hi <= low * 1.000001:
        print(f"   {low:.2f}–{hi:.2f} Hz: {size:,} (100.0%)")
        return
    edges = _safe_geomspace(low, hi, bins + 1)
    counts, _ = np.histogram(hz, bins=edges)
    for i in range(len(counts)):
        if counts[i] > 0:
            lo = edges[i]
            hi = edges[i+1]
            pct = 100 * counts[i] / size
            bar = '█' * int(pct / 0.5)
            print(f"   {lo:8.2f}–{hi:8.2f} Hz: {counts[i]:7,} ({pct:5.1f}%)  {bar}")

def _ms_bar_chart(ms, size, bins=20):
    low = max(ms.min(), 1e-6)
    hi = ms.max()
    if hi <= low * 1.000001:
        print(f"   {low:.2f}–{hi:.2f} ms: {size:,} (100.0%)")
        return
    edges = _safe_geomspace(low, hi, bins + 1)
    counts, _ = np.histogram(ms, bins=edges)
    for i in range(len(counts)):
        if counts[i] > 0:
            lo = edges[i]
            hi = edges[i+1]
            pct = 100 * counts[i] / size
            bar = '█' * int(pct / 0.5)
            print(f"   {lo:8.2f}–{hi:8.2f} ms: {counts[i]:7,} ({pct:5.1f}%)  {bar}")


# ----------------------------------------------------------------------------
# Retrieval diagnostics (corrected)
# ----------------------------------------------------------------------------

def _topk_similarity_concentration(arr_norm, queries, ks=(1,5,10,50)):
    results = {k: [] for k in ks}
    for q in queries:
        sim = arr_norm @ q
        idx = np.argsort(sim)[::-1]
        for k in ks:
            results[k].append(sim[idx[:k]].mean())
    return {k: np.mean(v) for k, v in results.items()}

def _interference_index(arr_norm, query_idx, thresholds=(0.95,0.90,0.85)):
    results = {th: [] for th in thresholds}
    for qi in query_idx:
        sim = arr_norm @ arr_norm[qi]
        sim[qi] = -np.inf
        for th in thresholds:
            results[th].append(np.sum(sim > th))
    return {th: np.mean(v) for th, v in results.items()}

def _retrieval_entropy(arr_norm, query_idx, labels, k=50):
    if len(query_idx) == 0:
        return 0.0, 0.0
    # k_eff and n_clusters do not depend on the query — compute once.
    k_eff = min(k, len(arr_norm) - 1)
    n_clusters = len(np.unique(labels))
    entropies = []
    for qi in query_idx:
        sim = arr_norm @ arr_norm[qi]
        sim[qi] = -np.inf
        top_idx = np.argsort(sim)[-k_eff:]
        cl = labels[top_idx]
        _, counts = np.unique(cl, return_counts=True)
        probs = counts / len(top_idx)
        ent = -np.sum(probs * np.log(probs + 1e-8))
        entropies.append(ent)
    mean_ent = float(np.mean(entropies))
    max_ent = np.log(min(n_clusters, k_eff))
    norm_ent = mean_ent / max_ent if max_ent > 0 else 0.0
    return norm_ent, mean_ent

def _mutual_symmetry(arr_norm, query_idx, n_neighbors=10):
    if len(arr_norm) < 2:
        return 0.0
    ratios = []
    for qi in query_idx:
        sim_q = arr_norm @ arr_norm[qi]
        sim_q[qi] = -np.inf
        top_idx = np.argsort(sim_q)[-n_neighbors:]
        for t in top_idx:
            sim_t = arr_norm @ arr_norm[t]
            sim_t[t] = -np.inf
            t_top = np.argsort(sim_t)[-n_neighbors:]
            ratios.append(1.0 if qi in t_top else 0.0)
    return np.mean(ratios) if ratios else 0.0


# ----------------------------------------------------------------------------
# Branch structure analysis (corrected)
# ----------------------------------------------------------------------------

def _branch_separation_ratio(coords_n, labels, n_clusters, rng, max_sample=5000):
    bsr_list = []
    for k in range(n_clusters):
        mask = labels == k
        n_in = int(mask.sum())
        if n_in < 2:
            continue
        pts = coords_n[mask]
        other = coords_n[~mask] if (~mask).sum() > 0 else pts

        # internal similarity
        if n_in <= 500:
            sim_in = cosine_similarity(pts)
            np.fill_diagonal(sim_in, np.nan)
            internal = np.nanmean(sim_in)
        else:
            n_pairs = min(n_in * (n_in - 1) // 2, max_sample)
            i_idx = rng.integers(0, n_in, size=n_pairs)
            j_idx = rng.integers(0, n_in, size=n_pairs)
            same = (i_idx == j_idx)
            while same.any():
                j_idx[same] = rng.integers(0, n_in, size=same.sum())
                same = (i_idx == j_idx)
            internal = (pts[i_idx] * pts[j_idx]).sum(axis=1).mean()

        # cross similarity
        n_other = len(other)
        if n_other == 0:
            continue
        n_cross = min(n_in * n_other, max_sample)
        i_cross = rng.integers(0, n_in, size=n_cross)
        j_cross = rng.integers(0, n_other, size=n_cross)
        sim_cross = (pts[i_cross] * other[j_cross]).sum(axis=1).mean()

        if sim_cross > 0:
            bsr_list.append(internal / sim_cross)
    return np.mean(bsr_list) if bsr_list else 0.0

def _branch_persistence_score(coords_n, timestamps, n_clusters, rng):
    tmax = timestamps.max()
    mask_600 = timestamps > (tmax - 600)
    mask_300 = timestamps > (tmax - 300)
    if mask_600.sum() < 10 or mask_300.sum() < 10:
        return 0.0

    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=int(rng.integers(2**32)), n_init=5, batch_size=4096)
    km.fit(coords_n[mask_600])
    centroids_600 = km.cluster_centers_

    labels_300 = km.predict(coords_n[mask_300])
    centroids_300 = np.zeros_like(centroids_600)
    for j in range(n_clusters):
        pts = coords_n[mask_300][labels_300 == j]
        if len(pts) > 0:
            centroids_300[j] = pts.mean(axis=0)
        else:
            centroids_300[j] = centroids_600[j]

    # Hungarian matching to align cluster order (robust against future
    # refactors that fit centroids_300 independently).
    sim = cosine_similarity(centroids_600, centroids_300)
    row_ind, col_ind = linear_sum_assignment(sim, maximize=True)
    matched_sim = sim[row_ind, col_ind].mean()
    return matched_sim

def _branch_load_balance(access, labels, n_clusters):
    cluster_access = []
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            cluster_access.append(access[mask].mean())
    if len(cluster_access) < 2:
        return 0.0
    arr = np.asarray(cluster_access, dtype=np.float64)
    mean = arr.mean()
    # If mean is effectively zero (fresh bank, all access==0), CV is undefined;
    # return 0.0 to denote "perfectly balanced (because nothing has happened)".
    if mean < 1e-12:
        return 0.0
    return arr.std() / mean


# ----------------------------------------------------------------------------
# Cross-bank branch correlation (vectorized binning)
# ----------------------------------------------------------------------------

def _cross_bank_branch_correlation(main_timestamps, main_arr, n_timestamps, n_arr,
                                   rng, time_window=600, sim_threshold=0.75, div_threshold=0.4,
                                   bin_size=60, max_checks=2000):
    def _fork_signal(timestamps, arr):
        order = np.argsort(timestamps)
        arr_s = arr[order]
        ts_s = timestamps[order]
        step = max(1, len(ts_s) // max_checks)
        start = rng.integers(0, step)
        forks = np.zeros(len(ts_s), dtype=bool)
        checked = np.zeros(len(ts_s), dtype=bool)   # which indices were evaluated
        for i in range(start, len(ts_s), step):
            checked[i] = True
            f_mask = (ts_s > ts_s[i]) & (ts_s <= ts_s[i] + time_window)
            f_idx = np.where(f_mask)[0]
            if len(f_idx) < 2:
                continue
            sims = arr_s[f_idx] @ arr_s[i]
            conn_mask = sims > sim_threshold
            if conn_mask.sum() < 2:
                continue
            conn_arr = arr_s[f_idx][conn_mask]
            if len(conn_arr) > 128:
                sel = rng.choice(len(conn_arr), 128, replace=False)
                conn_arr = conn_arr[sel]
            sim_mat = conn_arr @ conn_arr.T
            triu = np.triu_indices_from(sim_mat, k=1)
            if len(triu[0]) > 0 and sim_mat[triu].min() < div_threshold:
                forks[i] = True
        return ts_s, forks, checked

    ts_m, fork_m, checked_m = _fork_signal(main_timestamps, main_arr)
    ts_n, fork_n, checked_n = _fork_signal(n_timestamps, n_arr)

    t_min = min(ts_m.min(), ts_n.min())
    t_max = max(ts_m.max(), ts_n.max())
    bins = np.arange(t_min, t_max + bin_size, bin_size)
    n_bins = len(bins) - 1

    def _bin_rates(ts, forks, checked):
        # Vectorized: for each timestamp find its bin via searchsorted, then
        # bincount over the subset that was actually checked.
        if n_bins <= 0 or not checked.any():
            return np.zeros(max(n_bins, 0))
        bin_idx = np.searchsorted(bins, ts, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        bin_checked = bin_idx[checked]
        fork_at_checked = forks[checked].astype(np.float64)
        fork_sum      = np.bincount(bin_checked, weights=fork_at_checked, minlength=n_bins)
        checked_count = np.bincount(bin_checked,                          minlength=n_bins)
        rates = np.zeros(n_bins)
        nz = checked_count > 0
        rates[nz] = fork_sum[nz] / checked_count[nz]
        return rates

    rates_m = _bin_rates(ts_m, fork_m, checked_m)
    rates_n = _bin_rates(ts_n, fork_n, checked_n)

    if len(rates_m) > 0 and len(rates_n) > 0 and rates_m.std() > 0 and rates_n.std() > 0:
        corr = np.corrcoef(rates_m, rates_n)[0, 1]
    else:
        corr = 0.0
    return corr, rates_m, rates_n


# ----------------------------------------------------------------------------
# Main bank analysis (with full fixes)
# ----------------------------------------------------------------------------

def analyze_main(path, rng, branch_sim_thresh, branch_div_thresh):
    print(f"\n{'='*60}")
    print(f"MAIN BANK -- {path}")
    print(f"{'='*60}")

    # (Warning: torch.load with weights_only=False is used; only trusted files)
    path = _resolve_file(path, "magi_v10x_memory.pt")
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return None

    data = torch.load(path, map_location='cpu', weights_only=False)
    size = data['size']
    mems = data['memories'][:size].numpy().astype(np.float32)
    meta_freq  = data['meta_freq'][:size].numpy()
    meta_delay = data['meta_delay'][:size].numpy()
    access     = data['access_counts'][:size].numpy()
    timestamps = data['timestamps'][:size].numpy()

    mapping_version = data.get('mapping_version', 0)
    is_v131 = (mapping_version == 1)

    has_6d = 'mem_coords_6d' in data
    if has_6d:
        coords_6d = data['mem_coords_6d'][:size].numpy().astype(np.float32)
        if coords_6d.shape[1] < 6:
            raise ValueError(f"mem_coords_6d has {coords_6d.shape[1]} columns, need >=6")
        lens_phases = coords_6d[:, :4]
        freq_data   = coords_6d[:, 4]
        delay_data  = coords_6d[:, 5]
        if is_v131:
            freq_hz  = _log_coord_to_freq(freq_data)
            delay_ms = _log_coord_to_delay(delay_data)
        else:
            freq_hz  = _v130_phase_to_freq(freq_data)
            delay_ms = _v130_phase_to_delay(delay_data)
    else:
        lens_phases = None
        freq_hz  = meta_freq
        delay_ms = meta_delay

    print(f"  Loaded {size:,} memories  dim={data.get('dim', mems.shape[1])}")
    if has_6d:
        if is_v131:
            print(f"  Format: v131 log-wrapped torus (unwrapped log coords)")
        else:
            print(f"  Format: v130 linear toroidal (wrapped phases)")
    else:
        print(f"  Format: pre-v130 (scalar Hz/ms, no 6D coords)")

    # -- Summary ---
    print(f"\nSummary (Hz/ms):")
    print(f"   Freq:  {freq_hz.min():.2f} – {freq_hz.max():.2f} Hz  mean={freq_hz.mean():.2f}")
    print(f"   Delay: {delay_ms.min():.2f} – {delay_ms.max():.2f} ms  mean={delay_ms.mean():.2f}")

    print(f"\nFreq distribution (log-spaced bins):")
    _hz_bar_chart(freq_hz, size)

    print(f"\nDelay distribution (log-spaced bins):")
    _ms_bar_chart(delay_ms, size)

    if has_6d and lens_phases is not None:
        print(f"\nLens phase decomposition (6D dims 0-3):")
        for i, name in enumerate(LENS_NAMES):
            ph = lens_phases[:, i]
            # Validate if truly circular
            if np.all((ph >= 0) & (ph <= TWO_PI)):
                cm = _circ_mean(ph)
                cv = _circ_var(ph)
                R  = _circ_concentration(ph)
                print(f"   {name:6s}: circ_mean={cm:.3f} rad  circ_var={cv:.3f}  "
                      f"R={R:.3f}  [{ph.min():.3f}, {ph.max():.3f}]")
            else:
                print(f"   {name:6s}: (not circular) mean={ph.mean():.3f} std={ph.std():.3f}  "
                      f"[{ph.min():.3f}, {ph.max():.3f}]")
        # Dominant-lens counts (argmax across the 4 phase dims). Note: phases
        # are circular, so this is a convenience visual — comparable across runs
        # but not a circularly-correct statistic. Preserved for backward
        # compatibility with the original v131 metrics.
        dominant = np.argmax(lens_phases, axis=1)
        for i, name in enumerate(LENS_NAMES):
            cnt = int((dominant == i).sum())
            print(f"   Dominant {name}: {cnt:,} ({100*cnt/size:.1f}%)")

    print(f"\n   Access: {access.min():.0f} - {access.max():.0f}  mean={access.mean():.1f}")
    print(f"   Time span: {(timestamps.max()-timestamps.min())/3600:.1f} hr")
    print(f"\nAccess patterns:")
    p75 = np.percentile(access, 75)
    hi  = access > p75
    print(f"   Top-25% memories: {hi.sum():,}  mean_access={access[hi].mean():.1f}")
    print(f"   All memories:      mean_access={access.mean():.1f}")

    # -- Clustering ---
    mems_n = _norm(mems)
    print(f"\nClustering (cosine space):")
    sil = _silhouette(mems_n, rng)
    best_sil = 0.0
    best_partition = 0
    if sil:
        for k, s in sil:
            mark = " <-- best" if s == max(v for _, v in sil) else ""
            print(f"   k={k:4d}  silhouette={s:.4f}{mark}")
            if s > best_sil:
                best_sil = s
                best_partition = k
    else:
        print("   (insufficient samples for silhouette sweep)")

    # -- Tier effectiveness (original) ---
    print(f"\nTier effectiveness (sample query):")
    tidx = rng.integers(0, size)
    q    = mems_n[tidx:tidx+1]
    print(f"   Query: freq={freq_hz[tidx]:.2f} Hz  delay={delay_ms[tidx]:.2f} ms")
    gal_mean  = mems_n.mean(axis=0, keepdims=True)
    gal_mean /= (np.linalg.norm(gal_mean) + 1e-8)
    sim_gal = (q @ gal_mean.T).item()
    print(f"   Tier 1 (Galaxy): sim={sim_gal:.4f}")
    if size > 500:
        n_sol = min(50, size // 20)
        km = MiniBatchKMeans(n_clusters=n_sol, random_state=int(rng.integers(2**32)), n_init=5, batch_size=4096)
        km.fit(mems_n)
        cen = _norm(km.cluster_centers_)
        sol_s = (q @ cen.T)[0]
        print(f"   Tier 2 (Solar):  {(sol_s>0.70).sum()}/{n_sol} match  best={sol_s.max():.4f}")
    else:
        print(f"   Tier 2 (Solar):  skipped (size <= 500)")
    samp_idx = rng.choice(size, min(5000, size), replace=False)
    pl_s = (q @ mems_n[samp_idx].T)[0]
    print(f"   Tier 3 (Planet): {(pl_s>0.85).sum()}/{len(samp_idx)} match  best={pl_s.max():.4f}")

    # -- Temporal stability ---
    print(f"\nTemporal stability:")
    stab = _temporal_stability(mems_n, timestamps, rng)
    for w, n, si, sc, dr in stab:
        if si is None:
            print(f"   Last {w:4d}s: {n} memories -- insufficient")
        else:
            flag = "STABLE" if dr < 0.01 else "DRIFT"
            print(f"   Last {w:4d}s ({n:4d} mems): internal={si:.4f}  cross={sc:.4f}  drift={dr:.4f}  {flag}")

    # -- Branching & retrieval (Main) ---
    print(f"\nTrajectory Branching (Main):")
    b_count, b_rate = _detect_branching(mems_n, timestamps, rng,
                                         sim_threshold=branch_sim_thresh,
                                         div_threshold=branch_div_thresh)
    print(f"   Forks detected: {b_count} ({(b_rate*100):.1f}% of sampled nodes)")
    print(f"   Status: {'HIGH BRANCHING' if b_rate > 0.05 else 'LINEAR/CONVERGENT'}")

    n_queries = min(128, size // 100)
    if n_queries >= 10:
        query_idx = rng.choice(size, n_queries, replace=False)
        queries = mems_n[query_idx]
        print(f"\nRetrieval Diagnostics (batch of {n_queries} queries):")
        topk = _topk_similarity_concentration(mems_n, queries)
        print(f"   Top-k similarity concentration (mean similarity):")
        for k, v in topk.items():
            print(f"      k={k:2d}: {v:.4f}")
        interfer = _interference_index(mems_n, query_idx)
        print(f"   Interference index (mean #neighbors, excluding self):")
        for th, cnt in interfer.items():
            print(f"      sim>{th:.2f}: {cnt:.1f}")
        if best_partition > 1 and len(mems_n) >= best_partition * 5:
            km_main = MiniBatchKMeans(n_clusters=best_partition, random_state=int(rng.integers(2**32)), n_init=5, batch_size=4096)
            labels_main = km_main.fit_predict(mems_n)
            norm_ent, raw_ent = _retrieval_entropy(mems_n, query_idx, labels_main, k=50)
            print(f"   Retrieval entropy (normalized, 0=concentrated, 1=diffuse): {norm_ent:.3f} (raw={raw_ent:.2f})")
        else:
            print(f"   Retrieval entropy: insufficient clusters")
        sym = _mutual_symmetry(mems_n, query_idx, n_neighbors=10)
        print(f"   Mutual recall symmetry: {sym:.3f} (1.0 = perfect symmetry)")

    # -- Full plotting --
    # 4 columns when 6D coords are present (adds the lens phase bar chart and
    # the lens-coloured freq/delay scatter, matching the original v131 layout).
    n_cols = 4 if has_6d else 3
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9))
    title = f"Main Bank -- {size:,} memories  {'v131 log' if is_v131 else 'v130 linear'}"
    fig.suptitle(title, fontsize=14)

    samp = rng.choice(size, min(5000, size), replace=False)

    # Silhouette
    ax = axes[0,0]
    if sil:
        ks, ss = zip(*sil)
        ax.plot(ks, ss, 'o-', color='steelblue')
        ax.axhline(0.2, color='green', ls='--', alpha=0.6, label='0.2 (good)')
        ax.axhline(0.1, color='orange', ls='--', alpha=0.6, label='0.1 (marginal)')
    else:
        ax.text(0.5,0.5,"insufficient data", ha='center')
    ax.set_xlabel('k')
    ax.set_ylabel('Silhouette')
    ax.set_title('Clustering Quality')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Frequency distribution (log bins)
    ax = axes[0,1]
    if freq_hz.max() > freq_hz.min() * 1.01:
        bins = _safe_geomspace(max(freq_hz.min(),1e-6), freq_hz.max(), 60)
        ax.hist(freq_hz, bins=bins, color='steelblue', alpha=0.7, edgecolor='white')
        ax.set_xscale('log')
    else:
        ax.hist(freq_hz, bins=20, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(freq_hz.mean(), color='red', lw=2, ls='--', label='mean')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Count')
    ax.set_title('Frequency Distribution (log bins)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Delay distribution (log bins)
    ax = axes[0,2]
    if delay_ms.max() > delay_ms.min() * 1.01:
        bins = _safe_geomspace(max(delay_ms.min(),1e-6), delay_ms.max(), 60)
        ax.hist(delay_ms, bins=bins, color='coral', alpha=0.7, edgecolor='white')
        ax.set_xscale('log')
    else:
        ax.hist(delay_ms, bins=20, color='coral', alpha=0.7, edgecolor='white')
    ax.axvline(delay_ms.mean(), color='red', lw=2, ls='--', label='mean')
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Delay Distribution (log bins)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Lens phase bar chart (only when has_6d) — circular mean + concentration R
    if has_6d:
        ax = axes[0,3]
        lens_cmeans = [_circ_mean(lens_phases[:, i]) for i in range(4)]
        lens_Rs     = [_circ_concentration(lens_phases[:, i]) for i in range(4)]
        x_pos = np.arange(4)
        ax.bar(x_pos, lens_cmeans, color=LENS_COLORS, alpha=0.8, edgecolor='white')
        for i, (cm, R) in enumerate(zip(lens_cmeans, lens_Rs)):
            ax.text(i, cm + 0.1, f'R={R:.2f}', ha='center', fontsize=7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(LENS_NAMES)
        ax.set_ylabel('Circ Mean Phase (rad)')
        ax.set_title('Lens Phases (circ mean + R)')
        ax.axhline(math.pi, color='gray', ls=':', alpha=0.5, label='π (neutral)')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Freq vs Delay (access coloured)
    ax = axes[1,0]
    sc = ax.scatter(freq_hz[samp], delay_ms[samp], c=np.log1p(access[samp]), alpha=0.3, s=5, cmap='plasma')
    plt.colorbar(sc, ax=ax, label='log(access)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Delay (ms)')
    ax.set_title('Freq x Delay (colour=access)')
    ax.grid(alpha=0.3)

    # Access vs freq
    ax = axes[1,1]
    ax.scatter(freq_hz[samp], access[samp], alpha=0.2, s=5, color='steelblue')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Access')
    ax.set_title('Access vs Freq')
    ax.grid(alpha=0.3)

    # Access vs delay
    ax = axes[1,2]
    ax.scatter(delay_ms[samp], access[samp], alpha=0.2, s=5, color='coral')
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Access')
    ax.set_title('Access vs Delay')
    ax.grid(alpha=0.3)

    # Freq x Delay coloured by dominant lens (only when has_6d)
    if has_6d:
        ax = axes[1,3]
        dominant = np.argmax(lens_phases[samp, :4], axis=1)
        c = [LENS_COLORS[d] for d in dominant]
        ax.scatter(freq_hz[samp], delay_ms[samp], c=c, alpha=0.3, s=5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Delay (ms)')
        ax.set_title('Freq x Delay (colour=dominant lens)')
        for i, name in enumerate(LENS_NAMES):
            ax.scatter([], [], c=LENS_COLORS[i], label=name, s=20)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('analysis_main.png', dpi=150)
    plt.close()
    print("\nSaved: analysis_main.png")

    return {'size': size, 'best_sil': best_sil, 'best_partition': best_partition,
            'freq_hz': freq_hz, 'delay_ms': delay_ms,
            'access': access, 'timestamps': timestamps, 'has_6d': has_6d, 'is_v131': is_v131,
            'lens_phases': lens_phases, 'branch_rate': b_rate, 'mems_n': mems_n}


# ----------------------------------------------------------------------------
# N bank analysis (with full fixes — single KMeans fit, shared labels)
# ----------------------------------------------------------------------------

def analyze_n(path, rng, branch_sim_thresh, branch_div_thresh):
    print(f"\n{'='*60}")
    print(f"N BANK -- {path}")
    print(f"{'='*60}")

    path = _resolve_file(path, "n_v111_memory.pt")
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return None

    data = torch.load(path, map_location='cpu', weights_only=False)
    size = data['size']
    coords = data['coords'][:size].numpy().astype(np.float32)
    if coords.shape[1] < 7:
        raise ValueError(f"coords has {coords.shape[1]} columns, need >=7")
    meta_freq  = _safe_get(data, 'freq', 'metadata_freq', fallback=torch.zeros(size))[:size].numpy()
    meta_delay = _safe_get(data, 'delay', 'metadata_delay', fallback=torch.full((size,), 5.0))[:size].numpy()
    access     = data['access_counts'][:size].numpy()
    timestamps = data['timestamps'][:size].numpy()
    origin     = _safe_get(data, 'origin', fallback=torch.zeros(size))[:size].numpy()
    t_ce       = _safe_get(data, 'tension_ce', fallback=torch.zeros(size))[:size].numpy()
    t_ae       = _safe_get(data, 'tension_ae', fallback=torch.zeros(size))[:size].numpy()
    version    = _safe_get(data, 'version', fallback=torch.zeros(size))[:size].numpy()

    mapping_version = data.get('mapping_version', 0)
    is_v131 = (mapping_version == 1)

    # Detect lens metadata across any of the four lens fields. If a save
    # somehow lacks 'child' but has the other three, _safe_get's fallback
    # for the missing one yields zeros — better than ignoring the rest.
    _LENS_KEYS = (
        ('child', 'metadata_child'),
        ('youth', 'metadata_youth'),
        ('adult', 'metadata_adult'),
        ('elder', 'metadata_elder'),
    )
    has_lens = any(k in data for pair in _LENS_KEYS for k in pair)
    if has_lens:
        lens_child = _safe_get(data, 'child', 'metadata_child', fallback=torch.zeros(size))[:size].numpy()
        lens_youth = _safe_get(data, 'youth', 'metadata_youth', fallback=torch.zeros(size))[:size].numpy()
        lens_adult = _safe_get(data, 'adult', 'metadata_adult', fallback=torch.zeros(size))[:size].numpy()
        lens_elder = _safe_get(data, 'elder', 'metadata_elder', fallback=torch.zeros(size))[:size].numpy()

    n_dim = coords.shape[1]
    # n_dim is guaranteed >= 7 by the early check above; the lens / freq /
    # delay branches that follow are unconditional. Kept verbose for clarity
    # rather than gated, since the early check is the single source of truth.
    lens_phases = coords[:, :4]

    if is_v131:
        freq_log  = coords[:, 5]
        delay_log = coords[:, 6]
        freq_hz   = _log_coord_to_freq(freq_log)
        delay_ms  = _log_coord_to_delay(delay_log)
    else:
        freq_hz   = _v130_phase_to_freq(coords[:, 5])
        delay_ms  = _v130_phase_to_delay(coords[:, 6])

    print(f"  Loaded {size:,} memories  coords={n_dim}D")
    if is_v131:
        print(f"  Format: v131 log-wrapped torus (unwrapped log coords)")
    else:
        print(f"  Format: v130/v129 (linear phases or scaled norm)")

    # -- Summary ---
    print(f"\nSummary (Hz/ms):")
    print(f"   Freq:  {freq_hz.min():.2f} – {freq_hz.max():.2f} Hz  mean={freq_hz.mean():.2f}")
    print(f"   Delay: {delay_ms.min():.2f} – {delay_ms.max():.2f} ms  mean={delay_ms.mean():.2f}")

    print(f"\nFreq distribution (N bank, log-spaced bins):")
    _hz_bar_chart(freq_hz, size)

    print(f"\nDelay distribution (N bank, log-spaced bins):")
    _ms_bar_chart(delay_ms, size)

    print(f"\nLens phase decomposition (dims 0-3):")
    for i, name in enumerate(LENS_NAMES):
        ph = lens_phases[:, i]
        if np.all((ph >= 0) & (ph <= TWO_PI)):
            cm = _circ_mean(ph)
            cv = _circ_var(ph)
            R  = _circ_concentration(ph)
            print(f"   {name:6s}: circ_mean={cm:.3f} rad  circ_var={cv:.3f}  "
                  f"R={R:.3f}  [{ph.min():.3f}, {ph.max():.3f}]")
        else:
            print(f"   {name:6s}: (not circular) mean={ph.mean():.3f} std={ph.std():.3f}  "
                  f"[{ph.min():.3f}, {ph.max():.3f}]")
    # Dominant-lens counts (argmax across phase dims). Convenience visual,
    # preserved for backward compatibility — see analyze_main for caveat.
    dominant = np.argmax(lens_phases, axis=1)
    for i, name in enumerate(LENS_NAMES):
        cnt = int((dominant == i).sum())
        print(f"   Dominant {name}: {cnt:,} ({100*cnt/size:.1f}%)")

    print(f"\n   Access: {access.min():.0f} - {access.max():.0f}  mean={access.mean():.1f}")
    print(f"   Time span: {(timestamps.max()-timestamps.min())/3600:.1f} hr")
    n_audio = (origin < 0.5).sum()
    n_video = size - n_audio
    print(f"   Origin: audio={n_audio:,} ({100*n_audio/size:.1f}%)  "
          f"video={n_video:,} ({100*n_video/size:.1f}%)")

    # Treat the field's presence as ground truth; a flat-zero CE field is
    # a legitimate "all-laminar" run, not missing data.
    has_tension = ('tension_ce' in data)
    if has_tension:
        print(f"\nTension / Climate:")
        print(f"   CE: {t_ce.min():.3f} - {t_ce.max():.3f}  mean={t_ce.mean():.3f}")
        print(f"   AE: {t_ae.min():.3f} - {t_ae.max():.3f}  mean={t_ae.mean():.3f}")
        lam = (t_ce < 0.05).sum()
        cre = ((t_ce >= 0.05) & (t_ce < 0.15)).sum()
        turb = (t_ce >= 0.15).sum()
        print(f"   Laminar={lam:,}({100*lam/size:.1f}%) "
              f"Creative={cre:,}({100*cre/size:.1f}%) "
              f"Turbulent={turb:,}({100*turb/size:.1f}%)")
    else:
        print(f"\n  Tension data absent (pre-v121)")

    if 'version' in data:
        v_unique, v_counts = np.unique(version.astype(int), return_counts=True)
        print(f"  Versions: " + "  ".join(f"v{int(v)}={c:,}" for v, c in zip(v_unique, v_counts)))
    else:
        print(f"  Versions: (field not recorded in this save)")

    if has_lens:
        print(f"\nLens metadata (stored at write time):")
        for name, arr in [('child', lens_child), ('youth', lens_youth),
                          ('adult', lens_adult), ('elder', lens_elder)]:
            print(f"   {name:6s}: mean={arr.mean():.4f}  std={arr.std():.4f}  "
                  f"[{arr.min():.3f}, {arr.max():.3f}]")
    else:
        print(f"\n  Lens metadata absent (pre-v130)")

    # -- Clustering ---
    coords_n = _norm(coords)
    print(f"\nClustering (7D cosine):")
    sil = _silhouette(coords_n, rng)
    best_sil = 0.0
    best_partition = 0
    if sil:
        for k, s in sil:
            mark = " <-- best" if s == max(v for _, v in sil) else ""
            print(f"   k={k:4d}  silhouette={s:.4f}{mark}")
            if s > best_sil:
                best_sil = s
                best_partition = k
    else:
        print("   (insufficient samples for silhouette sweep)")
        best_partition = 2

    # -- Temporal stability ---
    print(f"\nTemporal stability:")
    stab = _temporal_stability(coords_n, timestamps, rng)
    for w, n, si, sc, dr in stab:
        if si is None:
            print(f"   Last {w:4d}s: {n} memories -- insufficient")
        else:
            flag = "STABLE" if dr < 0.01 else "DRIFT"
            print(f"   Last {w:4d}s ({n:4d} mems): internal={si:.4f}  cross={sc:.4f}  drift={dr:.4f}  {flag}")

    # -- Single KMeans fit, labels reused for BSR / load balance / entropy ----
    labels = None
    if best_partition > 1 and size >= best_partition * 5:
        km = MiniBatchKMeans(n_clusters=best_partition,
                             random_state=int(rng.integers(2**32)),
                             n_init=5, batch_size=4096)
        labels = km.fit_predict(coords_n)

    # -- Branching & structure ---
    print(f"\nTrajectory Branching (N Bank):")
    b_count, b_rate = _detect_branching(coords_n, timestamps, rng,
                                         sim_threshold=branch_sim_thresh,
                                         div_threshold=branch_div_thresh)
    print(f"   Forks detected: {b_count} ({(b_rate*100):.1f}% of sampled nodes)")
    print(f"   Status: {'HIGH BRANCHING' if b_rate > 0.05 else 'LINEAR/CONVERGENT'}")

    bsr = 0.0
    bps = 0.0
    cv_access = 0.0
    # Reliable BSR/BPS need slightly more density than the entropy threshold;
    # keep the original guard but reuse the labels we already computed.
    if labels is not None and size // best_partition > 10:
        bsr = _branch_separation_ratio(coords_n, labels, best_partition, rng)
        bps = _branch_persistence_score(coords_n, timestamps, best_partition, rng)
        cv_access = _branch_load_balance(access, labels, best_partition)
        print(f"\nBranch Structure Analysis:")
        print(f"   Branch Separation Ratio (BSR): {bsr:.2f}  (>2.0 = strong branch)")
        print(f"   Branch Persistence Score (BPS): {bps:.3f}  (1.0 = perfect persistence)")
        print(f"   Branch load balance (CV of mean access): {cv_access:.2f}  (low = balanced)")
        sizes = [np.sum(labels == k) for k in range(best_partition)]
        print(f"   Partition sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
    else:
        print(f"\nBranch Structure Analysis: insufficient clusters for reliable analysis")

    # -- Retrieval diagnostics ---
    n_queries = min(128, size // 100)
    if n_queries >= 10:
        query_idx = rng.choice(size, n_queries, replace=False)
        queries = coords_n[query_idx]
        print(f"\nRetrieval Diagnostics (batch of {n_queries} queries):")
        topk = _topk_similarity_concentration(coords_n, queries)
        print(f"   Top-k similarity concentration (mean similarity):")
        for k, v in topk.items():
            print(f"      k={k:2d}: {v:.4f}")
        interfer = _interference_index(coords_n, query_idx)
        print(f"   Interference index (mean #neighbors, excluding self):")
        for th, cnt in interfer.items():
            print(f"      sim>{th:.2f}: {cnt:.1f}")
        if labels is not None:
            # Reuse the same KMeans labels we computed above — saves one fit.
            norm_ent, raw_ent = _retrieval_entropy(coords_n, query_idx, labels, k=50)
            print(f"   Retrieval entropy (normalized, 0=concentrated, 1=diffuse): {norm_ent:.3f} (raw={raw_ent:.2f})")
        else:
            print(f"   Retrieval entropy: insufficient clusters")
        sym = _mutual_symmetry(coords_n, query_idx, n_neighbors=10)
        print(f"   Mutual recall symmetry: {sym:.3f} (1.0 = perfect symmetry)")

    # -- Execution regime (soft labels) ---
    print(f"\nExecution topology (preliminary):")
    label, _score = _execution_topology(best_partition, b_rate, bsr)
    if label == 'BRANCHED-LIKE':
        print(f"   → BRANCHED-LIKE (high internal partitioning + divergent forks)")
    elif label == 'DELIBERATIVE-LIKE':
        print(f"   → DELIBERATIVE-LIKE (structured workspace, proto-branching)")
    else:
        print(f"   → UNITARY-LIKE (low internal partitioning)")

    # -- Full plotting (log histograms, no lens-argmax subplot) ---
    # Layout: every plot gets its own axis, no overlapping. 5 cols when lens
    # metadata is present, 4 otherwise.
    n_cols = 5 if has_lens else 4
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9))
    title = f"N Bank -- {size:,} memories ({n_dim}D)  {'v131 log' if is_v131 else 'v130/v129'}"
    fig.suptitle(title, fontsize=14)

    samp = rng.choice(size, min(5000, size), replace=False)

    # Row 0, col 0: Freq distribution (log)
    ax = axes[0,0]
    if freq_hz.max() > freq_hz.min() * 1.01:
        bins = _safe_geomspace(max(freq_hz.min(),1e-6), freq_hz.max(), 60)
        ax.hist(freq_hz, bins=bins, color='teal', alpha=0.7, edgecolor='white')
        ax.set_xscale('log')
    else:
        ax.hist(freq_hz, bins=20, color='teal', alpha=0.7, edgecolor='white')
    ax.axvline(freq_hz.mean(), color='red', lw=2, ls='--', label='mean')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Count')
    ax.set_title('Frequency Distribution (log bins)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Row 0, col 1: Delay distribution (log)
    ax = axes[0,1]
    if delay_ms.max() > delay_ms.min() * 1.01:
        bins = _safe_geomspace(max(delay_ms.min(),1e-6), delay_ms.max(), 60)
        ax.hist(delay_ms, bins=bins, color='mediumseagreen', alpha=0.7, edgecolor='white')
        ax.set_xscale('log')
    else:
        ax.hist(delay_ms, bins=20, color='mediumseagreen', alpha=0.7, edgecolor='white')
    ax.axvline(delay_ms.mean(), color='red', lw=2, ls='--', label='mean')
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Delay Distribution (log bins)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Row 0, col 2: Freq vs Delay (access coloured)
    ax = axes[0,2]
    sc = ax.scatter(freq_hz[samp], delay_ms[samp], c=np.log1p(access[samp]), alpha=0.4, s=8, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='log(access)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Delay (ms)')
    ax.set_title('Freq x Delay (N) -- colour=access')
    ax.grid(alpha=0.3)

    # Row 0, col 3: Origin over time
    ax = axes[0,3]
    t_rel = (timestamps - timestamps.min()) / 3600
    ax.scatter(t_rel[origin < 0.5], freq_hz[origin < 0.5], alpha=0.2, s=4, color='royalblue', label='audio')
    ax.scatter(t_rel[origin >= 0.5], freq_hz[origin >= 0.5], alpha=0.2, s=4, color='tomato', label='video')
    ax.set_xlabel('Hours')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Origin Over Time')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 0, col 4: Lens distribution (only when has_lens)
    if has_lens:
        ax = axes[0,4]
        lens_means = [lens_child.mean(), lens_youth.mean(), lens_adult.mean(), lens_elder.mean()]
        lens_stds  = [lens_child.std(),  lens_youth.std(),  lens_adult.std(),  lens_elder.std()]
        x_pos = np.arange(4)
        ax.bar(x_pos, lens_means, yerr=lens_stds, color=LENS_COLORS, alpha=0.8, capsize=4, edgecolor='white')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(LENS_NAMES)
        ax.set_ylabel('Lens Value')
        ax.set_title('N Bank Lens Distribution')
        ax.grid(alpha=0.3)

    # Row 1, col 0: Tension CE
    ax = axes[1,0]
    if has_tension:
        ax.hist(t_ce[origin < 0.5], bins=40, alpha=0.6, color='royalblue', label='audio', density=True)
        ax.hist(t_ce[origin >= 0.5], bins=40, alpha=0.6, color='tomato', label='video', density=True)
        ax.axvline(0.05, color='green', ls='--', lw=1, label='Laminar')
        ax.axvline(0.15, color='orange', ls='--', lw=1, label='Creative')
        ax.set_xlabel('CE Tension (rad)')
        ax.set_ylabel('Density')
        ax.set_title('Climate (CE)')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5,0.5,'No tension data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('CE Tension (N/A)')

    # Row 1, col 1: Tension AE
    ax = axes[1,1]
    if has_tension:
        ax.hist(t_ae[origin < 0.5], bins=40, alpha=0.6, color='royalblue', label='audio', density=True)
        ax.hist(t_ae[origin >= 0.5], bins=40, alpha=0.6, color='tomato', label='video', density=True)
        ax.set_xlabel('AE Tension (rad)')
        ax.set_ylabel('Density')
        ax.set_title('Climate (AE)')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5,0.5,'No tension data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('AE Tension (N/A)')

    # Row 1, col 2: Temporal age
    ax = axes[1,2]
    lt_norm = coords[:, 4]
    ax.hist(lt_norm, bins=50, color='slateblue', alpha=0.7, edgecolor='white')
    ax.set_xlabel('log_time_norm')
    ax.set_title('Temporal Age (N)')
    ax.grid(alpha=0.3)

    # Row 1, col 3: Silhouette
    ax = axes[1,3]
    if sil:
        ks, ss = zip(*sil)
        ax.plot(ks, ss, 'o-', color='teal')
        ax.axhline(0.2, color='green', ls='--', alpha=0.6, label='0.2 good')
        ax.axhline(0.1, color='orange', ls='--', alpha=0.6, label='0.1 marginal')
        ax.set_xlabel('k')
        ax.set_ylabel('Silhouette')
        ax.set_title('7D Clustering Quality')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5,0.5,'insufficient data', ha='center', transform=ax.transAxes)

    # Row 1, col 4: Freq x Delay coloured by dominant lens (only when has_lens).
    # Mirrors the original v131 layout — convenience visual using argmax of phase
    # dims; see analyze_main caveat about circular semantics.
    if has_lens:
        ax = axes[1,4]
        dominant = np.argmax(lens_phases[samp, :4], axis=1)
        c = [LENS_COLORS[d] for d in dominant]
        ax.scatter(freq_hz[samp], delay_ms[samp], c=c, alpha=0.3, s=5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Delay (ms)')
        ax.set_title('Freq x Delay (colour=dominant lens)')
        for i, name in enumerate(LENS_NAMES):
            ax.scatter([], [], c=LENS_COLORS[i], label=name, s=20)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('analysis_n.png', dpi=150)
    plt.close()
    print("\nSaved: analysis_n.png")

    return {'size': size, 'best_sil': best_sil, 'best_partition': best_partition,
            'freq_hz': freq_hz, 'delay_ms': delay_ms,
            'access': access, 'timestamps': timestamps, 'origin': origin,
            't_ce': t_ce, 't_ae': t_ae, 'has_tension': has_tension,
            'has_lens': has_lens, 'is_v131': is_v131,
            'branch_rate': b_rate, 'coords_n': coords_n,
            'bsr': bsr, 'bps': bps, 'cv_access': cv_access}


# ----------------------------------------------------------------------------
# Joint analysis
# ----------------------------------------------------------------------------

def analyze_joint(main_res, n_res, rng, branch_sim_thresh, branch_div_thresh):
    if main_res is None or n_res is None:
        return

    print(f"\n{'='*60}")
    print(f"JOINT ANALYSIS")
    print(f"{'='*60}")

    ratio = n_res['size'] / main_res['size']
    print(f"  Main bank: {main_res['size']:,}  N bank: {n_res['size']:,}  ratio: {ratio:.3f}")

    print(f"\n  Freq (Hz):")
    for label, f in [("Main", main_res['freq_hz']), ("N", n_res['freq_hz'])]:
        print(f"    {label}: mean={f.mean():.2f}  std={f.std():.2f}  min={f.min():.2f}  max={f.max():.2f}")

    print(f"\n  Delay (ms):")
    for label, d in [("Main", main_res['delay_ms']), ("N", n_res['delay_ms'])]:
        print(f"    {label}: mean={d.mean():.2f}  std={d.std():.2f}  min={d.min():.2f}  max={d.max():.2f}")

    print(f"\n  Access gravity:")
    print(f"    Main: mean={main_res['access'].mean():.1f}  max={main_res['access'].max():.0f}")
    print(f"    N:    mean={n_res['access'].mean():.1f}  max={n_res['access'].max():.0f}")

    print(f"\n  Trajectory branching rate:")
    print(f"    Main: {main_res.get('branch_rate', 0)*100:.1f}% forks")
    print(f"    N:    {n_res.get('branch_rate', 0)*100:.1f}% forks")

    corr, rates_m, rates_n = _cross_bank_branch_correlation(
        main_res['timestamps'], main_res['mems_n'],
        n_res['timestamps'], n_res['coords_n'],
        rng,
        sim_threshold=branch_sim_thresh,
        div_threshold=branch_div_thresh
    )
    print(f"\n  Cross-bank branch correlation (normalized rates, 60s bins): {corr:.3f}")
    print(f"    Mean fork rates per bin: Main={rates_m.mean():.2f}, N={rates_n.mean():.2f}")

    # -- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Joint Analysis -- Main vs N Bank (log-wrapped)", fontsize=13)

    ax = axes[0]
    t_m = (main_res['timestamps'] - main_res['timestamps'].min()) / 3600
    samp = rng.choice(len(t_m), min(8000, len(t_m)), replace=False)
    sc = ax.scatter(t_m[samp], main_res['freq_hz'][samp],
                    alpha=0.15, s=3, c=np.log1p(main_res['access'][samp]), cmap='plasma')
    plt.colorbar(sc, ax=ax, label='log(access)')
    ax.set_xlabel('Hours'); ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Main: Freq x Time')
    ax.grid(alpha=0.3)

    ax = axes[1]
    t_n = (n_res['timestamps'] - n_res['timestamps'].min()) / 3600
    samp_n = rng.choice(len(t_n), min(8000, len(t_n)), replace=False)
    colors = np.where(n_res['origin'][samp_n] < 0.5, 0.0, 1.0)
    ax.scatter(t_n[samp_n], n_res['freq_hz'][samp_n], alpha=0.2, s=3, c=colors, cmap='bwr')
    ax.set_xlabel('Hours'); ax.set_ylabel('Frequency (Hz)')
    ax.set_title('N: Freq x Time (blue=audio  red=video)')
    ax.grid(alpha=0.3)

    ax = axes[2]
    for label, f, ts, col, ls in [
        ("Main freq", main_res['freq_hz'], main_res['timestamps'], 'steelblue', '-'),
        ("N freq",    n_res['freq_hz'],    n_res['timestamps'],    'teal',      '-'),
        ("Main delay",main_res['delay_ms'],main_res['timestamps'], 'tomato',    '--'),
        ("N delay",   n_res['delay_ms'],   n_res['timestamps'],    'coral',     '--'),
    ]:
        t_hr = (ts - ts.min()) / 3600
        bin_edges = np.arange(0, t_hr.max() + 6, 6)
        means, mids = [], []
        for i in range(len(bin_edges)-1):
            m = (t_hr >= bin_edges[i]) & (t_hr < bin_edges[i+1])
            if m.sum() > 10:
                means.append(f[m].mean())
                mids.append((bin_edges[i] + bin_edges[i+1])/2)
        ax.plot(mids, means, color=col, ls=ls, lw=1.5, label=label, alpha=0.8)

    ax.set_xlabel('Hours')
    ax.set_ylabel('Mean Value')
    ax.set_title('6-hr Rolling Means')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('analysis_joint.png', dpi=150)
    plt.close()
    print("\nSaved: analysis_joint.png")


# ----------------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------------

def verdict(main_res, n_res):
    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")

    if main_res:
        s = main_res['best_sil']
        if s > 0.2:
            print(f"  Main: HIGH clustering ({s:.3f}) -- tiered index warranted")
        elif s > 0.1:
            print(f"  Main: MEDIUM clustering ({s:.3f}) -- Solar+Planet tiers sufficient")
        else:
            print(f"  Main: LOW clustering ({s:.3f}) -- stochastic sampling preferred")

        if main_res.get('has_6d'):
            if main_res.get('is_v131'):
                print(f"  Main 6D: v131 log-wrapped (unwrapped log coords)")
                print(f"    Freq std: {main_res['freq_hz'].std():.2f} Hz, Delay std: {main_res['delay_ms'].std():.2f} ms")
            else:
                print(f"  Main 6D: v130 linear toroidal (wrapped phases)")
        else:
            print(f"  Main 6D: Missing -- run v130/v131 to generate toroidal coords")

    if n_res:
        s = n_res['best_sil']
        print(f"\n  N bank 7D: silhouette={s:.3f}")
        print(f"  Freq std: {n_res['freq_hz'].std():.2f} Hz, Delay std: {n_res['delay_ms'].std():.2f} ms")

        if n_res['has_tension']:
            ce = n_res['t_ce']
            lam = 100 * (ce < 0.05).sum() / n_res['size']
            cre = 100 * ((ce >= 0.05) & (ce < 0.15)).sum() / n_res['size']
            tur = 100 * (ce >= 0.15).sum() / n_res['size']
            print(f"  Climate: {lam:.1f}% laminar  {cre:.1f}% creative  {tur:.1f}% turbulent")

        print(f"  Coords: {'v131 log-wrapped' if n_res.get('is_v131') else 'v130/v129 (linear phases)'}")
        print(f"  Lens metadata: {'present' if n_res.get('has_lens') else 'absent (pre-v130)'}")
        print(f"  POLO: Galaxy threshold (silhouette>0.2) {'MET' if s > 0.2 else 'NOT MET'}")

        # Shared classifier — same logic the N-bank section already printed.
        label, _score = _execution_topology(
            n_res.get('best_partition', 0),
            n_res.get('branch_rate', 0.0),
            n_res.get('bsr', 0.0),
        )
        if label == 'BRANCHED-LIKE':
            print(f"  Execution topology: BRANCHED-LIKE")
        elif label == 'DELIBERATIVE-LIKE':
            print(f"  Execution topology: DELIBERATIVE-LIKE (proto-branching)")
        else:
            print(f"  Execution topology: UNITARY-LIKE")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="MaGi v131 dual-bank memory analysis (log-wrapped torus)")
    ap.add_argument('--main', default=DEFAULT_MAIN_FILE, help='Main bank file')
    ap.add_argument('--n',    default=DEFAULT_N_FILE,    help='N bank file')
    ap.add_argument('--only', choices=['main', 'n', 'both'], default='both')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--branch_sim_threshold', type=float, default=0.75,
                    help='Similarity threshold for connected branch detection')
    ap.add_argument('--branch_div_threshold', type=float, default=0.4,
                    help='Divergence threshold for branch detection')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    main_res = analyze_main(args.main, rng, args.branch_sim_threshold, args.branch_div_threshold) \
               if args.only in ('main', 'both') else None
    n_res = analyze_n(args.n, rng, args.branch_sim_threshold, args.branch_div_threshold) \
            if args.only in ('n', 'both') else None

    if main_res and n_res:
        analyze_joint(main_res, n_res, rng, args.branch_sim_threshold, args.branch_div_threshold)

    verdict(main_res, n_res)
    print(f"\nDone.")


if __name__ == '__main__':
    main()