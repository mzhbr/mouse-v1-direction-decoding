"""Orientation tuning + population decoding for Allen Brain Observatory
drifting-gratings data. Functions work on plain numpy/pandas objects, so they
can be tested on synthetic data (run `python analysis.py` for the self-test).

Expected inputs
    dff  : (n_cells, n_frames) float32 array
    stim : DataFrame with columns start, end (frame indices), orientation
           (NaN for blank sweeps)
"""
import numpy as np
from scipy.stats import f_oneway
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------- 1. per-trial responses ----------
def trial_responses(dff, stim, baseline_frames=15):
    """Mean dF/F during each presentation minus mean dF/F just before it.

    Returns (n_trials, n_cells) float32.
    Subtracting the pre-stimulus baseline removes slow drifts in the signal.
    """
    starts = stim["start"].astype(int).values
    ends = stim["end"].astype(int).values
    out = np.empty((len(stim), dff.shape[0]), dtype=np.float32)
    for k, (s, e) in enumerate(zip(starts, ends)):
        b0 = max(s - baseline_frames, 0)
        base = dff[:, b0:s].mean(axis=1) if s > b0 else 0.0
        out[k] = dff[:, s:e].mean(axis=1) - base
    return out


# ---------- 2. orientation tuning ----------
def orientation_tuning(resp, orientations):
    """Mean response per orientation. Returns (tuning[n_ori, n_cells], oris)."""
    orientations = np.asarray(orientations, dtype=float)
    valid = ~np.isnan(orientations)
    oris = np.array(sorted(np.unique(orientations[valid])))
    tuning = np.stack([resp[valid & (orientations == o)].mean(axis=0)
                       for o in oris])
    return tuning, oris


def gosi(tuning, oris):
    """Global orientation selectivity index per cell (0 = flat, 1 = sharp).

    |sum_k R_k * exp(2i * theta_k)| / sum_k R_k, with R clipped at >= 0.
    The factor 2 makes theta and theta+180 deg equivalent (orientation, not
    direction).
    """
    r = np.clip(tuning, 0, None)
    theta = np.deg2rad(oris)[:, None]
    num = np.abs((r * np.exp(2j * theta)).sum(axis=0))
    den = r.sum(axis=0)
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def preferred_orientation(tuning, oris):
    """Orientation (deg) of the strongest mean response, per cell."""
    return oris[np.argmax(tuning, axis=0)]


def responsive_mask(resp, orientations, alpha=0.01):
    """Cells whose response differs across orientations (one-way ANOVA).

    Note: with thousands of cells you should correct for multiple tests
    (e.g. Benjamini-Hochberg); here alpha is just a quick filter.
    """
    orientations = np.asarray(orientations, dtype=float)
    valid = ~np.isnan(orientations)
    oris = sorted(np.unique(orientations[valid]))
    pvals = np.empty(resp.shape[1])
    for c in range(resp.shape[1]):
        groups = [resp[valid & (orientations == o), c] for o in oris]
        pvals[c] = f_oneway(*groups).pvalue
    return pvals < alpha, pvals


# ---------- 3. decoding ----------
def make_decoder():
    # Scaling happens INSIDE the pipeline so it is fit on training folds only.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=2000))


def decode_accuracy(resp, orientations, n_splits=5, seed=0):
    """Cross-validated accuracy of decoding orientation from population."""
    orientations = np.asarray(orientations, dtype=float)
    valid = ~np.isnan(orientations)
    X, y = resp[valid], orientations[valid].astype(int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(make_decoder(), X, y, cv=cv)
    return scores.mean(), scores.std(), 1.0 / len(np.unique(y))


def shuffle_null(resp, orientations, n_shuffles=50, n_splits=5, seed=0):
    """Accuracy distribution with orientation labels shuffled (the null)."""
    rng = np.random.default_rng(seed)
    orientations = np.asarray(orientations, dtype=float)
    valid = ~np.isnan(orientations)
    X, y = resp[valid], orientations[valid].astype(int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    null = []
    for _ in range(n_shuffles):
        y_perm = rng.permutation(y)
        null.append(cross_val_score(make_decoder(), X, y_perm, cv=cv).mean())
    return np.array(null)


def accuracy_vs_neurons(resp, orientations, sizes, n_repeats=5, seed=0):
    """Decoding accuracy using random subsets of n neurons."""
    rng = np.random.default_rng(seed)
    n_cells = resp.shape[1]
    mean_acc, std_acc = [], []
    for n in sizes:
        n = min(n, n_cells)
        accs = []
        for _ in range(n_repeats):
            idx = rng.choice(n_cells, size=n, replace=False)
            a, _, _ = decode_accuracy(resp[:, idx], orientations, seed=seed)
            accs.append(a)
        mean_acc.append(np.mean(accs))
        std_acc.append(np.std(accs))
    return np.array(mean_acc), np.array(std_acc)


# ---------- self-test on synthetic data ----------
def _make_synthetic(n_cells=60, n_trials=400, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    oris = np.arange(0, 360, 45)
    pref = rng.choice(oris, size=n_cells)             # ground truth
    frames_per_trial, base, on = 90, 15, 60
    dff = rng.normal(0, 0.05, (n_cells, n_trials * frames_per_trial))
    dff = dff.astype(np.float32)
    trial_ori = rng.choice(oris, size=n_trials)
    rows = []
    for k, o in enumerate(trial_ori):
        s = k * frames_per_trial + base
        e = s + on
        d = np.deg2rad(o - pref)                      # angular distance
        amp = np.exp(2.0 * (np.cos(2 * d) - 1))       # orientation tuning
        dff[:, s:e] += amp[:, None].astype(np.float32)
        rows.append({"start": s, "end": e, "orientation": float(o)})
    return dff, pd.DataFrame(rows), pref


if __name__ == "__main__":
    dff, stim, pref_true = _make_synthetic()
    resp = trial_responses(dff, stim)
    tuning, oris = orientation_tuning(resp, stim["orientation"].values)

    pref_est = preferred_orientation(tuning, oris)
    # preferred orientation should match ground truth (mod 180: orientation)
    match = np.mean((pref_est - pref_true) % 180 == 0)
    print(f"preferred orientation recovered: {match:.0%} (expect ~100%)")

    g = gosi(tuning, oris)
    print(f"mean gOSI: {g.mean():.2f} (expect clearly > 0.3)")

    mask, _ = responsive_mask(resp, stim["orientation"].values)
    print(f"tuned cells: {mask.mean():.0%} (expect ~100%)")

    acc, sd, chance = decode_accuracy(resp, stim["orientation"].values)
    null = shuffle_null(resp, stim["orientation"].values, n_shuffles=10)
    print(f"decoding acc: {acc:.2f} +/- {sd:.2f} | chance {chance:.3f} | "
          f"shuffled mean {null.mean():.2f} (expect ~chance)")

    flat = np.zeros_like(tuning) + 1.0
    print(f"gOSI of a flat tuning curve: {gosi(flat, oris)[0]:.3f} (expect 0)")
