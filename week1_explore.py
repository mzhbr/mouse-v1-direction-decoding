# %% [markdown]
# # Week 1: Explore one Allen Brain Observatory V1 experiment (drifting gratings)
#
# Goal: download ONE experiment, understand its structure, make first plots.
# Memory-friendly: one experiment at a time, float32 arrays.
# Works as a plain script or cell-by-cell (VS Code / Jupyter / Colab, cells split by "# %%").

# %% 0. Setup
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info

OUT = Path("figs")
OUT.mkdir(exist_ok=True)

# Data is cached under ./boc, so each experiment downloads only once.
boc = BrainObservatoryCache(manifest_file="boc/manifest.json")

# %% 1. List candidate experiments (V1 = "VISp", has drifting gratings)
# Cux2-CreERT2 labels layer 2/3 excitatory neurons: usually many cells per session.
exps = boc.get_ophys_experiments(
    targeted_structures=["VISp"],
    stimuli=[stim_info.DRIFTING_GRATINGS],
    cre_lines=["Cux2-CreERT2"],
)
exp_table = pd.DataFrame(exps)
print(f"{len(exp_table)} candidate experiments")
cols = [c for c in ["id", "experiment_container_id", "imaging_depth",
                    "cre_line", "session_type"] if c in exp_table.columns]
print(exp_table[cols].head(10))

# %% 2. Choose ONE experiment and download it (a few hundred MB, one time only)
EXP_INDEX = 0  # change this to try another experiment
exp_id = int(exp_table.iloc[EXP_INDEX]["id"])
print("Using experiment id:", exp_id)

data_set = boc.get_ophys_experiment_data(exp_id)
print(data_set.get_metadata())

# %% 3. Load the data we need
# dF/F traces: shape (n_cells, n_frames). float32 halves memory vs float64.
timestamps, dff = data_set.get_dff_traces()
dff = dff.astype(np.float32)
timestamps = timestamps.astype(np.float32)
cell_ids = data_set.get_cell_specimen_ids()

print("dff shape (cells x frames):", dff.shape)
print("approx memory (MB):", round(dff.nbytes / 1e6, 1))
print("approx frame rate (Hz):", round(1 / np.median(np.diff(timestamps)), 1))

# %% 4. The stimulus table: one row per grating presentation (trial)
# start / end are frame INDICES into the dff columns.
stim = data_set.get_stimulus_table(stim_info.DRIFTING_GRATINGS)
print(stim.head())
print("\nColumns:", list(stim.columns))
print("Total presentations:", len(stim))

is_blank = stim["orientation"].isna()
print("Blank sweeps:", int(is_blank.sum()))
print("Orientations (deg):", sorted(stim["orientation"].dropna().unique()))
print("Temporal frequencies (Hz):",
      sorted(stim["temporal_frequency"].dropna().unique()))
print("\nPresentations per orientation:")
print(stim["orientation"].value_counts().sort_index())

# %% 5. Plot A: a few example cells, with grating epochs shaded
rng = np.random.default_rng(0)
example = rng.choice(dff.shape[0], size=5, replace=False)

t_start_frame = int(stim["start"].min())
window_frames = int(60 * 30)  # ~60 s at ~30 Hz
f0, f1 = t_start_frame, min(t_start_frame + window_frames, dff.shape[1])

fig, axes = plt.subplots(len(example), 1, figsize=(10, 7), sharex=True)
for ax, i in zip(axes, example):
    ax.plot(timestamps[f0:f1], dff[i, f0:f1], lw=0.8, color="k")
    in_win = stim[(stim["start"] >= f0) & (stim["end"] <= f1)]
    for s, e in zip(in_win["start"].astype(int), in_win["end"].astype(int)):
        ax.axvspan(timestamps[s], timestamps[e], color="tab:blue", alpha=0.12)
    ax.set_ylabel(f"cell {cell_ids[i]}\n(dF/F)", fontsize=8)
axes[-1].set_xlabel("time (s)")
fig.suptitle("Example V1 neurons (shaded = grating on)")
fig.tight_layout()
fig.savefig(OUT / "week1_example_traces.png", dpi=150)
plt.show()


# %% 6. Plot B: quick preview of orientation tuning (full analysis is week 2)
def trial_responses(dff, stim):
    """Mean dF/F per cell during each presentation -> (n_trials, n_cells)."""
    out = np.empty((len(stim), dff.shape[0]), dtype=np.float32)
    for k, (s, e) in enumerate(zip(stim["start"].astype(int),
                                   stim["end"].astype(int))):
        out[k] = dff[:, s:e].mean(axis=1)
    return out


resp = trial_responses(dff, stim)                      # (n_trials, n_cells)
valid = ~is_blank.values
oris = np.array(sorted(stim["orientation"].dropna().unique()))

# Mean response per orientation (averaged over trials and temporal frequencies)
tuning = np.stack(
    [resp[valid & (stim["orientation"].values == o)].mean(axis=0) for o in oris]
)  # (n_orientations, n_cells)

fig, axes = plt.subplots(1, len(example), figsize=(12, 2.8), sharey=False)
for ax, i in zip(axes, example):
    ax.plot(oris, tuning[:, i], "o-", color="tab:red")
    ax.set_title(f"cell {cell_ids[i]}", fontsize=8)
    ax.set_xlabel("orientation (deg)")
axes[0].set_ylabel("mean dF/F")
fig.suptitle("Quick tuning preview (no baseline subtraction yet)")
fig.tight_layout()
fig.savefig(OUT / "week1_tuning_preview.png", dpi=150)
plt.show()

# %% 7. Week 1 checklist (write answers in your README notes)
# - How many cells? How many frames? Frame rate?
# - How many presentations per orientation / temporal frequency?
# - Do some cells already look orientation-selective in Plot B? Which ones don't?
# - Any cells with weird/noisy traces? (note them, we handle quality in week 2)
