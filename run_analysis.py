"""Weeks 2-3: orientation tuning + population decoding on one Allen experiment.

Run from the project folder (the one that contains boc/ and analysis.py):
    C:\\v\\Scripts\\python.exe run_analysis.py

Needs analysis.py in the same folder. The experiment must already be
downloaded (week1_explore.py did that).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import analysis as A

EXP_ID = 645413759          # the experiment you used in week 1
N_SHUFFLES = 200             # raise to 100+ for the final README numbers
NEURON_SIZES = [5, 10, 20, 50, 100, 226]
OUT = Path("figs")
OUT.mkdir(exist_ok=True)


def analyze(dff, stim, cell_ids, out=OUT, n_shuffles=N_SHUFFLES,
            sizes=NEURON_SIZES):
    ori = stim["orientation"].values.astype(float)
    results = {"n_cells": int(dff.shape[0]),
               "n_trials": int(np.sum(~np.isnan(ori)))}

    # ---------- tuning ----------
    resp = A.trial_responses(dff, stim)
    tuning, oris = A.orientation_tuning(resp, ori)
    g = A.gosi(tuning, oris)
    pref = A.preferred_orientation(tuning, oris)
    resp_mask, pvals = A.responsive_mask(resp, ori, alpha=0.01)

    results["frac_responsive_p<0.01"] = float(resp_mask.mean())
    results["median_gOSI_all"] = float(np.median(g))
    results["median_gOSI_responsive"] = (
        float(np.median(g[resp_mask])) if resp_mask.any() else None)
    print(f"cells: {results['n_cells']} | responsive (ANOVA p<0.01): "
          f"{resp_mask.mean():.0%}")
    print(f"median gOSI: all {results['median_gOSI_all']:.2f}, "
          f"responsive {results['median_gOSI_responsive']}")

    # Figure 1: tuning summary
    fig1, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].hist(g, bins=20, alpha=0.6, label="all cells")
    if resp_mask.any():
        axes[0].hist(g[resp_mask], bins=20, alpha=0.6, label="responsive")
    axes[0].set_xlabel("gOSI")
    axes[0].set_ylabel("cells")
    axes[0].legend()
    axes[1].hist(pref[resp_mask] if resp_mask.any() else pref,
                 bins=np.append(oris, oris[-1] + 45) - 22.5)
    axes[1].set_xlabel("preferred direction (deg)")
    axes[1].set_ylabel("responsive cells")
    fig1.tight_layout()
    fig1.savefig(Path(out) / "tuning_summary.png", dpi=150)

    # Figure 2: tuning curves of the most selective responsive cells
    cand = np.where(resp_mask)[0]
    top = cand[np.argsort(g[cand])[::-1][:6]] if len(cand) else []
    valid = ~np.isnan(ori)
    fig2, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
    for ax, c in zip(axes.ravel(), top):
        means, sems = [], []
        for o in oris:
            r = resp[valid & (ori == o), c]
            means.append(r.mean())
            sems.append(r.std(ddof=1) / np.sqrt(len(r)))
        ax.errorbar(oris, means, yerr=sems, fmt="o-", capsize=2)
        ax.set_title(f"cell {cell_ids[c]} (gOSI {g[c]:.2f})", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("direction (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean dF/F")
    fig2.tight_layout()
    fig2.savefig(Path(out) / "top_tuning_curves.png", dpi=150)

    # ---------- decoding ----------
    acc8, sd8, chance8 = A.decode_accuracy(resp, ori)
    null8 = A.shuffle_null(resp, ori, n_shuffles=n_shuffles)
    p8 = (1 + np.sum(null8 >= acc8)) / (1 + len(null8))

    ori180 = np.mod(ori, 180)                      # orientation, not direction
    acc4, sd4, chance4 = A.decode_accuracy(resp, ori180)

    results.update({
        "decode_8dir_acc": float(acc8), "decode_8dir_sd": float(sd8),
        "chance_8dir": float(chance8),
        "shuffle_mean_8dir": float(null8.mean()),
        "shuffle_p_8dir": float(p8), "n_shuffles": int(n_shuffles),
        "decode_4ori_acc": float(acc4), "decode_4ori_sd": float(sd4),
        "chance_4ori": float(chance4),
    })
    print(f"decode 8 directions: {acc8:.2f} +/- {sd8:.2f} "
          f"(chance {chance8:.3f}, shuffled {null8.mean():.2f}, p~{p8:.3f})")
    print(f"decode 4 orientations: {acc4:.2f} +/- {sd4:.2f} "
          f"(chance {chance4:.2f})")

    sizes = sorted(set(min(s, dff.shape[0]) for s in sizes))
    m, s = A.accuracy_vs_neurons(resp, ori, sizes)
    results["accuracy_vs_neurons"] = {int(n): float(a)
                                      for n, a in zip(sizes, m)}

    fig3, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].bar(["8 directions", "4 orientations"], [acc8, acc4],
                yerr=[sd8, sd4], capsize=4, color="tab:blue", alpha=0.8)
    axes[0].hlines([chance8], -0.4, 0.4, colors="k", linestyles="--")
    axes[0].hlines([chance4], 0.6, 1.4, colors="k", linestyles="--")
    axes[0].set_ylabel("cross-validated accuracy (dashed = chance)")
    axes[0].set_ylim(0, 1)
    axes[1].errorbar(sizes, m, yerr=s, fmt="o-", capsize=3)
    axes[1].axhline(chance8, color="k", linestyle="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("number of neurons")
    axes[1].set_ylabel("accuracy (8 directions)")
    axes[1].set_ylim(0, 1)
    fig3.tight_layout()
    fig3.savefig(Path(out) / "decoding.png", dpi=150)

    with open(Path(out) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    import allensdk.brain_observatory.stimulus_info as stim_info

    boc = BrainObservatoryCache(manifest_file="boc/manifest.json")
    data_set = boc.get_ophys_experiment_data(EXP_ID)
    _, dff = data_set.get_dff_traces()
    dff = dff.astype(np.float32)
    cell_ids = data_set.get_cell_specimen_ids()
    stim = data_set.get_stimulus_table(stim_info.DRIFTING_GRATINGS)
    analyze(dff, stim, cell_ids)
    plt.show()


if __name__ == "__main__":
    main()
