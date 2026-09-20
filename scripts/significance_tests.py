#!/usr/bin/env python3
"""Paired significance tests across AWIR runs (structure scores + optional taskheads)."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def paired(a, b, name_a, name_b, metric):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return {"error": "insufficient paired samples", "n": int(min(len(a), len(b)))}
    # Wilcoxon signed-rank (two-sided); fall back if all zeros
    try:
        w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError as e:
        w_stat, w_p = float("nan"), float("nan")
        w_err = str(e)
    else:
        w_err = None
    t_stat, t_p = stats.ttest_rel(a, b)
    return {
        "metric": metric,
        "a": name_a,
        "b": name_b,
        "n": int(len(a)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff_a_minus_b": float(np.mean(a - b)),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_p": float(w_p) if w_p == w_p else None,
        "wilcoxon_note": w_err,
        "paired_ttest_stat": float(t_stat),
        "paired_ttest_p": float(t_p),
    }


def main():
    raw = pd.read_csv(os.path.join(REPO, "results/demo_metrics_raw.csv"))
    # structure scores already computed in demo_metrics as score_*
    # Reconstruct scores if needed
    if "score_cls" not in raw.columns:
        raise SystemExit("demo_metrics_raw.csv missing score_* columns; re-run demo_metrics.py")

    methods = ["MMCTL", "Concat", "DeepSetAttn", "DeepSetMean", "RGB", "Thermal"]
    # Best baseline per metric = lowest mean score among non-MMCTL
    comparisons = []
    for src, col in [("cls", "score_cls"), ("box", "score_box"), ("pos", "score_pos")]:
        means = {
            m: raw.loc[raw["method"] == m, col].mean()
            for m in methods
            if m != "MMCTL"
        }
        best = min(means, key=means.get)
        a = raw.loc[raw["method"] == "MMCTL"].sort_values("run")[col].values
        b = raw.loc[raw["method"] == best].sort_values("run")[col].values
        comparisons.append(paired(a, b, "MMCTL", best, f"structure_score_{src}"))

    # Optional downstream proxy
    th_path = os.path.join(REPO, "results/taskhead_proxy_awir.json")
    downstream = []
    if os.path.exists(th_path):
        with open(th_path) as fp:
            th = json.load(fp)
        per = pd.DataFrame(th["per_run"])
        per = per[per["setup"] == "full"]
        for metric, higher_better in [("acc", True), ("r2_box_mean", True), ("r2_pos_mean", True)]:
            means = {
                m: per.loc[per["method"] == m, metric].mean()
                for m in per["method"].unique()
                if m != "MMCTL"
            }
            best = max(means, key=means.get) if higher_better else min(means, key=means.get)
            a = per.loc[per["method"] == "MMCTL"].sort_values("run")[metric].values
            b = per.loc[per["method"] == best].sort_values("run")[metric].values
            downstream.append(paired(a, b, "MMCTL", best, f"downstream_proxy_{metric}"))

    out = {
        "design": (
            "Paired tests across 8 AWIR runs. Structure scores from demo_metrics "
            "(lower better). Downstream uses proxy taskhead CV if available (higher better)."
        ),
        "structure_vs_best_baseline": comparisons,
        "downstream_proxy_vs_best_baseline": downstream,
    }
    out_path = os.path.join(REPO, "results/significance.json")
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
