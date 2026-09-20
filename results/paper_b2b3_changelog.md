# Phase B2/B3 Changelog (Paper / Agent B)

Edits use **only** numbers from Agent A JSON/MD artifacts. AWIR latent-structure and absolute downstream table cells were **not** overwritten.

## Files changed
- `paper_files/bare_jrnl_new_sample4.tex`
- `paper_files/figures/awir_retrieval_failures.png` (copied from `results/figures/`)
- `paper_files/figures/awir_retrieval_successes.png` (copied; not included in PDF to save space)
- `results/paper_b2b3_changelog.md` (this file)

## B2 — Replication / reproducibility
| Edit | Reviewer |
|---|---|
| Added **Reproducibility** paragraph under Experimental Setup: structure via `demo_metrics.py` + released embeddings; ≤0.13 abs mean drift (MMCTL B); DeepSet/full train-protocol absolutes need Keras checkpoints — do not proxy-replace table cells | Replication gate / honesty |
| Kept `tab:latent_alignment` and `tab:downstream_performance_tasks` numbers unchanged (structure PASS; DeepSet proxy FAIL for absolutes) | Agent A instructions |

## B3 — New results insertions
| Edit | Reviewer |
|---|---|
| Wilcoxon paired tests (n=8 AWIR): structure MMCTL vs best baseline, all $p{=}0.0078$; proxy downstream vs Concat $p{\le}0.016$, framed as ranking support only | R1#7 |
| Missing-modality probe paragraph: full $93.44\pm1.37$; drop thermal $84.69\pm2.04$ (−8.75); drop RGB $86.15\pm1.99$ (−7.29); protocol = zero one 1024-d half of 2048-d joint test embedding | R2#4 |
| Triplet compute paragraph: ≤249,984 ordered triplets/batch @ B=32; ~2.0M/epoch upper bound; mask caveat; M4 microbench ~1.7 s/step @ B=8; full schedule not re-timed | R2#5 |
| Cross-modal retrieval + failure figure: class R1 80.8%, identity R1 7.5%; adjacent livestock confusions; caption warns against identity overclaim | R2#7, R1#2 |
| Trimmed Related Work + Ablation prose to offset new content (≤8 page budget) | Page limit |
| Updated Conclusion with significance / missing-mod / class-vs-identity takeaways | — |
| Removed B1 `% TODO(B2/B3)` comments | Cleanup |

## Numbers source map
- Structure drift 0.13 → `replication_awir_delta.md`
- Wilcoxon p=0.0078125 / proxy p≤0.015625 → `significance.json`
- Missing modality → `missing_modality.json`
- Compute → `compute_cost.json`
- Retrieval / failures → `failure_analysis.md`

## Explicitly not claimed
- No NEON code replication (`neon_status.md`: deferred)
- No stronger-baseline / freeze-vs-E2E numbers (skipped by Agent A)
- No DeepSet downstream overwrite from proxy (~45% collapse)

## Page impact (estimate)
- Prior B1 estimate ~7–7.5 pp; added ~0.6–0.9 pp of analyses + 1 figure; cut ~0.3–0.5 pp via Related Work/Ablation compression.
- Expected recompiled length: **~7.5–8.0 pages** (verify with `pdflatex`; trim figure width or drop failure grid if over).

## Leftover reviewer gaps (honest)
- R1#6 stronger multimodal baselines (SupCon / Transformer fusion): **not added** (Agent A skipped)
- R1#5 end-to-end vs freeze comparison: **not added** (skipped)
- R1#2 richer retrieval viz beyond failure grid / existing PCA: successes PNG copied but not placed
- NEON structure/downstream still unreplicated in code
- Full AWIR wall-clock retrain cost not measured (upper-bound + microbench only)
