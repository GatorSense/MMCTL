# MMCTL — Draft v2 (summary of changes)

**Manuscript:** `paper_files/wacv-2027-author-kit-template/mmctl_wacv.tex` (WACV template)
**Baseline for the diff:** `bare_jrnl_original_baseline.tex`
**Prepared:** 2026-10-04

This document summarizes what changed between the first submission (v1) and the
current draft (v2). All new content in the manuscript is wrapped in `\red{}` /
`\new{}` / `{\color{red}...}` so it is visible when the PDF is generated. v2
closes out the two round-2 experiments (loss-component ablation and Transformer
re-tuning) that were run on HiPerGator, and lands the round-2 writing items.

---

## 1. New experiments in v2 (round-2 items #4, #5)

### #4 — Loss-component ablation (NEON)
Ran `scripts/ablation_loss_components_neon.py` (HiPerGator `.venv_torch`, CUDA;
128-d embeddings, 20 MMCTL epochs, mean over 3 seeds). Each term of the MMCTL
objective is switched off in turn.

| Loss | Acc % | Box R² | Height R² | Align-cls (↓) |
|---|---|---|---|---|
| Full (adaptive + reg) | **67.6**±1.0 | 0.243 | 0.784 | 0.002 |
| Adaptive only | 67.2±2.1 | **0.256** | **0.786** | 0.014 |
| Reg only | 62.0±3.7 | 0.220 | 0.771 | **0.001** |

- **Adaptive triplet** carries discrimination + continuity: removing it
  (reg-only) drops accuracy ≈5.6 pts (67.6→62.0%) and weakens box/height R².
- **Distance-alignment regularizer** tightens class geometry: its presence
  sharpens the class-alignment residual ~7× (0.014 → 0.002) at negligible
  accuracy cost.
- **Full loss is the balanced choice:** top accuracy, within one std of the best
  variant on every other column. No single term wins every axis, which is the
  balanced behavior the paper relies on.
- **Manuscript:** new Table `tab:loss_ablation` + "Loss-component ablation"
  paragraph in the ablation section (red). Raw: `results/ablation_loss_components_neon.{json,md}`.

### #5 — Transformer-fusion re-tuning (NEON)
Ran `scripts/retune_transformer_neon.py` (HiPerGator `.venv_torch`, CUDA; 5-fold
LogReg probe on Transformer-fused NEON embeddings, N=592).

| Config | Acc % |
|---|---|
| default (cls, lr1e-3, 300ep, multitask) | 36.8±2.4 |
| mean-pool (lr1e-3, 300ep, multitask) | 41.5±4.0 |
| cls, lr3e-4, 600ep, multitask | 76.1±1.8 |
| mean-pool, lr3e-4, 600ep, cls-only | **80.7±1.5** |

- The original ~37–43% NEON Transformer number is an **optimization/tuning
  artifact on small data, not an architectural limit.** Lowering the learning
  rate to 3e-4, training longer, and using a mean-pooled token with a
  classification-only head lifts accuracy to **80.7%**, on par with the strongest
  fusion baselines.
- We **keep the small, shared default in the main tables** so that every fusion
  baseline uses one identical configuration (a fair controlled comparison), and
  report the tuned number as a caveat.
- **Manuscript:** red sentence appended to the "Fusion-baseline configurations"
  note. Raw: `results/transformer_retune_neon.{json,md}`.

---

## 2. Round-2 writing / figure items landed in v2 (#1, #2, #3, #6, #7, #8)

| # | Item | What changed in v2 | Where |
|---|---|---|---|
| 1 | Contribution statement in Intro | Added `\new{In summary...}` with four conceptual contributions (joint annotation vector; margin-free continuous triplet; distance alignment; multi-modal shared space), no equation numbers | Introduction |
| 2 | Related Work expansion | Red paragraphs on multi-modal fusion (cross-attention `lu2019vilbert`, Transformer `vaswani2017attention`), remote-sensing fusion, and metric learning | Related Work |
| 3 | Eq. 7 second term squared? | Verified against `mmctl/continuous_triplet_loss.py` — embedding distances squared, annotation gap not; manuscript already matches code. No change needed | Eq. `eq:mm-adaptive` |
| 6 | Per-model config lines | "Fusion-baseline configurations" note: SupCon head/temp, cross-attention heads, Transformer layers/heads/FFN, shared budget + selection | Experimental setup |
| 7 | Figure 3 ρ annotations | Each per-class zoom panel annotated with its within-class Spearman ρ | `fig:emb_zoom` |
| 8 | Table 2 vs. w_class=1.0 consistency | Red note: ablation row and downstream row are the *same* default model under different splits/fold seeds (80.1% vs. 80.3%, within one std) | Ablation paragraph |

---

## 3. Carried over from round-1 (already in v1→v2, for context)

- Embedding figures redesigned (PCA + t-SNE, color by class, size by box/height;
  fusion-only, baselines included): `fig:emb_awir`, `fig:emb_zoom`,
  `fig:neon_ablation`.
- Latent-space quality metrics table replaced the old "structure score":
  silhouette (class separation) + Spearman distance-preservation for box/
  position/height, with paired Wilcoxon significance (`tab:latent_alignment`).
- Downstream table with all five fusion methods (`tab:downstream_performance_tasks`).
- Class-weight sensitivity + frozen-vs-end-to-end comparisons (`tab:neon_weight_ablation`).
- DeepSet baselines removed; SupCon / cross-attention / Transformer added.
- Cross-modal failures figure and former Figure 3 removed; per-class Table 2
  moved to a commented supplementary block.
- Conclusion rewritten around the "balanced geometry" finding.

---

## 4. Honesty notes (kept consistent across the paper and the response)

- MMCTL is **not** top on every individual metric. Its claim is a *balanced*,
  coherent geometry (strong on both class separation and continuous attributes
  at once), which the loss-ablation and the metrics table both support.
- The Transformer re-tune number (80.7%) is reported as a tuning caveat; the
  main-table Transformer still uses the shared default for a fair comparison.
- The loss-ablation uses the NEON torch proxy (same proxy as the other NEON
  ablations), not the full Keras reconstruction pipeline — stated in the table
  caption.

---

## 5. Build / review checklist

- [ ] Generate the WACV PDF and confirm all new/changed text renders in red.
- [ ] Confirm `tab:loss_ablation` fits within the column (`\resizebox{\columnwidth}{!}`).
- [ ] Confirm `\eqref{eq:mm-adaptive}` / `\eqref{eq:mm-reg}` resolve (labels verified present).
- [ ] Re-read the two new numeric claims against `results/ablation_loss_components_neon.md`
      and `results/transformer_retune_neon.md`.
