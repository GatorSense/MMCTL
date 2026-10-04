# Round-2 Review Tracker (advisor comments)

**Manuscript:** `paper_files/wacv-2027-author-kit-template/mmctl_wacv.tex`
**Opened:** 2026-10-03

Status: **DONE** · **IN PROGRESS** · **TODO** · **NEEDS EXPERIMENT**

| # | Comment | Type | Status |
|---|---|---|---|
| 1 | End the Introduction with a concise contribution statement; move the Related-Work contribution list (line ~74) there; describe contributions conceptually, not with equations | Writing | **DONE** |
| 2 | Expand Related Work: multi-modal fusion (cross-attention, Transformer fusion), remote-sensing multi-modal fusion, metric learning for remote sensing (some already cited in experiments) | Writing | **DONE** |
| 3 | Check Eq. 7, second term — is it also squared? | Correctness | **DONE** |
| 4 | Loss-component ablation: effect of each loss component (only `L_mm_adaptive`, only `L_mm_reg`) | Experiment | **DONE** |
| 5 | Transformer NEON accuracy (43.11%) is far below RGB (76.22%); explain / check config + tuning | Experiment + writing | **DONE** |
| 6 | Add a short line per fusion model describing model-specific config/tuning (e.g., # attention heads) and how selected | Writing | **DONE** |
| 7 | Add the within-class Spearman ρ value to each panel of Figure 3 (per-class zoom) | Figure | **DONE** |
| 8 | Clarify why NEON MMCTL results differ between Table 2 and the w_class=1.0 default in Table 3 | Writing/consistency | **DONE** |

All eight round-2 items are addressed in `mmctl_wacv.tex` (all additions in red). See the per-item resolution below and `results/DRAFT_V2_CHANGES.md` / `results/reviewer_response_round2.md`.

---

## Notes / plan per item

1. **Contribution statement in Intro.** **DONE** — four conceptual contribution points added as an `\new{In summary...}` paragraph at the end of the Introduction (joint annotation vector; margin-free continuous triplet; distance-alignment; multi-modal shared space), phrased without equation numbers.
2. **Related Work expansion.** **DONE** — added red paragraphs on multi-modal fusion (cross-attention `lu2019vilbert`, Transformer `vaswani2017attention`), remote-sensing fusion, and metric learning, reusing experiment citations.
3. **Eq. 7 check.** **DONE** — verified against `mmctl/continuous_triplet_loss.py`: the embedding anchor–positive/anchor–negative distances are squared and the annotation-distance gap is not; the manuscript equation already matches the code (second term correctly squared). No change needed.
4. **Loss-component ablation.** **DONE** — ran `scripts/ablation_loss_components_neon.py` on HiPerGator (`.venv_torch`, CUDA; results in `results/ablation_loss_components_neon.{json,md}`). Full vs. adaptive-only vs. reg-only over 3 seeds: adaptive term drives accuracy (67.6→62.0% when removed) and continuous regression; reg term tightens class geometry (align-cls 0.014→0.002); full loss is the balanced choice. Added as Table (`tab:loss_ablation`) + "Loss-component ablation" paragraph in the ablation section of `mmctl_wacv.tex` (red).
5. **Transformer underperformance.** **DONE** — ran `scripts/retune_transformer_neon.py` on HiPerGator (results in `results/transformer_retune_neon.{json,md}`). The 37–43% is an optimization artifact, not architectural: lowering lr to 3e-4, training longer, and using a mean-pooled token with a classification-only head raises NEON Transformer probe accuracy to **80.7%**. Added a red sentence to the Fusion-baseline-configurations note reporting this, while keeping the small shared default in the main tables for a controlled comparison.
6. **Per-model config lines.** **DONE** — added the red "Fusion-baseline configurations" note giving SupCon (1024→1024→256 head, temp 0.07), cross-attention (1 block, 4 heads), Transformer (2 layers, 4 heads, FFN 2×1024, GELU, class-token readout), and the shared training budget/selection rationale.
7. **Figure 3 ρ annotations.** **DONE** — each per-class zoom panel (`fig:emb_zoom`) is annotated with its within-class Spearman ρ; the caption documents this (values from `results/perclass_smoothness.json`).
8. **Table consistency.** **DONE** — added a red clarifying sentence in the class-weight ablation paragraph: the w_class=1.0 ablation row and the NEON MMCTL downstream row are the *same* default model under different held-out splits/fold seeds (e.g., 80.1% vs. 80.3%, within one std). The former per-class Table 2 was already moved to the commented supplementary block.
