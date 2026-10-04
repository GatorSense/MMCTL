# Round-2 Review Tracker (advisor comments)

**Manuscript:** `paper_files/wacv-2027-author-kit-template/mmctl_wacv.tex`
**Opened:** 2026-10-03

Status: **DONE** · **IN PROGRESS** · **TODO** · **NEEDS EXPERIMENT**

| # | Comment | Type | Status |
|---|---|---|---|
| 1 | End the Introduction with a concise contribution statement; move the Related-Work contribution list (line ~74) there; describe contributions conceptually, not with equations | Writing | TODO |
| 2 | Expand Related Work: multi-modal fusion (cross-attention, Transformer fusion), remote-sensing multi-modal fusion, metric learning for remote sensing (some already cited in experiments) | Writing | TODO |
| 3 | Check Eq. 7, second term — is it also squared? | Correctness | TODO |
| 4 | Loss-component ablation: effect of each loss component (only `L_mm_adaptive`, only `L_mm_reg`) | Experiment | NEEDS EXPERIMENT |
| 5 | Transformer NEON accuracy (43.11%) is far below RGB (76.22%); explain / check config + tuning | Experiment + writing | NEEDS EXPERIMENT |
| 6 | Add a short line per fusion model describing model-specific config/tuning (e.g., # attention heads) and how selected | Writing | TODO |
| 7 | Add the within-class Spearman ρ value to each panel of Figure 3 (per-class zoom) | Figure | TODO |
| 8 | Clarify why NEON MMCTL results differ between Table 2 and the w_class=1.0 default in Table 3 | Writing/consistency | TODO |

---

## Notes / plan per item

1. **Contribution statement in Intro.** Move the four contribution points from Related Work to the end of the Introduction, phrased conceptually (joint annotation vector; margin-free continuous/ordinal triplet; distance-alignment; multi-modal shared space) rather than by equation number.
2. **Related Work expansion.** Add short paragraphs on (a) attention/Transformer multimodal fusion, (b) remote-sensing multimodal fusion, (c) metric learning in remote sensing, reusing citations already present in the experiments where possible.
3. **Eq. 7 check.** Verify against `mmctl/continuous_triplet_loss.py`: the embedding anchor–positive and anchor–negative distances are **squared**; the annotation-distance gap is **not** squared. Fix the equation if the second term's squaring is inconsistent.
4. **Loss-component ablation.** Train MMCTL with only the adaptive triplet term and only the distance-alignment regularizer, compare structure + downstream vs. the full loss. Plan: extend the NEON torch proxy (`scripts/reproduce_neon_ae_mmctl.py` style) with loss-term switches; report a small table.
5. **Transformer underperformance.** Inspect the Transformer-fusion config/training in `scripts/eval_neon_released_embeddings.py` (heads/layers/epochs/lr); re-tune (more epochs / lr / warmup) and re-run NEON; update the number or add a caveat on small-data instability.
6. **Per-model config lines.** Add implementation details: SupCon (projection dim), cross-attention (# heads), Transformer (# heads, # layers, FFN dim), plus training budget and selection.
7. **Figure 3 ρ annotations.** Annotate each panel of `embedding_zoom_perclass.pdf` with its within-class Spearman ρ (from `results/perclass_smoothness.json`).
8. **Table consistency.** Reconcile NEON MMCTL numbers between the downstream table and the ablation w_class=1.0 row (same embedding, different eval split/seed); add a clarifying note. (Note: the former per-class Table 2 was moved to the supplementary; confirm which tables the comment refers to.)
