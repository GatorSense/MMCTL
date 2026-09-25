# MMCTL Resubmission — Reviewer Comments Tracker

**Last updated:** 2026-09-24  
**Manuscript:** `paper_files/bare_jrnl_new_sample4.tex`  
**Results summary:** `results/RESULTS_TRACKER.md`

Status key: **DONE** | **PARTIAL** | **OPEN** | **DEFERRED** | **N/A**

---

## Reviewer 1

| # | Request | Status | Notes |
|---|---|---|---|
| R1.1 | Difference vs SupCon / cross-modal triplet | **DONE** | Related Work + contribution bullets |
| R1.2 | Interpretable latent space / retrieval | **PARTIAL** | PCA + retrieval failure fig; optional successes fig |
| R1.3 | Pos/neg selection; margins | **DONE** | Batch-all; order-only; margin-free |
| R1.4 | Annotation norm/weighting; does weight matter? | **PARTIAL** | Eqs + fixed weights stated; **class-weight ablation figure removed** (no sensitivity plot) |
| R1.5 | Freeze rationale; freeze vs E2E | **PARTIAL** | Freeze justified as fair protocol; **no E2E experiment** |
| R1.6 | Stronger baselines | **DONE** | SupCon / CrossAttn / Transformer on AWIR+NEON; DeepSet removed |
| R1.7 | Std + significance | **PARTIAL** | AWIR Wilcoxon done; **NEON significance not reported** |

---

## Reviewer 2

| # | Request | Status | Notes |
|---|---|---|---|
| R2.1 | Substantive increment vs prior | **DONE** | Contribution bullets tied to eqs |
| R2.2 | Gradient-conflict intro | **DONE** | Rewritten |
| R2.3 | Soft/continuous / adaptive-margin | **DONE** | Related Work |
| R2.4 | Cross-dataset/region **or** missing-modality | **DONE** (via missing-mod) | AWIR missing-modality in paper; cross-region optional |
| R2.5 | Triplets / time / GPU memory | **PARTIAL** | Upper bound + M4 microbench; no training-GPU peak memory |
| R2.6 | k-fold vs “runs” caption | **DONE** | |
| R2.7 | Error analysis | **PARTIAL** | AWIR retrieval failures; NEON failure cases thin |
| R2.8 | Published refs | **PARTIAL** | Some updates; finish arXiv→venue audit |

---

## Still open (actionable)

### Must before resubmit
1. **Compile PDF** and confirm ≤8 pages (ablation removal helps)
2. **Response-to-reviewers letter** mapping each R1/R2 point → manuscript location
3. **Ref audit** (published versions where available)

### Should (reviewers may push)
4. **R1.5** Freeze vs end-to-end (even one dataset / short run)
5. **R1.4** If they want weight sensitivity without a full ablation fig: 1–2 sentences or a tiny table from existing `neon_mmctl_all_6_{0.5,0.8,1.0,1.5}.npy`
6. **R1.7** NEON paired significance (structure/downstream across 5 folds)
7. **R2.5** Real train time/peak memory if you have GPU logs

### Optional / nice
8. Retrieval **successes** figure (R1.2)
9. NEON missing-modality or cross-region (R2.4 already covered by AWIR missing-mod)
10. Stronger NEON failure discussion (R2.7)

---

## Change log
| Date | Update |
|---|---|
| 2026-09-20 | Initial tracker |
| 2026-09-24 | R1.6 done; NEON/AWIR full-set baselines; ablation study removed from manuscript |
