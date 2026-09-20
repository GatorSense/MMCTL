# MMCTL Resubmission — Reviewer Comments Tracker

**Last updated:** 2026-09-20  
**Manuscript:** `paper_files/bare_jrnl_new_sample4.tex`  
**Results summary:** `results/RESULTS_TRACKER.md`

Status key: **DONE** | **PARTIAL** | **OPEN** | **DEFERRED** | **N/A**

---

## Reviewer 1 (Attached standard file — reviewer.pdf)

| # | Request | Status | Where addressed / what’s left |
|---|---|---|---|
| R1.1 | Clarify essential difference vs supervised contrastive / cross-modal triplet; continuous similarity contribution | **DONE** | Related Work + contribution bullets (i)–(iv); intro inductive-bias framing |
| R1.2 | Interpretable latent space: visualizations or cross-modal retrieval | **PARTIAL** | Existing PCA fig + new failure/retrieval fig + class R1 80.8% / identity 7.5%. Successes grid not placed in PDF (space). Optional: add successes fig if pages allow |
| R1.3 | Specify pos/neg selection rules; similarity thresholds / margins | **DONE** | Method: batch-all; order-only criterion; no hard threshold; margin-free adaptive gap |
| R1.4 | How annotations normalized/weighted; does weighting affect results? | **DONE** | Eqs. for norm/weight; class-weight ablation Fig. already in paper |
| R1.5 | Rationale for freezing encoder; compare end-to-end fine-tuning | **PARTIAL** | Freeze framed as fair fusion protocol. **OPEN:** freeze vs E2E experiment (skipped; no checkpoints / optional) |
| R1.6 | Stronger baselines: SupCon, cross-attention, Transformer fusion | **OPEN** | Not run. Need at least one stronger baseline for a complete response |
| R1.7 | Report std + statistical significance | **DONE** (AWIR) | Wilcoxon n=8, structure p=0.0078; proxy downstream p≤0.016. **OPEN:** NEON significance once NEON replicated |
| — | Overall “method too simple / limited contribution” | **PARTIAL** | Related Work + incremental bullets. Still need NEON repro + ideally R1.5/R1.6 for strength |

---

## Reviewer 2 (Comments to the Author)

| # | Request | Status | Where addressed / what’s left |
|---|---|---|---|
| R2.1 | State substantive increment vs prior work (Eqs 1–3, continuous triplet, reg, multimodal extension) | **DONE** | Related Work + explicit contribution bullets tied to eqs |
| R2.2 | Intro uses gradient conflict without experiments — revise motivation | **DONE** | Intro rewritten; gradient-conflict claim removed |
| R2.3 | Discuss vs soft/continuous similarity ML and adaptive-margin triplets | **DONE** | Related Work (Kim 2019, Ha adaptive-margin, etc.) |
| R2.4 | Add cross-dataset/cross-region transfer **or** missing-modality | **PARTIAL** | AWIR missing-modality probes done. **OPEN:** cross-region / NEON transfer; NEON missing-modality |
| R2.5 | Report #triplets, time/epoch, GPU memory (batch-all) | **PARTIAL** | Upper-bound triplets + M4 microbench. **OPEN:** full schedule wall-clock / true peak GPU memory on training GPU |
| R2.6 | Clarify k-fold vs “averaged over 5 runs” in Fig. 3 caption | **DONE** | Caption aligned with 5 NEON CV folds |
| R2.7 | Error analysis: when method works / fails | **PARTIAL** | AWIR retrieval failure analysis. **OPEN:** richer NEON failure cases; more typical downstream failure discussion |
| R2.8 | Update refs to formally published versions (e.g. [8]) | **PARTIAL** | `zhou2025multi` → IGARSS 2025; added Khosla/Ha. **OPEN:** audit remaining arXiv-only cites (`reiter2020deep` still arXiv) |

---

## Replication / experimental gates

| Item | Status | Notes |
|---|---|---|
| AWIR structure table | **DONE — PASS** | Keep paper numbers; max drift 0.13 |
| AWIR downstream table | **PARTIAL** | Keep paper absolutes; proxy cls OK; DeepSet proxy FAIL |
| AWIR missing modality | **DONE** | In manuscript + `missing_modality.json` |
| AWIR significance | **DONE** | In manuscript + `significance.json` |
| AWIR compute cost | **PARTIAL** | Upper bound + microbench in manuscript |
| AWIR failure/retrieval | **DONE** | In manuscript + figures |
| **NEON structure table** | **OPEN / IN PROGRESS** | No pretrained NEON embeddings in repo; proxy training required |
| **NEON downstream table** | **OPEN / IN PROGRESS** | Same |
| NEON significance | **OPEN** | Depends on NEON repro |
| Stronger baseline (R1.6) | **OPEN** | Recommended before resubmit |
| Freeze vs E2E (R1.5) | **OPEN** | Recommended if compute allows |
| Cross-region transfer | **OPEN** | Alternative/addition to missing-mod for R2.4 |
| PDF ≤8 pages verified | **OPEN** | Estimate ~7.5–8; need local `pdflatex` compile |

---

## Manuscript text edits (writing)

| Edit | Status |
|---|---|
| Intro rewrite | **DONE** |
| Related Work + contributions | **DONE** |
| Method clarifications (triplets/margin/weights/freeze) | **DONE** |
| Same-labels / different-bias claim | **DONE** |
| Reproducibility paragraph | **DONE** |
| Significance paragraph | **DONE** |
| Missing-modality paragraph | **DONE** |
| Compute-cost paragraph | **DONE** |
| Failure/retrieval subsection + fig | **DONE** |
| Fig. ablation caption fix | **DONE** |
| LaTeX `\err{0.88}` fix | **DONE** |
| Bibliography updates | **PARTIAL** |
| Response-to-reviewers letter | **OPEN** | Draft after remaining experiments |
| Final page-budget trim | **OPEN** | After NEON + optional experiments |

---

## Priority checklist — what you still need to do

### Must (before confident resubmit)
1. **Reproduce / proxy-reproduce NEON** structure + downstream; document PASS/FAIL vs paper tables → update `RESULTS_TRACKER.md`
2. **Compile PDF** and confirm ≤8 pages; trim if over
3. **Write response-to-reviewers** mapping each R1/R2 point to manuscript locations
4. Finish **ref audit** (published versions where available)

### Should (strongly recommended for R1)
5. Add **one stronger baseline** (SupCon *or* cross-attention) on AWIR at minimum
6. Add **freeze vs end-to-end** ablation (even one dataset)
7. Improve **compute reporting** if you have access to the original training GPU logs (time/epoch, peak memory)

### Nice / optional
8. Place retrieval **successes** figure if space
9. Cross-site NEON transfer experiment
10. Deeper downstream failure case study (confused species / small crowns)

---

## Change log of this tracker

| Date | Update |
|---|---|
| 2026-09-20 | Initial tracker from R1+R2 + Agent A/B outcomes; NEON marked open |
