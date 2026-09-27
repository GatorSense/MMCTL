# Reviewer Response Map

**Manuscript:** `paper_files/wacv-2027-author-kit-template/mmctl_wacv.tex` (WACV port)
**Reviews:** `paper_files/reviewer.pdf` (single reviewer, 7 comments)
**Last updated:** 2026-09-26

This maps each reviewer comment to the manuscript section(s) that address it. All
newly added revision content renders in **red** in the compiled PDF (`\new` /
`\red` / `{\color{red} …}`), so each item below is visible on compile.

Status key: **DONE** · **PARTIAL** · **OPEN**

| # | Reviewer comment (paraphrased) | Addressed in | Status |
|---|---|---|---|
| 1 | Clarify essential difference vs. supervised contrastive / cross-modal triplet loss; the added contribution of continuous similarity | §Related Work (rewritten) + contribution list; §Methodology (Problem Setup Eqs, Shared Latent Space) | **DONE** |
| 2 | Interpretable latent space — provide visualizations or cross-modal retrieval examples | Fig. 1 (AWIR embedding space), Fig. 3 (per-class zoom) + §Experimental Results "Within-class continuous transitions" | **DONE** |
| 3 | Rules for selecting positive/negative samples in continuous triplets; thresholds/margins | §Shared Latent Space with Modality-Specific Encoders — batch-all construction, Eq. (triplet-order), explicit margin-free / no-threshold statement | **DONE** |
| 4 | How are category/size/height normalized and weighted? Does weighting affect results? | §Problem Setup Eqs (ann-norm–ann-weight); §Experimental Results "Class-weight sensitivity (NEON)" + Table 3 + Fig. 4 | **DONE** |
| 5 | Rationale for freezing the encoder; compare with end-to-end fine-tuning | §Modality-Specific Encoder Architecture (freeze = fair-fusion protocol); §"Frozen vs. end-to-end encoders (NEON)" | **DONE** |
| 6 | Limited baselines — add SupCon, cross-attention, Transformer-based fusion | §Experimental Setup (baseline paragraph, with citations); Table 1 + Table 2 (downstream) | **DONE** |
| 7 | Report standard deviations and statistical significance | All tables report mean ± std; §Experimental Results "Statistical significance" (paired Wilcoxon signed-rank) | **DONE** |

---

## Detail per comment

### R1 — Difference from SupCon / cross-modal triplet; continuous-similarity contribution
- **§Related Work** (red) contrasts binary SupCon / fixed-margin cross-modal triplets and adaptive-margin work with MMCTL, and states the four-point contribution: joint annotation vector, margin-free continuous triplet with adaptive gap, pairwise distance alignment, and a multi-modal shared space where same-object cross-modal pairs have zero annotation distance.
- **§Methodology** formalizes each contribution (Eqs. `ann-norm`–`ann-concat`, `mm-adaptive`, `mm-reg`).

### R2 — Interpretable latent space (visualizations / retrieval)
- **Fig. 1**: AWIR embedding space across fusion methods (PCA + t-SNE), colored by class, sized by box area.
- **Fig. 3**: per-class zoom (horse, cow, PSME, PIPA2) colored by the continuous label, showing smooth within-class transitions.
- **"Within-class continuous transitions"** paragraph discusses these.
- *Note:* the reviewer asked for "visualizations **or** cross-modal retrieval examples." We address this with visualizations; the earlier cross-modal retrieval-failure figure was removed at the author's request. The detailed per-class metric table is retained (commented) in the manuscript's SUPPLEMENTARY block if a quantitative version is wanted back.

### R3 — Positive/negative selection; thresholds/margins
- **§Shared Latent Space** describes the **batch-all** strategy, defines valid (positive, negative) pairs purely by annotation-distance ordering (Eq. `triplet-order`), and states explicitly there is **no** hard similarity threshold and **no** tuned margin (margin-free continuous triplet).

### R4 — Normalization/weighting of annotations; effect of weighting
- **§Problem Setup**: feature-wise min–max normalization, length rescaling, and per-task weights (Eqs. `ann-norm`–`ann-weight`); fixed weights stated (AWIR w_class=0.5, NEON w_class=1.0).
- **"Class-weight sensitivity (NEON)"** + **Table 3** + **Fig. 4**: sweep w_class ∈ {0.5, 0.8, 1.0, 1.5}; accuracy rises with w_class while box/height regression stays stable — the weighting behaves as intended.

### R5 — Freeze rationale + end-to-end comparison
- **§Modality-Specific Encoder Architecture**: freezing justified as a fair fusion-evaluation protocol.
- **"Frozen vs. end-to-end encoders (NEON)"**: controlled A/B from the same pretrained encoders — freezing matches/edges end-to-end on classification (81.99% vs. 80.66%) with only marginal E2E gains on regression, justifying the frozen protocol.

### R6 — Stronger baselines
- **§Experimental Setup** introduces SupCon `\cite{khosla2020supervised}`, cross-attention `\cite{lu2019vilbert}`, and Transformer fusion `\cite{vaswani2017attention}` (DeepSet removed).
- Reported in **Table 1** (embedding-space quality) and **Table 2** (downstream).

### R7 — Standard deviations + significance
- Every results table reports mean ± std over CV folds.
- **"Statistical significance"**: paired one-sided Wilcoxon signed-rank tests `\cite{wilcoxon1945individual}` (MMCTL vs. best competitor) — significant on AWIR position (p=0.0078), positive trend on box (p=0.098); the paper is explicit that MMCTL's value is balance, not winning every single metric.

---

## Overall reviewer concern ("method too simple / contributions limited")
Addressed by: the sharpened four-point contribution and Related-Work distinction (R1), the expanded stronger-baseline comparison (R6), the standardized statistical validation (R7), the annotation-weight and freeze-vs-E2E analyses (R4, R5), and the interpretability visualizations (R2).

## Table / figure numbering (current WACV build)
- **Table 1** — Embedding-space quality (silhouette + Spearman ρ). *(main)*
- **Table 2** — Downstream task performance. *(main)*
- **Table 3** — NEON class-weight ablation. *(main)*
- **Fig. 1** — AWIR embedding space; **Fig. 2** — framework; **Fig. 3** — per-class zoom; **Fig. 4** — class-weight ablation.
- Removed from main paper: NEON overall embedding-space figure (former Fig. 3) and the per-class within-class metric table (former Table 2, kept commented in SUPPLEMENTARY).
