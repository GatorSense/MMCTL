# Phase B1 Changelog (Paper / Agent B)

Text-only manuscript edits for resubmission. **No numeric table values invented or overwritten.** Agent A replication / new-experiment gates not present at B1 completion (`results/` contains only `HANDOFF.md`).

## Files changed
- `paper_files/bare_jrnl_new_sample4.tex`
- `paper_files/paper.bib`
- `results/paper_b1_changelog.md` (this file)

## Substantive edits

### 1. Introduction rewrite — R2#2
- Removed unsupported gradient-conflict / multi-task dominance motivation (and the `wang2025multi` framing that depended on it).
- Reframed problem as: task-driven fusion shapes embeddings only *indirectly* via prediction losses; binary contrastive/triplet assumptions fail for continuous RS annotations; need annotation-driven continuous geometry in a shared multi-modal latent space.
- Clarified claim: MMCTL uses the **same annotation sources** as task-driven baselines; difference is inductive bias (geometry matching vs prediction-loss fusion), not “no labels.”

### 2. New Related Work + contribution clarity — R1#1, R2#1, R2#3
- Added short `Related Work` section contrasting:
  - supervised contrastive / binary cross-modal triplets (`khosla2020supervised`, `dutt2022shared`);
  - soft/continuous metric learning (`kim2019deep`) and adaptive-margin triplets (`ha2021adaptive`);
  - prior shared-manifold multi-sensor triplets (`dutt2022shared`);
  - authors’ prior multi-annotation continuous triplet (`zhou2025multi`, unimodal detection).
- Explicit contribution bullets: (i) joint annotation vector + weighting Eqs.~\eqref{eq:ann-norm}–\eqref{eq:ann-concat}; (ii) margin-free continuous triplet / annotation-distance gap Eq.~\eqref{eq:mm-adaptive}; (iii) pairwise distance alignment Eq.~\eqref{eq:mm-reg}; (iv) multi-modal shared space with zero annotation distance for same-object cross-modal pairs.

### 3. Method clarifications — R1#3, R1#4, R1#5 (R2#5 prep only in comments)
- **Triplet construction:** batch-all; positives/negatives from annotation-distance ordering only; no hard similarity threshold (`eq:triplet-order` + prose).
- **Margin:** no margin hyperparameter; adaptive margin = \(D(y_a,y_q)-D(y_a,y_p)\) in `eq:mm-adaptive`.
- **Normalization/weighting:** pointed to Eqs.~\eqref{eq:ann-norm}–\eqref{eq:ann-weight} and class-weight ablation Fig.~\ref{fig:ablation2}.
- **Two-stage freeze:** reframed as fair fusion evaluation protocol (frozen modality encoders for all fusion methods); figure caption updated accordingly. End-to-end unfreeze left as optional follow-up (no fake E2E numbers).

### 4. Experimental Setup claim fix — R1#1 / R2#1 support
- Replaced “MMCTL relies solely on annotation-derived distance supervision” wording that could be read as “no labels.”
- Now states: same annotation sources as baselines; MMCTL applies geometry matching rather than supervised prediction losses during fusion training.
- Clarified that reported “runs” = CV folds (8 AWIR / 5 NEON).

### 5. Easy fixes — R2#6, R2#8 (+ typo)
- Fig.~ablation caption: aligned “5 runs” with Experimental Setup → mean±std over **5 NEON CV folds**.
- LaTeX typo: `83.65 \err0.88` → `83.65 \err{0.88}`.
- Bibliography:
  - `zhou2025multi`: updated from arXiv-only to **IGARSS 2025** (DOI `10.1109/IGARSS55030.2025.11242980`, pages 7004–7008 per citing sources).
  - `reiter2020deep`: cleaned arXiv metadata; **no peer-reviewed venue found** (noted in bib).
  - Added `khosla2020supervised`, `ha2021adaptive` for Related Work.

### 6. Placeholders for B2/B3 — structure only
- Commented `% TODO(B2)` / `% TODO(B3)` near Experimental Setup and Conclusion for compute cost, missing modality, significance, failure analysis, and table updates.
- **No empty unfinished sections** added to the body.

## Numeric tables
- **Unchanged.** Downstream / structure-score numbers left as in prior draft pending Agent A.

## Page-budget estimate
- Prior compiled PDF: **6 pages**.
- B1 adds Related Work + method/setup clarifications (~0.75–1.25 pages estimated).
- Expected after recompile: roughly **7–7.5 pages**, within ≤8 target (recompile not run here: no `pdflatex` in environment).

## Gate / readiness notes
- Agent A artifacts **not yet present** (`replication_awir.json`, `new_experiments_summary.md`, etc.).
- B1 complete. **Do not start B2/B3 numeric edits** until parent resumes Agent B after A2/A3 gates.
- Ready for B2 when `results/replication_awir.json` + `results/replication_awir_delta.md` exist.
- Ready for B3 when `results/new_experiments_summary.md` exists.

## Conflicts with Agent A
- None. No `mmctl/` code touched; no table numbers overwritten; no experiment outputs claimed.
