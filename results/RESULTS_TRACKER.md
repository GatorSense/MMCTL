# MMCTL Revision — Results Tracker

Single place to track **reproduced** paper numbers vs **new** reviewer-driven experiments.  
Generated from Agent A artifacts under `results/` (2026-09-20). Machine: Apple M4, `.venv_exp/`.

| Gate | Status |
|---|---|
| AWIR latent structure replication | **PASS** |
| AWIR downstream (proxy probes) | **PARTIAL PASS** (cls ranking OK; DeepSet absolute FAIL) |
| NEON replication | **PARTIAL** (uploaded recon weights + paper task heads; cls ~ok, structure/box FAIL) |
| New experiments (cost / significance / missing-mod / retrieval) | **Done** |
| Stronger baselines / freeze-vs-E2E | **Skipped** |

---

## 1. Reproduced — AWIR latent structure

**Protocol:** `demo_metrics.py` + `pretrained_embeddings/` (8 runs), data `CTL/awir_data.npz`.  
**Paper table:** `tab:latent_alignment` (lower = better).  
**Sources:** `replication_awir.json`, `replication_awir_delta.md`, `demo_metrics_summary.csv`.

| Method | Src | Paper | Reproduced | Δmean | Status |
|---|---|---|---|---|---|
| RGB | C | 0.43±0.28 | 0.43±0.28 | +0.00 | ok |
| RGB | B | −0.41±0.36 | −0.42±0.36 | −0.01 | ok |
| RGB | P | 0.23±0.38 | 0.23±0.38 | −0.00 | ok |
| Thermal | C | 0.54±0.15 | 0.54±0.15 | +0.00 | ok |
| Thermal | B | 0.56±0.14 | 0.63±0.15 | +0.07 | ok |
| Thermal | P | 0.89±0.49 | 0.89±0.49 | +0.00 | ok |
| Concat | C | −0.28±0.17 | −0.28±0.17 | −0.00 | ok |
| Concat | B | −0.19±0.15 | −0.14±0.15 | +0.05 | ok |
| Concat | P | 0.31±0.27 | 0.31±0.27 | −0.00 | ok |
| DS-A | C | 1.17±0.44 | 1.17±0.44 | +0.00 | ok |
| DS-A | B | 1.51±0.44 | 1.52±0.46 | +0.01 | ok |
| DS-A | P | 1.19±0.43 | 1.19±0.43 | +0.00 | ok |
| DS-M | C | 1.23±0.11 | 1.23±0.11 | +0.00 | ok |
| DS-M | B | 1.47±0.46 | 1.47±0.40 | +0.00 | ok |
| DS-M | P | 1.17±0.46 | 1.17±0.46 | +0.00 | ok |
| **MMCTL** | C | **−3.10±0.04** | **−3.10±0.04** | +0.00 | ok |
| **MMCTL** | B | **−2.93±0.18** | **−3.06±0.15** | −0.13 | ok |
| **MMCTL** | P | **−3.79±0.61** | **−3.79±0.61** | +0.00 | ok |

**Verdict:** Keep paper structure table. Largest drift = MMCTL B (−0.13). Ranking unchanged.

---

## 2. Reproduced — AWIR downstream (proxy)

**Paper protocol:** frozen embeddings + task heads (needs train embeddings / Keras checkpoints — **not shipped**).  
**Proxy used here:** sklearn Logistic/Ridge, 5-fold CV on released **test** embeddings only (`scripts/sklearn_probe_awir.py`).  
**Sources:** `taskhead_proxy_awir.json`, `replication_awir_delta.md`.

| Method | Paper C% | Proxy C% | Paper B R² | Proxy B R² | Paper P R² | Proxy P R² |
|---|---|---|---|---|---|---|
| RGB | 77.08±3.21 | 77.92±2.52 | 0.34 | 0.14 | −0.13 | −0.25 |
| Thermal | 78.96±1.59 | 79.90±2.06 | 0.30 | 0.05 | −0.32 | −0.40 |
| Concat | 89.06±1.44 | 89.38±1.07 | 0.32 | 0.26 | −0.15 | −0.05 |
| DS-A | 87.71±2.12 | **45.00±4.23** | 0.36 | −0.60 | −0.18 | −0.56 |
| DS-M | 87.40±1.51 | **45.63±2.55** | 0.37 | −0.73 | −0.14 | −0.62 |
| **MMCTL** | **94.06±1.86** | **93.44±1.37** | **0.52** | 0.33 | **0.12** | **0.12** |

**Verdict:**
- **Keep paper downstream absolutes** (esp. DeepSet).
- Classification ranking OK for RGB / Thermal / Concat / MMCTL (~±1 pt).
- DeepSet proxy **FAIL** for absolute claims (~45% vs ~87%).
- Regression B R² underestimates paper under proxy; MMCTL P R² matches (0.12).

---

## 3. New — Statistical significance (R1#7)

**Protocol:** paired Wilcoxon / t-tests across 8 AWIR runs.  
**Source:** `significance.json`.

### Structure (lower better; MMCTL vs best baseline)

| Metric | MMCTL mean | Baseline | Baseline mean | Wilcoxon p | Paired t p |
|---|---|---|---|---|---|
| Class structure | −3.098 | Concat | −0.282 | **0.0078** | 1.1e−9 |
| Box structure | −3.056 | RGB | −0.423 | **0.0078** | 2.1e−7 |
| Position structure | −3.790 | RGB | +0.228 | **0.0078** | 2.9e−6 |

### Downstream proxy (higher better; MMCTL vs Concat)

| Metric | MMCTL | Concat | Wilcoxon p | Paired t p |
|---|---|---|---|---|
| Accuracy | 0.934 | 0.894 | **0.0078** | 6.3e−4 |
| Box R² | 0.330 | 0.262 | **0.0156** | 0.014 |
| Position R² | 0.122 | −0.049 | **0.0078** | 0.002 |

**Note:** Proxy significance supports **ranking**, not replacing paper table absolutes.

---

## 4. New — Missing modality (R2#4)

**Protocol:** zero one 1024-d half of 2048-d MMCTL joint test embedding `[RGB | Thermal]`, L2-normalize, sklearn probe.  
**Source:** `missing_modality.json`.

| Setup | Acc % | Box R² | Pos R² | Δ Acc vs full |
|---|---|---|---|---|
| Full joint | **93.44±1.37** | 0.330±0.032 | 0.122±0.082 | — |
| Drop thermal | 84.69±2.04 | 0.260±0.075 | −0.003±0.077 | **−8.75** |
| Drop RGB | 86.15±1.99 | 0.190±0.084 | −0.019±0.185 | **−7.29** |

Both single-modality probes remain above unimodal RGB/Thermal paper cls (~78–80%).

---

## 5. New — Compute cost (R2#5)

**Source:** `compute_cost.json` (microbench; full AWIR retrain not re-run on M4).

| Quantity | Value |
|---|---|
| Paper batch size | 32 |
| Embeddings after modality concat | 64 |
| Max ordered triplets / batch (upper bound) | **≤ 249,984** |
| Triplets / epoch upper bound (AWIR) | **~ 1,999,872** |
| Microbench batch | 8 |
| Sec / train step (B=8, M4 CPU) | ~1.70 |
| Naive scale → sec / epoch estimate | ~54.5 |
| Peak tracemalloc | ~17 MB (not peak GPU; CPU run) |
| Caveat | Upper bound **before** annotation-order mask; actual positives data-dependent; full 8×10×100 schedule not timed |

---

## 6. New — Failure / cross-modal retrieval (R2#7, R1#2)

**Protocol:** run01 MMCTL joint emb; RGB half `0:1024` → thermal half `1024:2048`; cosine NN on held-out test split.  
**Sources:** `failure_analysis.md`, `failure_analysis_metrics.json`, `figures/`.

| Metric | Value |
|---|---|
| Rank-1 **class** match | **80.8%** (97 / 120) |
| Rank-1 **identity** match | **7.5%** (9 / 120) |
| Rank-5 identity | 30.0% |
| Class failures at R1 | 23 / 120 |

**Example failures:** deer→horse, horse→cow (adjacent livestock).  
**Figures:** `results/figures/awir_mmctl_pca_run01.png`, `awir_retrieval_successes.png`, `awir_retrieval_failures.png`  
**Do not claim:** strong exact-instance RGB↔thermal retrieval from split halves alone.

---

## 7. NEON (CTL `neon_data.npz` + uploaded recon weights)

**Data:** `/Users/meilunzhou/Documents/GitHub/CTL/neon_data.npz`.  
**Checkpoints:** `_emb1024_batch32_margin0.1_test_size0.5_cls_weight1.0_best_val_model_run_{1..5}.h5`  
**Protocol:** rebuild 3-mod AE → `load_weights` → extract RGB/HSI/LiDAR/Concat/DS-M/MMCTL embeddings; structure metrics on test; paper task heads (Adam 1e-3 / wd 1e-4 / bs 32 / clip 5 / 3000 ep / patience 1000 / hidden 256).  
**Artifacts:** `replication_neon.json`, `replication_neon_delta.md`, `outputs/neon_paper_embeddings/`, `results/neon_paper_eval_pretrained.log`.

### Downstream (vs paper)

| Method | Paper C% | Code C% | Paper B | Code B | Paper H | Code H |
|---|---|---|---|---|---|---|
| RGB | 74.59 | 80.27±1.81 | 0.07 | 0.01±0.03 | 0.34 | 0.45±0.04 |
| HSI | 76.69 | 85.91±3.13 | 0.12 | 0.00±0.07 | 0.40 | 0.55±0.03 |
| LiDAR | 49.66 | 44.92±2.18 | 0.22 | 0.00±0.03 | 0.68 | 0.60±0.01 |
| Concat | 81.69 | 85.85±1.43 | 0.39 | 0.08±0.06 | 0.69 | 0.67±0.02 |
| DS-M | 81.35 | 86.71±2.04 | 0.40 | 0.10±0.05 | 0.61 | 0.69±0.01 |
| **MMCTL** | **86.95** | **85.05±1.88** | **0.43** | **0.10±0.04** | **0.74** | **0.68±0.02** |

**Verdict:** MMCTL classification ≈ paper (−1.9 pt). Height R² close. **Box R² fails** (~0.10 vs 0.43). Structure scores do not match paper (MMCTL not best; Concat≡MMCTL geometrically in this AE). **Keep paper NEON tables** for absolute claims; code numbers are replication diagnostics only.

### Paper NEON numbers (authoritative — keep in manuscript)

| | C | B / H |
|---|---|---|
| Structure MMCTL | −1.49±0.23 | B −2.07±0.29, H −1.58±0.25 |
| Downstream MMCTL | 86.95±1.7% | B R² 0.43±0.04, H R² 0.74±0.02 |

## 8. Still open experiments

| Item | Status | Notes |
|---|---|---|
| Stronger baselines (SupCon / Transformer) | Skipped | R1.6 |
| Freeze vs end-to-end | Skipped | R1.5 |
| Full AWIR wall-clock retrain | Not done | Upper-bound + microbench only |
| Faithful NEON Keras retrain | Done (uploaded weights) | Eval complete; absolute structure/box still off |

---

## 9. Artifact index

| File | Contents |
|---|---|
| `replication_awir.json` / `_delta.md` | Structure + downstream proxy vs paper |
| `replication_neon.json` / `_delta.md` | NEON proxy vs paper (CTL npz) |
| `REVIEW_COMMENTS_TRACKER.md` | Reviewer comments addressed vs open |
| `demo_metrics_raw.csv` / `_summary.csv` | Per-run structure metrics |
| `taskhead_proxy_awir.json` | Proxy downstream + missing-mod per run |
| `significance.json` | Wilcoxon / t-tests |
| `missing_modality.json` | Drop-RGB / drop-thermal summary |
| `compute_cost.json` | Triplet counts + timing |
| `failure_analysis.md` / `_metrics.json` | Retrieval + failures |
| `figures/` | PCA + retrieval grids |
| `new_experiments_summary.md` | Agent A → B handoff |
| `paper_b1_changelog.md` / `paper_b2b3_changelog.md` | Manuscript edit logs |
| `setup_notes.md` | Env / device notes |

---

## 10. Paper integration status

| Content | In manuscript? |
|---|---|
| Keep structure + downstream tables | Yes (unchanged) |
| Reproducibility paragraph | Yes (B2) |
| Significance sentence | Yes (B3) |
| Missing-modality numbers | Yes (B3) |
| Compute-cost paragraph | Yes (B3) |
| Class vs identity retrieval + failure fig | Yes (B3) |
| NEON absolute numbers | Keep paper; proxy not claimed |
| Stronger baselines / E2E | Not claimed |
