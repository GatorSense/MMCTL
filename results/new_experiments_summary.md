# New Experiments Summary (Agent A → Agent B)

## Completed

| Item | Status | Path |
|---|---|---|
| Setup notes | Done | `results/setup_notes.md` |
| AWIR structure replication | **PASS** | `results/replication_awir.json`, `results/replication_awir_delta.md` |
| AWIR downstream (proxy) | Partial PASS | `results/taskhead_proxy_awir.json` (+ delta MD) |
| NEON | Deferred | `results/neon_status.md` |
| Compute cost (R2#5) | Done (microbench) | `results/compute_cost.json` |
| Significance (R1#7) | Done | `results/significance.json` |
| Missing modality (R2#4) | Done (AWIR) | `results/missing_modality.json` |
| Failure/retrieval (R2#7, R1#2) | Done | `results/failure_analysis.md`, `results/figures/` |
| Stronger baseline / freeze-vs-E2E | **Skipped** (optional; A2 DeepSet gap + no checkpoints) | — |

## Key takeaways for the paper agent

1. **Latent structure table (AWIR):** Safe to keep. Code reproduces paper numbers; largest drift MMCTL B (−0.13). No table rewrite needed.
2. **Downstream table (AWIR):** Keep paper numbers. Proxy confirms MMCTL/Concat/unimodal **classification**; DeepSet proxy is broken/unusable for absolute claims. Say replication used released test embeddings + frozen linear probes where checkpoints unavailable.
3. **Significance:** MMCTL vs best baseline significant on all three structure scores (Wilcoxon p=0.0078, n=8) and on proxy acc / B R² / P R² vs Concat (p≤0.016). Good for R1#7 response.
4. **Missing modality:** Zeroing thermal half of MMCTL joint emb drops proxy Acc **93.4 → 84.7** (−8.8 pts); zeroing RGB → **86.1** (−7.3 pts). Still above unimodal RGB/Thermal (~78–80). Use for R2#4.
5. **Compute cost:** Paper batch 32 ⇒ ≤249,984 ordered triplets/batch after modality concat (upper bound); ~2.0M/epoch upper bound at 8 steps. M4 microbench ~1.7 s/step at B=8; full 8×10×100 schedule not re-run. Be honest about upper-bound vs masked positives.
6. **Failure analysis:** Emphasize **class-level** cross-modal NN (~81% R1); do **not** claim strong identity retrieval (only 7.5% R1). Figures ready under `results/figures/`.
7. **NEON / ablation:** No NEON embeddings in repo — do not claim code replication for NEON tables.

## Scripts added
- `scripts/sklearn_probe_awir.py` — downstream proxy + missing modality
- `scripts/eval_pretrained_taskheads.py` — torch head trainer (slower; smoke only)
- `scripts/measure_compute_cost.py`
- `scripts/significance_tests.py`
- `scripts/failure_analysis_awir.py`
- Minimal lazy-TF import in `mmctl/embedding_metrics.py` (metrics without loading TF)

## Env
- `.venv_exp/` (Python 3.9 + torch 2.8 MPS + TF 2.20). See `results/setup_notes.md`.

## Recommended next steps for Agent B
1. Add a short “Reproducibility” note: structure from `demo_metrics.py` + pretrained embeddings; downstream absolute DeepSet/train-protocol numbers require original checkpoints.
2. Insert significance + missing-modality sentences/tables citing the JSON numbers above (do not invent).
3. Optionally add retrieval PCA / failure figure if space; caption must match class-vs-identity nuance.
4. Leave NEON tables as-is with Zenodo/CTL provenance; flag NEON code gap if reviewers ask for scripts.
