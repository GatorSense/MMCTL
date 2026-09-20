# Agent Handoff Contract

## Ownership
- **Agent A (Experiments):** `mmctl/`, training scripts, `outputs/`, `results/` (except paper edits)
- **Agent B (Paper):** `paper_files/*.tex`, `paper.bib`, `paper_files/figures/` (new figure files only)

## Required artifacts from Agent A
1. `results/replication_awir.json` + `results/replication_awir_delta.md` (Phase A2 gate)
2. `results/compute_cost.json` (R2#5)
3. `results/significance.json` (R1#7)
4. `results/missing_modality.json` (R2#4)
5. `results/failure_analysis.md` + optional figures under `paper_files/figures/` or `results/figures/`
6. `results/new_experiments_summary.md` (Phase A3 gate)
7. Optional: `results/stronger_baseline.json`, `results/freeze_vs_e2e.json`

## Data paths
- AWIR: `/Users/meilunzhou/Documents/GitHub/CTL/awir_data.npz`
- NEON: `/Users/meilunzhou/Documents/GitHub/CTL/neon_data.npz`
- Pretrained AWIR embeddings: `pretrained_embeddings/`
- Sample: `sample_data.npz`

## Defaults
- Prefer missing-modality over cross-region for transfer stress test
- Optional stronger baseline / freeze-vs-E2E only after clean A2
- If replication deltas are large: document clearly; do not silently overwrite paper tables
