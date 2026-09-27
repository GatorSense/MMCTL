# NEON Freeze vs. End-to-End (R1.5)

Protocol: NEON 3-mod MMCTL recon checkpoints (/Users/meilunzhou/Documents/GitHub/MMCTL/outputs/neon_paper_recon); freeze vs E2E; runs=5, epochs=120, patience=20, freeze_lr=0.001, e2e_lr=0.0001, bs=32, hidden=256; standardized box/height targets; data=/Users/meilunzhou/Documents/GitHub/MMCTL/neon_data.npz

| Protocol | Acc % | Box R² | Height R² |
|---|---|---|---|
| frozen | 81.99±1.02 | 0.316±0.040 | 0.514±0.091 |
| e2e | 80.66±1.35 | 0.344±0.068 | 0.572±0.034 |

Both arms start from the same MMCTL-pretrained encoders and share the head architecture, splits, targets and budget; the only difference is whether the encoders are frozen or fine-tuned.
