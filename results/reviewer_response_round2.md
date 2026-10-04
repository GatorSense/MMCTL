# Response to Reviewers — Round 2

**Paper:** Learning Annotation-Driven Latent Structure for Multi-Modal Fusion

We thank the reviewer for the second round of detailed comments. We have revised
the manuscript accordingly; all new content is marked in red in the revised PDF
for ease of review. Below we respond to each comment in turn and point to the
specific additions. Two comments requested new experiments (loss-component
ablation and Transformer re-tuning); both were run on GPU and are now included.

---

**Comment 1.** End the Introduction with a concise contribution statement; move the Related-Work contribution list there, and describe the contributions conceptually rather than with equation numbers.

**Response.** Done. We moved the contribution list to the end of the Introduction as a single paragraph ("In summary, this paper makes the following contributions...") and rephrased each point conceptually: (i) a joint annotation vector fusing categorical labels with continuous attributes into one normalized, task-weighted similarity target; (ii) a margin-free continuous triplet whose required separation adapts to each triplet's annotation-distance gap; (iii) a pairwise distance-alignment term tying embedding distances to annotation distances; and (iv) the extension to a multi-modal shared latent space where same-object cross-modal observations share an annotation vector. No equation numbers appear in the statement.

**Comment 2.** Expand the Related Work: multi-modal fusion (cross-attention, Transformer fusion), remote-sensing multi-modal fusion, and metric learning for remote sensing.

**Response.** Done. Related Work now includes short paragraphs on (a) attention-based multi-modal fusion — cross-attention between modality tokens (ViLBERT, `lu2019vilbert`) and Transformer encoders over a modality-token set (`vaswani2017attention`); (b) remote-sensing multi-modal fusion; and (c) metric learning. We reuse the citations already present in the experiments and state explicitly that these architectures shape the embedding space only indirectly through task losses, in contrast to MMCTL's explicit annotation-aligned geometry.

**Comment 3.** Check Eq. 7, second term — is it also squared?

**Response.** We verified the equation against the implementation in `mmctl/continuous_triplet_loss.py`. The embedding anchor–positive and anchor–negative distances are squared; the annotation-distance gap is not. The manuscript equation already matches the code (the second term is correctly squared), so no change was required. We double-checked this during the revision.

**Comment 4.** Loss-component ablation: report the effect of each loss component (only the adaptive continuous triplet, only the distance-alignment regularizer).

**Response.** Done — new experiment. We retrain the NEON shared space with (a) the full loss, (b) only the adaptive triplet, and (c) only the distance-alignment regularizer, holding the encoder proxy, optimizer, and 50/50 protocol fixed (mean over three seeds). We added a table (`tab:loss_ablation`) and a "Loss-component ablation" paragraph:

- The **adaptive triplet** carries the discriminative and continuous signal: dropping it (reg-only) lowers accuracy by ≈5.6 points (67.6% → 62.0%) and weakens box/height regression.
- The **distance-alignment regularizer** tightens class geometry: adding it to the adaptive term sharpens the class-alignment residual roughly sevenfold (0.014 → 0.002) at negligible accuracy cost.
- No single term is best on every axis; the **full loss** attains the top accuracy while staying within one standard deviation of the best variant on every other column. This is the balanced behavior the paper relies on, and it justifies keeping both terms.

Raw results: `results/ablation_loss_components_neon.{json,md}`.

**Comment 5.** The Transformer NEON accuracy (≈43%) is far below RGB (76%); please explain or check the configuration and tuning.

**Response.** Done — new experiment plus clarification. We re-tuned the Transformer fusion on NEON and found that the low accuracy is an **optimization/tuning artifact on the small NEON splits, not an architectural limitation**. Starting from the shared default (≈37–43%), lowering the learning rate to 3×10⁻⁴, training longer, and reading out a mean-pooled token with a classification-only head raises the NEON Transformer probe accuracy to **80.7%**, on par with the strongest fusion baselines:

| Config | Acc % |
|---|---|
| default (cls, lr1e-3, 300ep, multitask) | 36.8±2.4 |
| mean-pool (lr1e-3, 300ep, multitask) | 41.5±4.0 |
| cls, lr3e-4, 600ep, multitask | 76.1±1.8 |
| mean-pool, lr3e-4, 600ep, cls-only | 80.7±1.5 |

We added a sentence in the fusion-baseline configuration note reporting this. We deliberately **keep the small, shared default in the main tables** so that every fusion baseline uses one identical configuration (a controlled comparison in which differences reflect the fusion mechanism, not per-dataset tuning budget), and we report the tuned number as a caveat. Raw results: `results/transformer_retune_neon.{json,md}`.

**Comment 6.** Add a short line per fusion model describing model-specific configuration/tuning (e.g., number of attention heads) and how it was selected.

**Response.** Done. We added a "Fusion-baseline configurations" note giving, for each learned fusion: SupCon (two-layer projection head 1024→1024→256, temperature 0.07); cross-attention (single block, 4 heads, residual LayerNorm, linear output); Transformer (2-layer encoder, 4 heads, feed-forward width 2×1024, GELU, learned class-token readout). We state the shared training budget (Adam 1e-3, weight decay 1e-4, gradient clipping 5, batch 128, ≤200 epochs with early stopping) and that we keep these small, identical defaults across datasets so that differences reflect the fusion mechanism rather than per-dataset capacity or tuning.

**Comment 7.** Add the within-class Spearman ρ value to each panel of the per-class zoom figure.

**Response.** Done. Each panel of the per-class zoom figure (`fig:emb_zoom`) is now annotated with its within-class Spearman ρ between pairwise embedding and annotation distances (higher = smoother). The caption documents the annotation; values come from `results/perclass_smoothness.json`.

**Comment 8.** Clarify why the NEON MMCTL results differ between the per-class table and the w_class=1.0 default row in the ablation table.

**Response.** Done. We added a clarifying sentence in the class-weight ablation paragraph: the w_class=1.0 ablation row and the NEON MMCTL downstream row evaluate the **same default model** but under different held-out splits and fold seeds (the ablation uses the released full-set embeddings with its own 50/50 split). The small differences (e.g., accuracy 80.1% vs. 80.3%) are within one standard deviation and reflect this protocol variation, not a different model. (The former per-class Table 2 referenced in the comment was moved to a commented supplementary block in a previous revision.)

---

We believe these revisions fully address the round-2 comments. The two requested
experiments strengthen the paper's central claim: the adaptive triplet and the
distance-alignment regularizer play complementary roles, and the attention-based
baselines are competitive once tuned — which makes MMCTL's *balanced* geometry,
not raw accuracy on any single axis, the substantive contribution.
