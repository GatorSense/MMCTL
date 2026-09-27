#!/usr/bin/env python3
"""Render the point-by-point reviewer response letter to PDF (reportlab).

pdflatex is unavailable in this environment, so we render the rebuttal directly.
Outputs:
  paper_files/MMCTL_reviewer_response.pdf
  results/reviewer_response.md  (editable source)
"""
from __future__ import annotations
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, ListFlowable,
                                ListItem, HRFlowable)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "paper_files", "MMCTL_reviewer_response.pdf")
MD = os.path.join(ROOT, "results", "reviewer_response.md")

INTRO = ("We thank the reviewers for their careful reading and constructive "
         "suggestions. We have revised the manuscript accordingly; all new "
         "content is marked in red in the revised PDF for ease of review. Below "
         "we respond to each comment in turn and indicate the specific additions.")

# Each item: (comment, response, [bullets])
R1 = [
 ("Clarify the essential difference between MMCTL and supervised contrastive "
  "learning or cross-modal triplet loss, especially the added contribution of "
  "continuous similarity constraints.",
  "We rewrote the Related Work section and added an explicit statement of the "
  "contribution. Unlike binary supervised contrastive learning and fixed-margin "
  "cross-modal triplets, MMCTL orders embedding distances by graded, continuous "
  "annotation proximity rather than binary co-membership. Concretely, MMCTL adds:",
  ["a joint annotation vector with feature-wise normalization and per-task weights (Eqs. 1-3);",
   "a margin-free continuous triplet whose separation target is the annotation-distance gap (Eq. 7), with no tuned margin;",
   "a pairwise distance-alignment regularizer matching embedding to annotation distances (Eq. 8);",
   "extension to a multi-modal shared space where same-object, different-modality pairs have zero annotation distance, directly pulling cross-modal embeddings together."]),
 ("The paper claims MMCTL learns a more interpretable latent space; "
  "visualizations or cross-modal retrieval examples are recommended.",
  "We added embedding-space visualizations and quantitative interpretability metrics:",
  ["Figure 1: AWIR fused embeddings (PCA and t-SNE) across all fusion methods, colored by class and sized by bounding-box area;",
   "Figure 3: per-class \"zoom\" panels colored by the continuous label, showing smooth within-class transitions;",
   "Table 1: standard metrics -- silhouette (class separation) and Spearman distance-preservation (continuous attributes)."]),
 ("Specify the rules for selecting positive and negative samples in continuous "
  "triplets, as well as the similarity thresholds or margin settings.",
  "Clarified in the Shared Latent Space subsection. We use a batch-all strategy: "
  "every in-batch index is an anchor, and a pair (p,q) is a valid (positive, "
  "negative) pair iff p is closer to the anchor than q in joint annotation space "
  "(Eq. for triplet ordering). Positives/negatives are defined solely by "
  "annotation-distance ordering; there is no hard similarity threshold and no "
  "tuned margin (the loss is margin-free).", []),
 ("How are annotation attributes such as category, object size, and height "
  "normalized and weighted? Does the weighting strategy affect the final results?",
  "The Problem Setup gives feature-wise min-max normalization, task-length "
  "rescaling, and fixed per-task weights (Eqs. 1-3; AWIR w_class=0.5, NEON "
  "w_class=1.0). We added a class-weight sensitivity study:",
  ["Table 3 and Figure 4 sweep w_class in {0.5, 0.8, 1.0, 1.5};",
   "accuracy rises monotonically with w_class (74.9% -> 84.6%) while box/height regression stays stable, confirming the weight acts as an interpretable semantic-vs-geometric trade-off."]),
 ("What is the rationale for freezing the encoder in the two-stage training "
  "scheme? A comparison with end-to-end fine-tuning would help validate this.",
  "We justify freezing as a fair fusion-evaluation protocol (it isolates fusion "
  "geometry from unequal encoder capacity) and added a controlled comparison on "
  "NEON from the same pretrained encoders:",
  ["frozen: 81.99 +/- 1.02% accuracy; end-to-end: 80.66 +/- 1.35%;",
   "end-to-end yields only marginal gains on the continuous heads, so freezing is justified and substantially cheaper."]),
 ("The baseline set is limited. Representative multimodal methods such as "
  "supervised contrastive learning, cross-attention, or Transformer-based fusion "
  "should be included.",
  "We removed the DeepSet baselines and added three stronger, representative "
  "fusion baselines with citations: supervised contrastive fusion (SupCon), "
  "cross-attention pooling, and a Transformer encoder over modality tokens. They "
  "appear in Tables 1 and 2 and in the embedding visualizations.", []),
 ("Report standard deviations and statistical significance tests to verify "
  "whether MMCTL's improvements are stable and reliable.",
  "All results tables now report mean +/- std over the cross-validation folds, and "
  "we added paired one-sided Wilcoxon signed-rank tests:",
  ["MMCTL's continuous-geometry advantage is significant on AWIR position (p=0.0078), with a positive trend on box (p=0.098);",
   "we are explicit that MMCTL's contribution is balanced coherence across metrics rather than winning every individual metric."]),
]

R2 = [
 ("State where the substantive increment relative to prior work lies -- the joint "
  "annotation vector/weighting (Eqs. 1-3), the margin-free continuous triplet "
  "(Eq. 7), the distance-alignment regularizer (Eq. 8), or the extension from "
  "single-modal to multi-modal shared latent space.",
  "The primary increment is the extension to a multi-modal shared latent space in "
  "which same-object, different-modality pairs share an identical joint annotation "
  "vector (zero annotation distance), directly coupling cross-modal embeddings. "
  "This is accompanied by the joint annotation vector construction/weighting "
  "(Eqs. 1-3), the margin-free continuous triplet (Eq. 7), and the "
  "distance-alignment regularizer (Eq. 8). Related Work now states this "
  "explicitly relative to our prior unimodal work.", []),
 ("The introduction uses gradient conflict and task dominance as motivation but "
  "provides no supporting experiments; revise the introduction to highlight the "
  "problem the paper solves.",
  "We revised the Introduction to remove the unsupported gradient-conflict/"
  "task-dominance motivation and instead frame the problem directly: task-driven "
  "fusion shapes latent geometry only indirectly, so embedding distances are not "
  "constrained to reflect annotation-defined similarity. The motivation now "
  "matches the problem addressed and the evidence provided.", []),
 ("Discuss the differences between MMCTL and existing soft/continuous similarity "
  "metric learning and adaptive-margin triplet losses.",
  "Related Work now contrasts MMCTL with continuous/soft-similarity metric "
  "learning (Kim et al.) and adaptive-margin triplets (Ha et al.): MMCTL is "
  "margin-free (its separation target is the annotation-distance gap, with no "
  "tuned margin) and operates in a multi-modal shared space, which those methods "
  "do not.", []),
 ("MMCTL also uses task labels; the difference from baselines is the loss form, "
  "not the information used. Add cross-dataset/cross-region transfer or "
  "missing-modality experiments.",
  "We agree the distinction is the inductive bias (loss form), not the label "
  "information, and we now state this explicitly. We added missing-modality "
  "experiments:",
  ["zeroing one modality half of the MMCTL joint embedding and re-probing yields graceful degradation (drop-thermal 84.7%, drop-RGB 86.2%) that stays above the unimodal baselines (~78-80%), evidencing complementary but non-redundant modality use;",
   "cross-region transfer is noted as future work in the Conclusion."]),
 ("The batch-all strategy is used but the number of triplets and computational "
  "cost are not reported. Report total triplets, training time per epoch, and GPU "
  "memory usage.",
  "We added a triplet-compute analysis. With batch size 32 and modality "
  "concatenation (64 embeddings), the batch-all construction yields at most "
  "249,984 ordered triplets per batch (~2.0M per epoch) before the "
  "annotation-order mask; a microbenchmark measures ~1.7 s per training step "
  "(batch 8) on an Apple M4 CPU. Our runs were CPU-based, so peak GPU memory was "
  "not measured; we state this explicitly and will add GPU timing/memory if "
  "GPU logs are required.", []),
 ("Clarify the relationship between \"k-fold cross-validation within the training "
  "set\" and \"averaged over 5 runs\" in the caption of Figure 3.",
  "Clarified in the Experimental Setup: the \"runs\" reported in the captions ARE "
  "the cross-validation folds (8 for AWIR, 5 for NEON), and each mean +/- std is "
  "computed across these folds. The ambiguous \"runs\" wording has been "
  "reconciled with the k-fold description.", []),
 ("Some downstream results need more detailed discussion: add error analysis of "
  "typical failure cases and explain when the method is effective and when it is "
  "not.",
  "We expanded the downstream discussion to explain when MMCTL helps and when it "
  "does not:",
  ["class-supervised baselines win raw class separation, and a directly sensing modality (LiDAR) wins its own continuous attribute, but each fails the other axis; MMCTL is the only method balanced across both;",
   "the per-class within-class analysis (Figure 3) identifies classes with strong continuous transitions (e.g., PIPA2 canopy height) versus weaker ones, providing a class-level effectiveness/error analysis."]),
 ("Update references to their formally published versions, e.g., reference [8].",
  "We are updating the bibliography to cite formally published versions in place "
  "of arXiv preprints where available, including reference [8]. (This audit is "
  "being finalized against the compiled reference list.)", []),
]


def build_md():
    lines = ["# Response to Reviewers", "",
             "**Paper:** Learning Annotation-Driven Latent Structure for Multi-Modal Fusion",
             "", INTRO, ""]
    for rname, items in (("Reviewer 1", R1), ("Reviewer 2", R2)):
        lines += [f"## {rname}", ""]
        for i, (c, r, bl) in enumerate(items, 1):
            lines += [f"**Comment {i}.** {c}", "", f"**Response.** {r}", ""]
            for b in bl:
                lines.append(f"- {b}")
            if bl:
                lines.append("")
    open(MD, "w").write("\n".join(lines))


def build_pdf():
    ss = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=ss["Title"], fontSize=17, spaceAfter=6)
    sub = ParagraphStyle("S", parent=ss["Normal"], fontSize=10.5,
                         textColor=colors.HexColor("#444444"), spaceAfter=10)
    rh = ParagraphStyle("RH", parent=ss["Heading2"], fontSize=13,
                        textColor=colors.HexColor("#1a3e6e"), spaceBefore=12, spaceAfter=6)
    comment = ParagraphStyle("C", parent=ss["Normal"], fontSize=10.5, leading=14,
                             leftIndent=2, spaceAfter=3, textColor=colors.HexColor("#222222"))
    resp = ParagraphStyle("R", parent=ss["Normal"], fontSize=10.5, leading=14, spaceAfter=4)
    bullet = ParagraphStyle("B", parent=ss["Normal"], fontSize=10.5, leading=13)
    intro = ParagraphStyle("I", parent=ss["Normal"], fontSize=10.5, leading=14, spaceAfter=8)

    story = [Paragraph("Response to Reviewers", title),
             Paragraph("Learning Annotation-Driven Latent Structure for Multi-Modal Fusion", sub),
             Paragraph(INTRO, intro),
             HRFlowable(width="100%", color=colors.HexColor("#cccccc")),
             Spacer(1, 6)]
    for rname, items in (("Reviewer 1", R1), ("Reviewer 2", R2)):
        story.append(Paragraph(rname, rh))
        for i, (c, r, bl) in enumerate(items, 1):
            story.append(Paragraph(f"<b>Comment {i}.</b> <i>{c}</i>", comment))
            story.append(Paragraph(f"<b>Response.</b> {r}", resp))
            if bl:
                story.append(ListFlowable(
                    [ListItem(Paragraph(b, bullet), leftIndent=12) for b in bl],
                    bulletType="bullet", start="•", leftIndent=14, spaceAfter=6))
            story.append(Spacer(1, 4))
    doc = SimpleDocTemplate(PDF, pagesize=letter, topMargin=0.8*inch,
                            bottomMargin=0.8*inch, leftMargin=0.9*inch, rightMargin=0.9*inch,
                            title="MMCTL Response to Reviewers")
    doc.build(story)


if __name__ == "__main__":
    build_md()
    build_pdf()
    print("wrote", PDF)
    print("wrote", MD)
