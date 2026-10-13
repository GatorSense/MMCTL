# Round-3 Review Tracker (annotations on WACV v2 PDF)

**Manuscript:** `paper_files/wacv-2027-author-kit-template/mmctl_wacv.tex`
**Sources:** `WACV_v2_dutt.pdf` (Aditya Dutt), `WACV_v2 azare.pdf` (Taylor Glenn)
**Opened:** 2026-10-13
**Status: DONE (approved 2026-10-13) — all items applied to `mmctl_wacv.tex`.**

Two global tasks requested for this round:
- Apply the approved round-3 edits below. **Done.**
- **Final version:** remove all red highlighting (`\red{}`, `\new{}`, `{\color{red}…}`) so the text is plain. **Done** — `\new`/`\red`/`\todo` neutralized to pass-through and all `{\color{red}…}` groups unwrapped; 0 red remaining.

**Decisions (resolved):** ① global rename "margin-free" → "annotation-derived adaptive margin" (0 "margin-free" remaining). ② anonymized repo URL set to `https://anonymous.4open.science/r/MMCTL-C831/`. ③ red-diff backup saved as `paper_files/wacv-2027-author-kit-template/reddiff_final.tex` (round-3 edits + red intact); `mmctl_wacv.tex` is the clean final.

---

## A. Aditya Dutt (`WACV_v2_dutt.pdf`)

| # | Loc | Comment / marked text | Proposed action |
|---|---|---|---|
| D1 | Intro, contribution para (L50) | Suggested a condensed **3-point** contribution statement (quoted below) replacing the current 4-point `(i)–(iv)`. | Replace the `\new{In summary…}` paragraph with Dutt's 3-point version (verbatim, lightly LaTeX-formatted). Reframes the triplet as an **"annotation-derived adaptive margin"** and states AWIR/NEON results more precisely. |
| D2 | Abstract, last sentence (L40) | Suggested replacing the final "Experiments on AWIR and NEON … fusion approaches." with a 2-sentence results-oriented version (quoted below). | Replace that one sentence. Adds "best classification and regression on AWIR; competitive classification on NEON" — consistent with Tables 1/3. |
| D3 | Intro (L50, part of contribution para) | Plain highlight of "…multi-modal shared latent space, where observations of the same object…preserved." No explicit edit. | No separate change — this span is inside the paragraph replaced by **D1**. |
| D4 | Intro (L50, part of contribution para) | Plain highlight of "Together, these produce a latent space organized by annotation-defined similarity…" No explicit edit. | No separate change — removed as part of **D1** (Dutt's version drops this summary sentence). |
| D5 | Related Work (L73) | Highlighted "while our"; note "It reveals identity." + suggested "while Zhou et al. [18] organize a unimodal detector using joint class and box annotations." | **Double-blind fix.** Reword "while our prior multi-annotation continuous triplet organizes a *unimodal* detector from joint class and box labels \cite{zhou2025multi}" → third person, e.g. "while Zhou et al.\ \cite{zhou2025multi} organize a *unimodal* detector from joint class and box annotations." (Same issue as **G1**.) |
| D6 | Within-class para (L409) | Struck out "—unlike those baselines—". | Delete the phrase "---unlike those baselines---" (sentence still reads cleanly). |
| D7 | Conclusion (L577) | Highlighted "a margin-free continuous triplet,"; note "a continuous triplet with an annotation-derived margin". | Reword in the conclusion. **See decision ① on terminology.** |
| D8 | Conclusion (L577) | Highlighted "matches end-to-end fine-tuning,"; note "yields comparable classification performance to end-to-end fine-tuning". | Replace "matches end-to-end fine-tuning" → "yields comparable classification performance to end-to-end fine-tuning". |

**D1 — Dutt's suggested contribution text (verbatim):**
> This paper makes three contributions. (i) We propose an annotation-driven multimodal framework that constructs a shared, structured latent space using a normalized, task-weighted joint annotation vector combining categorical labels and continuous attributes. Observations of the same object across modalities share this vector, encouraging cross-modal alignment. (ii) We combine annotation-ordered continuous triplets with pairwise distance alignment, using an annotation-derived adaptive margin in place of a fixed triplet margin. (iii) We evaluate class separation, continuous-attribute distance correspondence, and downstream performance on AWIR and NEON, demonstrating improved geometry over concatenation on both datasets and classification and regression gains on AWIR. It is also evaluated how changing the class-label weight affects classification and regression performance on NEON.

**D2 — Dutt's suggested abstract text (verbatim):**
> Across AWIR and NEON, MMCTL learns a shared, structured latent space that reflects both class identity and continuous attributes across modalities. It achieves the best classification and regression results among the evaluated methods on AWIR and competitive classification on NEON, demonstrating the value of explicitly learning shared latent structure from joint annotations.

---

## B. Taylor Glenn (`WACV_v2 azare.pdf`)

| # | Loc | Comment / marked text | Proposed action |
|---|---|---|---|
| G1 | Related Work (L73) | Highlighted "while our prior multi-annotation"; note "double blind, drop 'our' – use 'while prior'". | **Double-blind fix**, same span as **D5**. Change "while our prior …" to third person ("while prior work \cite{zhou2025multi} organizes …" / "while Zhou et al.\ …"). |
| G2 | Problem Setup (~L147–153) | Highlighted "class" / "wclass"; note "unclear at this point what 'class' is". | At first mention of $w_{\text{class}}$, add a short gloss, e.g. "$w_{\text{class}}$ (the weight on the categorical class-label task)", so the reader knows what "class" denotes before the weight sweep. |
| G3 | Intro (L174) | Highlighted "provide"; note "*may* provide". | Hedge: "Multi-modal observations **may** provide complementary and non-redundant information…". |
| G4 | Exp. setup (L299) | Highlighted "requested for this revision"; note "remove". | Delete "requested for this revision" (reads: "…and three stronger fusion baselines: SupCon …"). |
| G5 | Reproducibility (L304) | Highlighted "at github.com/GatorSense/MMCTL"; note "double blind, make an anonymous repo". | **Double-blind fix.** Replace the GatorSense URL with an anonymized repo placeholder (e.g. `anonymous.4open.science`), or "[anonymized repository]" until the anon repo exists. **See decision ②.** |

---

## Decisions needed before editing

**① "margin-free" vs. "annotation-derived adaptive margin" (from D1/D7).**
Dutt reframes the loss as using an *annotation-derived adaptive margin* rather than being *margin-free*. "margin-free" currently appears **5×** (L50, L73, L548, L552 caption, L577). Mixing both terms would be inconsistent.
- **Recommended:** globally rename "margin-free" → "annotation-derived adaptive margin" (and adjust "MMCTL is margin-free" on L73 to "MMCTL uses an annotation-derived adaptive margin"). Consistent with Dutt's framing.
- Alternative: change only the two spots Dutt flagged (L50, L577) and leave "margin-free" elsewhere (ris—inconsistent).

**② Anonymous repo URL (from G5).**
What should replace `github.com/GatorSense/MMCTL` for the double-blind submission?
- `anonymous.4open.science` link (you create the anon mirror), **or**
- neutral text "[anonymized repository]" / "an anonymized repository" placeholder for now.

**③ Red highlighting removal scope.**
"For the final version, remove the red highlighting." Plan: neutralize `\red{}` and `\new{}` (keep their content, drop the color) and unwrap `{\color{red}…}` table/section groups, across `mmctl_wacv.tex`. Keep a red-diff backup copy, or edit in place?
