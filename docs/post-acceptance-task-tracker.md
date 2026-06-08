# ALIFE 2026 Camera-Ready Task Tracker

Source spec: `docs/post-acceptance-task-specs.md`  
Manuscript: `docs/recips_social_alife2026/main.tex`  
Last local audit: 2026-06-08
Implementation pass: 2026-06-08

## Goal

Prepare the accepted ALIFE 2026 paper for camera-ready / arXiv v2 by resolving
review-driven clarity and rigor issues without changing the core empirical
claim.

The final manuscript should make this narrow claim hard to misread:

> Partner-state access is not the active ingredient; regulatory routing is.

## Completion Notes

- Implemented all P0/P1/P2 tracker tasks on 2026-06-08.
- Regenerated `docs/recips_social_alife2026/fig_summary.pdf` and `.png`.
- Added reproducible score-decomposition code and artifact:
  `analysis/social_exact_solver.py` and
  `results/social-paper-paper/score_decomposition.csv`.
- Verified with targeted social tests and a successful LaTeX build of
  `docs/recips_social_alife2026/main.pdf` (7 letter-sized pages).

## Non-Negotiables

- Do not add new experiments unless existing code/data already support the
  needed value.
- Do not invent empirical values, confidence intervals, stochastic statistics,
  or new effect-size claims.
- Do not overclaim empathy, altruism, consciousness, moral status, or human-like
  social cognition.
- Preserve the clean minimal-intervention-device framing.
- Keep ALIFE template compatibility and page-limit pressure in mind.
- Report the exact FoodShareToy switch consistently as
  `\lambda^\star \approx 0.91` for the default state.
- If `0.95` appears, it must be clearly labeled as a tested/paper-profile
  coupled setting, not the exact threshold.

## Priority Key

- `P0`: camera-ready blocker.
- `P1`: important review-response improvement.
- `P2`: polish or future-proofing.

## Tasks

### [x] T01 P0 - Clean Up Front Matter

**Review driver:** Camera-ready status must not look like an under-review full
paper submission.

**Targets**

- `docs/recips_social_alife2026/main.tex`

**Current audit**

- `Submission type: \textbf{Full Paper}` is still present.
- `\blfootnote{Preprint. Under review at ALIFE 2026.}` is still present.
- Data/code line is present but should be normalized.

**Implementation notes**

- Remove the `Submission type` line.
- For arXiv v2, use accepted-preprint wording such as:
  `\textbf{Status:} Accepted as a talk at ALIFE 2026; proceedings camera-ready version forthcoming.`
- For the actual proceedings camera-ready, remove the status line unless the
  conference/template permits it.
- Keep the GitHub data/code URL.
- Verify author name, affiliation, and any copyright notice required by the
  final template.

**Acceptance checks**

- `rg -n "Submission type|Under review|under review" docs/recips_social_alife2026/main.tex`
  returns no stale pre-acceptance wording.
- Data/code URL remains visible in the manuscript source.

### [x] T02 P0 - Replace Abstract With Short Hardened Version

**Review driver:** Reviewers found the abstract too long and broad; arXiv HTML
may strip crucial math.

**Targets**

- `docs/recips_social_alife2026/main.tex`

**Current audit**

- Abstract is still long.
- It reports `\lambda^{\ast}\approx 0.91` only in math form.
- The wording is mostly careful but should be tightened to the spec's target.

**Implementation notes**

- Replace the abstract with a 170-210 word version based on the spec.
- Include both plain-text and LaTeX threshold wording:
  `lambda-star approximately 0.91 ($\lambda^\star \approx 0.91$)`.
- Use "routed into self-regulation" or "routed into homeostatic regulation".
- Avoid unqualified slogans such as "knowing is not enough" unless immediately
  tied to partner-state access lacking a route into self-directed scoring.

**Acceptance checks**

- Abstract is about 170-210 words.
- Abstract contains plain-text `lambda-star approximately 0.91`.
- Abstract explicitly says no partner-welfare reward/objective is used.
- Abstract limits claims about empathy, altruism, consciousness, and moral
  status.

### [x] T03 P0 - Add Explicit Self-Directed Score Equation

**Review driver:** Reviewer 3 needs a precise distinction between homeostatic
coupling and adding partner welfare to the action objective.

**Targets**

- `docs/recips_social_alife2026/main.tex`

**Current audit**

- The prose says the policy is self-directed.
- The explicit `J_{\mathrm{self}}` equation is missing.
- The forbidden contrast with `J_{\mathrm{self}} + \beta U_{\mathrm{partner}}`
  is missing.

**Implementation notes**

- Insert immediately after the homeostat equations and before the condition
  table.
- Define:
  `J_{\mathrm{self}}(a_{0:H-1}) = \sum_{\tau=1}^{H}[w_V V_\tau + w_A A_\tau + w_N N_{s,\tau} + w_B B_\tau]`.
- State fixed social-task weights:
  `w_V=2.0`, `w_A=-1.2`, `w_N=-0.8`, `w_B=-0.4`.
- Add the explicit non-used welfare objective:
  `J_{\mathrm{welfare}} = J_{\mathrm{self}} + \beta U_{\mathrm{partner}}`.
- State that no term of that form is used.
- Clarify that the additive partner term is upstream inside the homeostatic
  state update, not appended to the action objective.
- Add a source-only LaTeX comment:
  `% Reviewer-driven clarity: the key distinction is objective-level partner welfare vs upstream homeostatic routing.`

**Acceptance checks**

- `J_{\mathrm{self}}` appears in Methods.
- `J_{\mathrm{welfare}}` or an equivalent explicit contrast appears nearby.
- The text does not claim coupling is "non-additive"; it says the additive
  operation is routed through regulatory state before scoring.

### [x] T04 P0 - Add Notation Table Or Compact Notation Paragraph

**Review driver:** Reviewer 1 asked for a legend for variables in the homeostat
equations.

**Targets**

- `docs/recips_social_alife2026/main.tex`

**Current audit**

- Only `s`, `\lambda`, and `\hat E^{\mathrm{other}}` are defined in prose.
- `c_b`, `c_m`, `a_t`, `c_h`, `h_t`, `g_e`, `g_p`, `p_t`, and readout symbols
  are not fully defined.

**Implementation notes**

- Add a compact table after the homeostat equations.
- Include at minimum:
  `E^{\mathrm{true}}_t`, `E^{\mathrm{model}}_t`,
  `E^{\mathrm{pred}}_t`, `s`, `c_b`, `c_m a_t`, `c_h h_t`,
  `g_e e_t`, `g_p p_t`, `d^{\mathrm{self}}_t`,
  `d^{\mathrm{other}}_t`, `d^{\mathrm{cpl}}_t`, `\lambda`, `PE_t`,
  `V_t`, `A_t`, and `N_s`.
- If page space is tight, convert to a dense `\paragraph{Notation.}` instead
  of deleting the definitions.

**Acceptance checks**

- Every symbol in the homeostat equation is defined before Results.
- The notation block is compact enough not to crowd out core claims.

### [x] T05 P0 - Define Metrics Precisely

**Review driver:** Reviewer 1 asked for clear definitions of help rate, partner
recovery/rescue, mutual viability, and related readouts.

**Targets**

- `docs/recips_social_alife2026/main.tex`
- Optional source check: `analysis/social_summary.py`

**Current audit**

- A short readouts paragraph exists.
- It names metrics but does not define them precisely.

**Implementation notes**

- Replace the current readouts paragraph with a compact table or precise
  paragraph.
- Define:
  help rate, partner rescue/recovery, mutual viability, rescue latency,
  self-cost of help, final energy.
- Prefer "partner rescue" or "partner recovery/rescue" over bare "recovery".
- If recovery is not restoration to a threshold or setpoint, avoid implying it.
- If the code uses a viability threshold, state it explicitly.

**Acceptance checks**

- A reader can reproduce the interpretation of every headline metric from the
  manuscript text.
- The text does not imply partner restoration to setpoint unless code supports
  that statement.

### [x] T06 P0 - Reframe Conditions As Routing Matrix

**Review driver:** Reviewers 1 and 2 were confused because the four direct-state
conditions collapse into two behavioral pairs.

**Targets**

- `docs/recips_social_alife2026/main.tex`

**Current audit**

- The prose already partially frames the condition table as access control.
- Table 1 still uses compressed condition names and columns
  `Partner access` / `Coupling`.
- Caption does not explicitly say "routing matrix".

**Implementation notes**

- Revise Table 1 columns to:
  `Condition`, `Partner-state access`, `Route into self-regulation`.
- Use full condition names:
  `social_none`, `social_cognitive_direct`,
  `social_affective_direct`, `social_full_direct`.
- State that these are not four rich social-cognitive models.
- Add prose that the expected collapse into two behavioral pairs is diagnostic.
- Ensure Results headings use "Mere partner-state access" and "routing
  dissociation".

**Acceptance checks**

- Table caption explicitly calls the table a routing matrix.
- The text makes clear that the cognitive-direct row is an access-control
  condition.
- The two-pair equality is framed as expected and diagnostic.

### [x] T07 P0 - Clarify Deterministic Seed Reruns

**Review driver:** Reviewer 2 found 64 seed reruns confusing and potentially
statistical-looking.

**Targets**

- `docs/recips_social_alife2026/main.tex`
- `experiments/viz_utils/social_paper_figures.py`

**Current audit**

- Manuscript already says the validated setup is deterministic.
- It still uses "rerun over 64 nominal seeds" wording in a few places.
- Figure 1B y-axis is currently `mean value` in the figure script.

**Implementation notes**

- Use wording from the spec:
  "The validated tasks are deterministic. We nevertheless reran each
  paper-profile condition under 64 nominal seed settings to check that no
  hidden stochastic dependence remained."
- Avoid "trials", unqualified "replications", confidence intervals, or
  stochastic-estimate language.
- Change Figure 1B y-axis to `condition value` or `deterministic value`.

**Acceptance checks**

- Manuscript says values are exact condition-level outcomes rather than
  statistical estimates.
- `rg -n "mean value" experiments/viz_utils/social_paper_figures.py docs/recips_social_alife2026`
  no longer finds the Figure 1B label after figure script update.

### [x] T08 P1 - Add Score-Decomposition Analysis Or Explicit Fallback

**Review driver:** Reviewer 3 asked which internal factors are responsible for
the choice flip.

**Targets**

- `analysis/social_exact_solver.py`
- `core/driver/active_perception.py`
- `docs/recips_social_alife2026/main.tex`
- Optional generated table under `results/social-paper-paper/` or paper
  directory if following existing artifact style.

**Current audit**

- `analysis/social_exact_solver.py` returns total action scores only.
- `core/driver/active_perception.py` already exposes
  `score_internal_components`.
- Existing FoodShareToy episode logs contain valence, arousal, `Ns`, and body
  budget fields, but not a ready-made weighted contribution table.

**Implementation notes**

- First determine whether weighted component contributions can be computed from
  existing forward-model code without changing experimental results.
- Preferred output columns:
  `case`, `lambda`, `w_V Delta V`, `w_A Delta A`, `w_N Delta N_s`,
  `w_B Delta B`, `Delta J`, `choice`.
- Include at least:
  uncoupled default, above-switch coupled case, coupling-off lesion if
  available.
- If component extraction creates too much camera-ready risk, do not invent a
  table. Add one precise prose sentence explaining that the switch is through
  the actor's predicted internal variables under the self-directed scorer, and
  leave component logging as a deferred implementation note in this tracker.

**Acceptance checks**

- Any numeric decomposition values are generated from code, not hand-entered.
- If no table is added, the manuscript still explicitly explains the mechanism
  and does not pretend a decomposition was performed.

### [x] T09 P1 - Strengthen Yoshida & Man Comparison

**Review driver:** Reviewer 1 requested a more substantial comparison.

**Targets**

- `docs/recips_social_alife2026/main.tex`
- `docs/recips_social_alife2026/refs.bib`

**Current audit**

- A comparison paragraph exists in Discussion.
- `YoshidaMan2025HomeostaticCoupling` looks substantially complete in
  `refs.bib`.

**Implementation notes**

- Strengthen the paragraph or add a compact comparison table.
- Make the tradeoff explicit:
  Yoshida & Man provide richer learned multi-agent systems; this paper provides
  a smaller hand-specified controller with exact solver, matched routing
  controls, transparent rollout inspection, and causal lesions.
- Do not claim a stronger behavioral benchmark than Yoshida & Man.

**Acceptance checks**

- The comparison says the present contribution is a minimal executable
  decomposition, not a new empirical discovery about prosociality.
- Reference metadata is complete enough for camera-ready bibliography.

### [x] T10 P1 - Add Perceptual-Crossing Contrast

**Review driver:** Reviewer 2 asked for engagement with perceptual crossing and
minimal social interaction.

**Targets**

- `docs/recips_social_alife2026/main.tex`
- `docs/recips_social_alife2026/refs.bib`

**Current audit**

- No perceptual-crossing contrast appears in the manuscript.
- No perceptual-crossing references are present in `refs.bib`.

**Implementation notes**

- Add a short contrast in Introduction or Discussion.
- State that perceptual-crossing studies ask how reciprocal sensorimotor
  contingencies can constitute social encounter or agency detection.
- State that this paper instead fixes an asymmetric sharing/rescue setting and
  asks a regulatory-routing question.
- Add only verified canonical references. Candidate families to verify:
  Auvray, Lenay, and Stewart; Froese, Iizuka, and Ikegami.
- Do not invent a Severino citation.

**Acceptance checks**

- Manuscript contains a short, accurate perceptual-crossing contrast.
- All added BibTeX entries are verified and complete.

### [x] T11 P1 - Replace Ambiguous Routing Language

**Review driver:** The phrase "dynamically affected" was too broad and could be
misread.

**Targets**

- `docs/recips_social_alife2026/main.tex`

**Current audit**

- Discussion still contains "being dynamically affected".
- "Knowing" appears in the same conceptual contrast.

**Implementation notes**

- Replace with precise language such as:
  "routed into self-regulation",
  "routed into homeostatic regulation",
  "perturbs endogenous homeostatic error before rollout",
  "changes the actor's own predicted internal trajectory".
- Keep any use of empathy/consciousness as explicit non-claims only.

**Acceptance checks**

- `rg -n "dynamically affected|dynamically effected" docs/recips_social_alife2026/main.tex`
  returns no matches.
- The Discussion distinguishes access to another's need from regulatory
  integration of that need.

### [x] T12 P1 - Polish Figures And Captions

**Review driver:** Figure labels should match deterministic and routing framing.

**Targets**

- `experiments/viz_utils/social_paper_figures.py`
- `docs/recips_social_alife2026/fig_summary.pdf`
- `docs/recips_social_alife2026/fig_summary.png`
- `docs/recips_social_alife2026/main.tex`

**Current audit**

- Figure source labels Panel B as `mean value`.
- Manuscript includes `fig_summary.pdf` directly from the paper directory.
- README points to an older `figures/fig_summary.pdf` path that does not match
  current paper source usage.

**Implementation notes**

- Change Panel B y-axis to `condition value` or `deterministic value`.
- Regenerate `fig_summary.pdf` and `fig_summary.png` into the path used by
  `main.tex`.
- Update Figure 1 caption to mention deterministic corridor outcomes and, if
  true for the graphic, that filled markers denote helping present.
- Keep Panel A threshold at `\lambda^\star \approx 0.91`.

**Acceptance checks**

- Regenerated figure files are present at the paths used by `main.tex`.
- Figure 1B no longer says `mean value`.
- Caption uses "partner rescue" or "partner recovery/rescue" consistently with
  metric definitions.

### [x] T13 P1 - Complete And Normalize References

**Review driver:** Camera-ready instructions require complete references.

**Targets**

- `docs/recips_social_alife2026/refs.bib`

**Current audit**

- `Sanyal2026ReCoNIpsundrum`, `YoshidaMan2025HomeostaticCoupling`, and
  `ButlinEtAl2025Indicators` are present and mostly complete.
- Perceptual-crossing references are missing.

**Implementation notes**

- Verify and complete all existing references.
- Add perceptual-crossing references only after verification.
- Preserve capitalization where needed, for example `{AI}`.
- Avoid bare author/title-only entries.

**Acceptance checks**

- Bibliography compiles cleanly.
- No placeholder or incomplete references remain.
- Added references are actually cited in the manuscript.

### [x] T14 P0 - Add Acknowledgements And AI-Use Disclosure

**Review driver:** Camera-ready / transparency requirements.

**Targets**

- `docs/recips_social_alife2026/main.tex`

**Current audit**

- No acknowledgements section is present.
- No AI-use disclosure is present.

**Implementation notes**

- Add `\section*{Acknowledgements}` after Conclusion and before references, or
  wherever the final ALIFE template expects it.
- Include reviewer thanks.
- Include generative-AI disclosure from the spec, adapted only for factual
  accuracy.
- Keep the statement that the author reviewed, ran, validated, and takes
  responsibility for all code, experiments, figures, and manuscript claims.

**Acceptance checks**

- `rg -n "Acknowledgements|generative-AI|OpenAI|Codex" docs/recips_social_alife2026/main.tex`
  finds the section and disclosure.

### [x] T15 P2 - Final Tone And Claim Pass

**Review driver:** Preserve the clean minimal intervention framing from
Reviewer 4 and the meta-review.

**Targets**

- `docs/recips_social_alife2026/main.tex`

**Implementation notes**

- Remove repetitive or inflated phrasing after all required additions are in.
- Keep the manuscript punchy but rigorous.
- Ensure no section implies a rich taxonomy of social inference.
- Ensure no section implies direct partner welfare is in the reward/objective.

**Acceptance checks**

- The paper's final stated contribution is a minimal ALife mechanism
  decomposition.
- The line "Partner-state access is not the active ingredient; regulatory
  routing is" feels supported by Methods, Results, and Discussion.

## Final Validation Checklist

Run these after all manuscript/figure edits are complete:

- [x] Compile `docs/recips_social_alife2026/main.tex` successfully.
- [x] Inspect the generated PDF for table/figure placement and page pressure.
- [x] No stale pre-acceptance status:
  `rg -n "Submission type|Under review|under review" docs/recips_social_alife2026/main.tex`
- [x] No ambiguous routing phrase:
  `rg -n "dynamically affected|dynamically effected" docs/recips_social_alife2026/main.tex`
- [x] Exact threshold is consistently `\lambda^\star \approx 0.91`.
- [x] Any `0.95` mention is labeled as a tested/paper-profile setting.
- [x] Abstract is under about 210 words and includes plain-text threshold.
- [x] Methods include self-directed score equation and welfare-objective
  contrast.
- [x] Notation and metric definitions are present.
- [x] Conditions are framed as a routing matrix.
- [x] Deterministic reruns are described as invariance checks, not statistics.
- [x] Yoshida & Man comparison is strengthened.
- [x] Perceptual-crossing contrast and verified references are added.
- [x] Acknowledgements and AI-use disclosure are present.
- [x] Figure 1B no longer says `mean value`.
- [x] Empathy, consciousness, and moral status appear only as explicit
  non-claims.

## Deferred Or Conditional Items

- Score-decomposition table is conditional on low-risk extraction from existing
  code. If extraction is not straightforward, use a prose decomposition and do
  not fabricate values.
- Any new perceptual-crossing citation must be verified before inclusion.
- Proceedings-specific status/copyright wording may depend on final ALIFE
  camera-ready instructions.
