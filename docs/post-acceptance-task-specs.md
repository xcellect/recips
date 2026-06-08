The remaining work is mostly **clarity repair**, not new experiments. The reviews point to four concrete needs: notation/metric definitions and Yoshida–Man comparison from Reviewer 1; shorter abstract, clearer equations, perceptual-crossing contrast, deterministic-seed clarity, and two-condition narrative from Reviewer 2; explicit distinction between homeostatic coupling and a partner-welfare objective from Reviewer 3; and preservation of the clean “minimal intervention device” framing from Reviewer 4 and the meta-review.    

Below is a Codex-ready Markdown spec.

````markdown
# ALIFE 2026 / arXiv v2 LaTeX brush-up spec

## Context

We are revising the LaTeX for:

**Prosociality by Coupling, Not Observation: Homeostatic Sharing in an Inspectable Recurrent Artificial Life Agent**

The lambda inconsistency has already been fixed: the exact FoodShareToy switch point should be reported consistently as `\lambda^\star \approx 0.91` for the default state. Do not reintroduce `0.95` as the threshold. If `0.95` appears, it must mean the paper-profile coupled setting, not the exact threshold.

The goal is not to add major new experiments. The paper has been accepted as an ALIFE 2026 talk, but the camera-ready/preprint revision should address review-driven presentation and rigor issues.

Primary objective: make the manuscript impossible to misread as:
1. adding partner welfare directly to the reward/objective;
2. claiming a rich four-model taxonomy of social inference;
3. using deterministic seed reruns as statistical evidence;
4. overclaiming empathy, altruism, or consciousness.

The revised manuscript should instead read as:

> A minimal mechanistic ALife intervention showing that, in this controller, partner-state access is inert unless partner distress is routed into the actor's own homeostatic regulation.

---

## Global constraints

- Preserve the current core claim and figures.
- Do not invent new empirical values.
- Do not add confidence intervals or stochastic statistics for deterministic runs.
- Do not add new experiments unless existing result files already contain the needed numbers.
- Do not overclaim empathy, moral status, consciousness, or human altruism.
- Keep the manuscript within the ALIFE full-paper limit.
- Maintain ALIFE2026 template compatibility.
- Ensure the arXiv HTML abstract remains readable even if LaTeX math is stripped.

---

## High-priority edits

### 1. Shorten and harden the abstract

Reviewer feedback: the abstract was perceived as long/bloated, and the phrase “dynamically affected” was too broad.

Replace the abstract with a tighter version around 170–210 words. Use plain-text numbers in addition to math where critical, because arXiv HTML may strip or fail to render mathematical values.

Suggested abstract:

```tex
Artificial agents can be made to ``help'' through explicit social rewards,
hard-coded prosocial bonuses, or direct access to another agent's state. I
isolate a narrower route: homeostatic coupling. Building on
ReCoN-Ipsundrum, I add a scalar homeostat and a social coupling channel while
keeping action selection self-directed: the planner scores only the actor's
predicted internal state, with no partner-welfare reward. In a one-step
\textsc{FoodShareToy}, an exact solver finds a switch from \textsc{Eat} to
\textsc{Pass} at lambda-star approximately 0.91
($\lambda^\star \approx 0.91$) for the default state. In a multi-step
\textsc{SocialCorridorWorld}, observation-only agents never help, whereas
coupled agents fetch, carry, and pass food to the partner. Sham lesions
preserve helping; coupling-off and shuffled-partner lesions abolish it. A
coupling/load sweep shows that coupling creates a low-load helping regime but
does not guarantee rescue under higher metabolic load. This is not a claim
about empathy, altruism, consciousness, or moral status. It is a minimal ALife
demonstration that, in this controller, partner-state access is behaviorally
inert unless partner distress is routed into self-regulation.
````

Implementation notes:

* Use `lambda-star approximately 0.91` in text so arXiv HTML does not drop the value.
* Replace all uses of “dynamically affected” with “routed into self-regulation,” “routed into homeostatic regulation,” or “perturbs endogenous homeostatic error.”
* Do not say “knowing is not enough” without immediately specifying that this means partner-state access has no route into the self-directed scoring pathway.

---

### 2. Remove or update preprint-status lines

Find and edit the front-matter lines.

Remove:

```tex
Submission type: Full Paper
```

For arXiv v2, replace any “under review” wording with a truthful accepted-preprint line, for example:

```tex
\noindent\textbf{Data/Code:} \url{https://github.com/xcellect/recips}.\\
\textbf{Status:} Accepted as a talk at ALIFE 2026; proceedings camera-ready version forthcoming.
```

For the actual camera-ready proceedings version, remove the `Status:` line unless the template/conference permits it.

Also ensure:

* author name and affiliation are fully present;
* the copyright notice uses the real author name;
* acknowledgements are added;
* the references are complete.

---

### 3. Add an explicit self-directed score equation

Reviewer 3’s main concern is that the manuscript does not make clear enough how homeostatic coupling differs from directly adding partner welfare to the evaluation function. Add this immediately after the current homeostat equations and before the condition table.

Insert something like:

```tex
Candidate action sequences are then scored only through the actor's predicted
internal variables:
\begin{equation}
J_{\mathrm{self}}(a_{0:H-1}) =
\sum_{\tau=1}^{H}
\left[
w_V V_\tau
+ w_A A_\tau
+ w_N N_{s,\tau}
+ w_B B_\tau
\right],
\end{equation}
where $V_\tau$ is predicted valence, $A_\tau$ is predicted arousal,
$N_{s,\tau}$ is recurrent salience, and $B_\tau$ is the nonnegative
body-budget error used by the parent controller. In the social tasks,
the fixed weights are
$w_V=2.0$, $w_A=-1.2$, $w_N=-0.8$, and $w_B=-0.4$.
```

Then add the decisive contrast:

```tex
This is different from adding a partner-welfare objective such as
\begin{equation}
J_{\mathrm{welfare}} =
J_{\mathrm{self}} + \beta U_{\mathrm{partner}} .
\end{equation}
No term of this form is used. Partner state enters only upstream, through
\[
d^{\mathrm{cpl}}_t =
d^{\mathrm{self}}_t + \lambda d^{\mathrm{other}}_t,
\]
after which valence, arousal, prediction error, recurrent salience, and
body-budget error are computed by the ordinary self-regulatory machinery.
Thus the additive operation is inside the homeostatic state update, not a
separate partner-welfare term in the action objective.
```

Do not claim that coupling is “non-additive.” The correct claim is that the additive partner term is routed into the actor’s regulatory state before rollout/scoring rather than appended to the action objective.

---

### 4. Add a notation table

Reviewer 1 explicitly asked for a legend for variables such as `c_b`, `c_m`, `a_t`, `c_h`. Add a compact table after the homeostat equations.

Suggested LaTeX:

```tex
\begin{table}[t]
\centering
\small
\begin{tabular}{ll}
\toprule
Symbol & Meaning \\
\midrule
$E^{\mathrm{true}}_t$ & actual energy/resource state \\
$E^{\mathrm{model}}_t$ & internal estimate of energy \\
$E^{\mathrm{pred}}_t$ & predicted energy after control update \\
$s$ & homeostatic setpoint \\
$c_b$ & basal metabolic cost \\
$c_m a_t$ & movement cost, with $a_t=1$ for movement actions \\
$c_h h_t$ & hazard cost, with $h_t=1$ on hazard contact \\
$g_e e_t$ & energy gain from self-eating \\
$g_p p_t$ & energy gain from receiving/passing food \\
$d^{\mathrm{self}}_t$ & self distress: shortfall from setpoint \\
$d^{\mathrm{other}}_t$ & estimated partner distress \\
$d^{\mathrm{cpl}}_t$ & coupled distress used by the actor \\
$\lambda$ & affective coupling strength \\
$PE_t$ & prediction error \\
$V_t, A_t$ & valence and arousal proxies \\
$N_s$ & recurrent salience/persistence variable \\
\bottomrule
\end{tabular}
\caption{Notation for the social homeostat and self-directed scorer.}
\label{tab:notation}
\end{table}
```

If space is tight, convert this to a compressed inline “Notation.” paragraph rather than deleting it.

---

### 5. Define all metrics precisely

Reviewer 1 asked for more explanation of help rate, partner recovery, mutual viability, and related readouts. Add a compact “Readouts” table or paragraph in `Tasks, lesions, and metrics`.

Suggested table:

```tex
\begin{table}[t]
\centering
\small
\begin{tabular}{ll}
\toprule
Metric & Definition \\
\midrule
Help rate & fraction of episodes with an effective \textsc{Pass} while the partner is distressed \\
Partner rescue/recovery & fraction of episodes in which a transfer raises partner energy after distress \\
Mutual viability & $T^{-1}\sum_t \mathbf{1}[E^{self}_t>\theta \land E^{partner}_t>\theta]$ \\
Rescue latency & first timestep of effective transfer; horizon if none occurs \\
Self-cost of help & actor final-energy drop relative to matched uncoupled baseline \\
Final energy & terminal $E^{\mathrm{true}}$ for actor and partner \\
\bottomrule
\end{tabular}
\caption{Readouts used in the food-sharing and corridor tasks.}
\label{tab:metrics}
\end{table}
```

Important wording:

* Prefer “partner rescue” or “partner recovery/rescue” over just “recovery” unless the code defines recovery as restoration above a particular threshold.
* If the code uses a threshold for recovery, state the threshold explicitly.
* If “recovery” only means “receives a transfer and rises from collapse trajectory,” do not imply restoration to the setpoint.

---

### 6. Reframe Table 1 as a routing matrix, not four rich social models

Reviewers 1 and 2 were confused because the four direct-state conditions collapse into two behavioral pairs. The current design is fine, but the narrative must make this expected.

Revise Table 1 caption and surrounding prose.

Suggested table:

```tex
\begin{table}[t]
\centering
\small
\begin{tabular}{lll}
\toprule
Condition & Partner-state access & Route into self-regulation \\
\midrule
\texttt{social\_none} & no & no, $\lambda=0$ \\
\texttt{social\_cognitive\_direct} & yes & no, $\lambda=0$ \\
\texttt{social\_affective\_direct} & no separate cognitive channel & yes, $\lambda>0$ \\
\texttt{social\_full\_direct} & yes & yes, $\lambda>0$ \\
\bottomrule
\end{tabular}
\caption{Routing matrix for the direct-state experiments. These rows are not
four rich social-cognitive models. They separate partner-state access from
regulatory routing while keeping the controller, planner, and action space
fixed.}
\label{tab:conditions}
\end{table}
```

Add prose immediately before or after the table:

```tex
The four rows form a routing matrix. The cognitive-direct row is an
access-control condition: partner state is available but has no route into the
self-directed score. The affective-direct row tests the opposite intervention:
partner distress is coupled into homeostatic error. Because the current
direct-state implementation gives the full condition no additional information
beyond the coupled channel, the design is expected to collapse into two
behavioral pairs. That collapse is diagnostic rather than disappointing.
```

Also adjust Results headings:

* Replace “Observation alone...” with “Mere partner-state access...”
* Replace “The same dissociation...” with “The same routing dissociation...”

---

### 7. Clarify deterministic reruns

Reviewer 2 found the 64 seed reruns confusing. Replace language that sounds statistical.

Use:

```tex
The validated tasks are deterministic. We nevertheless reran each
paper-profile condition under 64 nominal seed settings to check that no hidden
stochastic dependence remained. Trajectories were invariant, so the reported
values are exact condition-level outcomes rather than statistical estimates.
```

Avoid:

* “trials”
* “replications” unless immediately qualified
* confidence intervals
* “mean” unless it is literally an average over deterministic identical runs and the figure requires that label

In Figure 1B, consider changing y-axis label from `mean value` to `condition value` or `deterministic value`.

---

### 8. Add a score-decomposition analysis if existing data/code supports it

Reviewer 3 asked not merely for condition differences but for which factors are responsible. The best low-scope fix is a small score-decomposition table using the existing exact solver or logged rollout terms.

Codex task:

1. Search the repo/results for exact solver outputs or action-score logs.
2. If contribution terms are available or can be recomputed from existing code without new experiments, add a compact table.
3. Do not fabricate values.
4. If the code only exposes total scores, add a prose decomposition and create an issue/TODO for future logging rather than inventing a table.

Target table:

```tex
\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrrrl}
\toprule
Case & $\lambda$ & $\Delta V$ & $\Delta A$ & $\Delta N_s$ & $\Delta B$ & Choice \\
\midrule
default, uncoupled & 0.00 & ... & ... & ... & ... & \textsc{Eat} \\
default, above switch & 0.95 & ... & ... & ... & ... & \textsc{Pass} \\
coupling-off lesion & 0.95 routed off & ... & ... & ... & ... & \textsc{Eat} \\
\bottomrule
\end{tabular}
\caption{Score decomposition for \textsc{Pass} minus \textsc{Eat} in
\textsc{FoodShareToy}. Entries are weighted contributions to the
self-directed score.}
\label{tab:scoredecomp}
\end{table}
```

If exact weighted contributions are available, use columns:

```tex
$w_V\Delta V$, $w_A\Delta A$, $w_N\Delta N_s$, $w_B\Delta B$, $\Delta J$
```

This is better than raw `\Delta V` etc., because it shows what actually flips the decision.

Add one sentence in Results:

```tex
The switch is driven by the fact that partner distress changes the actor's own
predicted valence/arousal/body-budget trajectory under the ordinary
self-directed scorer; no partner-utility term is added at the action-selection
stage.
```

---

### 9. Strengthen the Yoshida & Man comparison

Reviewer 1 explicitly wanted a more substantial comparison to Yoshida and Man. Add a paragraph or compact table in Discussion.

Suggested text:

```tex
The relation to Yoshida and Man is deliberately asymmetric. Their model
demonstrates the observation-versus-coupling distinction in richer learned
multi-agent systems. The present model does not try to match that adaptive
breadth. Instead, it makes the same distinction inspectable in a smaller
hand-specified controller. The tradeoff is useful: the current system admits an
exact one-step solver, matched routing controls, transparent rollout
inspection, and causal lesions that sever only the coupling channel. Thus the
contribution is not a stronger behavioral benchmark, but a minimal executable
decomposition of the same architectural hypothesis.
```

Optional comparison table:

```tex
\begin{table}[t]
\centering
\small
\begin{tabular}{lll}
\toprule
Dimension & Yoshida \& Man & This paper \\
\midrule
Controller & learned multi-agent system & hand-specified recurrent controller \\
Main contrast & observation vs coupling & access route vs regulatory route \\
Partner behavior & richer interaction & passive partner / controlled rescue \\
Strength & adaptive breadth & exact solver and causal lesions \\
Limitation & less analytically transparent & no learning, passive partners \\
Claim & coupling can support prosociality & minimal decomposition of that route \\
\bottomrule
\end{tabular}
\caption{Relation to the homeostatic-coupling result this paper decomposes.}
\label{tab:yoshida_compare}
\end{table}
```

Also complete the Yoshida & Man BibTeX entry. Do not leave it as only:

```bibtex
Yoshida, N. and Man, K. (2025). Homeostatic coupling for prosocial behavior.
```

Add venue, DOI/arXiv, pages, or preprint metadata, based on the real source.

---

### 10. Add a short perceptual-crossing contrast

Reviewer 2 asked for engagement with perceptual crossing / minimal social interaction. Add a short related-work contrast in the Discussion or Introduction.

Suggested text:

```tex
This work is adjacent to, but distinct from, perceptual-crossing models of
minimal social interaction. Perceptual-crossing studies ask how reciprocal
sensorimotor contingencies can constitute social encounter or agency
detection. The present model fixes interaction to an asymmetric sharing/rescue
setting and asks a different routing question: once partner state is available,
does it influence action as mere observation or by entering the actor's own
regulatory dynamics? The contribution is therefore not a general model of
social interaction, but a causal decomposition of one proposed route to
artificial prosociality.
```

Add BibTeX entries for canonical perceptual-crossing work. Suggested candidates to verify before committing:

* Auvray, Lenay, and Stewart on perceptual crossing / minimalist virtual environment.
* Froese, Iizuka, and Ikegami on embodied social interaction in minimalist virtual reality.
* Any specific Severino et al. paper only if the exact reference can be verified. Do not invent a Severino citation.

---

### 11. Replace “dynamically affected” throughout

Search:

```bash
grep -R "dynamically affected\|dynamically effected\|knowing" -n *.tex
```

Replace with precise routing language.

Good replacements:

* “routed into self-regulation”
* “routed into homeostatic regulation”
* “perturbs endogenous homeostatic error before rollout”
* “changes the actor’s own predicted internal trajectory”

Avoid:

* “dynamically affected”
* “really cares”
* “empathy”
* “altruism” except in explicitly limited philosophical framing

Suggested Discussion revision:

```tex
The central result is simple: in this controller, another agent's need changes
behavior only when it changes the actor's own regulatory state. Direct
partner-state representation is not enough. That distinction matters
philosophically because it separates two interpretations that are often blurred
in synthetic social behavior: access to another's need and regulatory
integration of that need.
```

---

### 12. Polish figure labels and captions

Figure 1:

* Panel A: keep `$\lambda^\star \approx 0.91$`.
* Panel B y-axis: change `mean value` to `condition value` or `deterministic value`.
* Panel B legend: consider `help`, `partner rescue`, `mutual viability`.
* Panel C title: keep punchy, but maybe “Coupling lesions abolish helping” rather than “Causal lesions abolish helping.”
* Panel D: caption should say “filled markers denote helping present” if markers encode that.

Caption revision:

```tex
Figure 1: Mechanism-linked summary. A: exact one-step switch in
\textsc{FoodShareToy}, computed using the same forward model and
self-directed scorer as the policy. B: deterministic corridor outcomes by
routing condition. Partner-state access without coupling leaves behavior
unchanged; coupling flips helping and partner rescue from 0 to 1. C: sham
lesions preserve helping, while coupling-off and shuffled-partner lesions
abolish it in both tasks. D: coupling/load sweep. Filled markers indicate
runs with helping. Under low load, helping appears for $\lambda \ge 0.25$;
under medium and high load, no tested coupling value rescues the partner
within horizon.
```

Figure 2:

* Caption is strong. Keep the sentence that frames deterministic runs as representative rather than cherry-picked.
* Ensure visual labels use the same names as text: `self-only`, `coupled`, `partner rescue`, etc.

---

### 13. Complete and normalize references

Fix incomplete references:

* `Sanyal (2026)` must include title, arXiv ID/URL or venue.
* `Yoshida and Man (2025)` must include full title, venue/preprint info, DOI/arXiv if available.
* Add perceptual-crossing references after verification.
* Ensure `Butlin et al. (2025)` has full journal metadata if available.
* Ensure “AI” capitalization is correct in titles, depending on bibliography style.

Do not leave bare references with only author/title.

---

### 14. Add acknowledgements and AI-use disclosure

For the camera-ready version, add an acknowledgements section after the conclusion and before references, or wherever the ALIFE template expects it.

Suggested text:

```tex
\section*{Acknowledgements}
The author thanks the ALIFE 2026 reviewers for constructive comments. The
author used generative-AI tools, including OpenAI GPT/Codex systems, for
code-generation assistance, analysis-script support, and editorial suggestions.
The author reviewed, ran, validated, and takes responsibility for all code,
experiments, figures, and manuscript claims.
```

For arXiv v2, this disclosure is also useful and transparent.

---

### 15. Add a short “Reviewer-driven clarity” comment in source only

Add a LaTeX comment near the mechanism section for future maintenance:

```tex
% Reviewer-driven clarity: the key distinction is not "additive vs non-additive".
% It is objective-level partner welfare vs upstream homeostatic routing.
% Do not remove the score equation or routing explanation.
```

Do not render this comment in the PDF.

---

## Acceptance checks

After edits, compile and inspect the PDF.

Required checks:

* [ ] No `Submission type: Full Paper` line remains in arXiv/camera-ready version.
* [ ] `\lambda^\star \approx 0.91` appears consistently as the exact FoodShareToy switch.
* [ ] Any `0.95` mention is clearly labeled as a tested/paper-profile coupled parameter, not the threshold.
* [ ] Abstract is under ~210 words.
* [ ] Abstract includes plain-text critical values, e.g. “lambda-star approximately 0.91.”
* [ ] The self-directed score equation appears in Methods.
* [ ] The contrast with `J_self + beta U_partner` appears explicitly.
* [ ] Notation table or compact notation paragraph is present.
* [ ] Metric definitions are present.
* [ ] Four conditions are framed as a routing matrix.
* [ ] Deterministic seed reruns are described as reproducibility checks, not stochastic estimation.
* [ ] Yoshida & Man comparison is strengthened.
* [ ] Perceptual-crossing contrast paragraph is added.
* [ ] References are complete.
* [ ] AI-use acknowledgement is present for camera-ready/arXiv v2.
* [ ] Figure 1B y-axis no longer says `mean value` unless justified.
* [ ] No overclaiming language: empathy/consciousness/moral status are only mentioned as things not claimed.

---

## Final tone target

Punchy but rigorous. The final manuscript should make this sentence feel earned:

> Partner-state access is not the active ingredient; regulatory routing is.

```

A few specific notes behind the spec: the camera-ready instructions explicitly require reviewer comments to be considered, the submission-type line removed, references checked, copyright notice fixed, and AI-use disclosed where applicable. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5} Also, arXiv’s current HTML rendering can drop crucial math/numbers in the abstract, so duplicating critical values in plain text is not cosmetic; it preserves the claim for readers using the HTML view. :contentReference[oaicite:6]{index=6}
```
