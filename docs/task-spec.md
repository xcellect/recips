Here is the implementation brief I would hand to Codex. It stays inside the lane your current paper already names as future work: move from scalar recurrence to a structured latent space, compare against other formal theories, and keep the work inspectable, lesionable, and explicit about non-claims. The repo already has clean extension points in `core/`, `experiments/`, `analysis/`, and `tests/`, and the ALife special session is explicitly about what computational experiments can contribute to philosophical inquiry, so the code should optimize for mechanism-versus-assay dissociations rather than raw benchmark score. ([arXiv][1])

Paste this into Codex as the top-level task brief:

```text
Extend recips additively with one main model family and one comparator:

MAIN MODEL
- structured fast latent z_t
- slow perspective latent p_t
- affect-coupled local plasticity on sensory-to-latent weights
- keep existing affect/body-budget module compatible with current code

COMPARATOR
- minimal GW-lite broadcast bottleneck
- explicit modality selector + low-capacity workspace state
- no slow perspective latent
- no local plasticity by default

DELIVERABLE
- theory-adversarial assay suite
- causal lesions
- latent-state visualizations
- claims-as-code analysis
- no consciousness claims

NON-NEGOTIABLES
1. preserve existing models recon / humphrey / humphrey_barrett
2. preserve existing external interfaces used by gridworld_exp.py, corridor_exp.py, evaluation_harness.py
3. preserve visible state keys Ns, Ne, Ni, Nv, Na, valence, arousal, bb_model, etc.
4. every new model must have a pure non-mutating one-step dynamics function used both online and by forward prediction
5. do not rewrite the repo in torch; keep the first implementation numpy-only
6. keep parameter counts of perspective_plastic and gw_lite within 20%
7. all primary claims must have lesion tests and bootstrap CIs
8. use additive changes; do not break old tests

IMPLEMENT THESE MODEL IDS
- perspective
- perspective_plastic
- gw_lite

PRIMARY ASSAYS
- hysteresis probe assay
- contextual fork / perceptual aliasing assay
- multimodal conflict / corruption assay

CONTINUITY CHECKS
- rerun a narrow subset of old assays (pain-tail + familiarity-controlled route choice)

REQUIRED OUTPUTS
- runnable quick profile and paper profile
- per-step logs with flattened latent columns
- summary CSVs
- publication-quality PNG/PDF figures
- updated dissociation table
- updated paper claims
- tests for determinism, forward-model alignment, lesions, and history dependence
```

## The target scientific claim

The paper should make one bounded claim: **different minimal internal organizations support different families of subjectivity-like indicators, and those differences can be separated by causal lesions and theory-adversarial assays**. That is aligned with your current paper’s indicator-based, non-decisive framing and with its stated future-work direction toward structured latent state plus comparisons to other theories. It also fits the ALife session’s “computational experiment as philosophy” framing. ([arXiv][1])

The paper should **not** claim consciousness, GNWT realization, or full predictive processing. Keep the same “minimal correspondence, explicit gaps” discipline your current Table 1 already uses. ([arXiv][1])

## Hard constraints Codex should obey

Do not replace the outer shell. The safest path is to keep the current ReCoN/Stage-D shell and swap only the internal dynamical kernel, because the current stack expects `build_model_network()` to return `(builder, net)`, `EvalAgent` reads `net._ipsundrum_state`, the policy reads `agent.b.params` and `agent.b.affect`, and `gridworld_exp.py` / `corridor_exp.py` choose forward models through `select_forward_model()`. The current policy layer is already abstracted around `EnvAdapter`, `compute_I_affect`, and a pluggable forward model, so assay-specific sensory manipulations should mostly happen at the adapter/sensory layer rather than by rewriting the base environments. ([GitHub][2])

Also preserve the current discipline of **pure step dynamics + forward-model alignment**. Right now `ipsundrum_step()` is explicitly pure and non-mutating, and the repo already checks forward-model alignment against the online update. The new models should copy that pattern exactly. ([GitHub][3])

## Model family A: structured perspective latent

This model is justified because Pae’s 2026 papers operationalize perspective with a slowly evolving global latent that modulates fast dynamics and yields hysteresis and history-dependent perceptual reorganization, while Kagan et al. give you a principled reason to add adaptive rule change rather than stay at fixed-rule recurrence. The affect-coupled local plasticity piece is also a good fit with the neuromodulation literature’s emphasis on multi-scale modulation rather than a single global reward signal. ([arXiv][4])

Use this state:

```python
state = {
    "model_family": "perspective",
    "z": np.ndarray,          # shape (z_dim,)
    "p": np.ndarray,          # shape (p_dim,)
    "delta_w_in": np.ndarray, # shape (z_dim, obs_dim), zero init unless plastic
    "Ns": float,
    "motor": float,
    "efference": float,
    "precision_eff": float,   # keep scalar for policy compatibility
    "precision_vec": np.ndarray,  # shape (4,), for logging and channel gating
    "g_eff": float,
    "bb_true": float,
    "bb_model": float,
    "bb_pred": float,
    "pe": float,
    "valence": float,
    "arousal": float,
    "plasticity_mod": float,
    "plasticity_open": float,
    "lesion_integrator": False,
    "lesion_feedback": False,
    "lesion_affect": False,
    "lesion_perspective": False,
    "lesion_plasticity": False,
}
```

Use this fixed observation order everywhere:

```python
obs_vec = np.array([
    I_total,      # signed
    I_touch,      # signed
    I_smell,      # signed
    I_vision,     # signed
    eff_prev,     # [0,1]
    1.0,          # bias
], dtype=float)
```

Use this kernel:

```text
# sensory channel gating (4 exteroceptive channels only)
pi_t = clip(pi0 + B_p @ p_{t-1} + b_ar * arousal_{t-1}, pi_min, pi_max)

obs_gated = obs_vec.copy()
obs_gated[:4] *= pi_t

z_prop = tanh((W_in0 + ΔW_t) @ obs_gated + W_rec @ z_{t-1} + W_pz @ p_{t-1})

z_t = z_prop                                      if lesion_integrator
      (1-α_z) z_{t-1} + α_z z_prop               otherwise

p_prop = tanh(W_zp @ z_t + W_pp @ p_{t-1})

p_t = 0                                          if lesion_perspective
      (1-α_p) p_{t-1} + α_p p_prop               otherwise
```

Keep the timescale separation explicit:

* `z_dim = 8`
* `p_dim = 4`
* `α_z ≈ 0.40–0.55`
* `α_p ≈ 0.03–0.08`
* `α_p << α_z` is a required invariant, not a tuning accident.

Readouts must remain policy-compatible:

```text
Ns_t      = clamp01(sigmoid(w_ns · z_t + b_ns))
motor_t   = clamp01(sigmoid(w_m  · z_t + b_m))
Ne_t      = d_e * Ne_{t-1} + (1-d_e) * abs(motor_t)
```

Keep the current affect/body-budget module semantically unchanged. The new model may read affect, but it should not redefine what `valence`, `arousal`, or `bb_model` mean, because the existing policy scores those quantities directly. ([GitHub][5])

### Plasticity rule

Plasticity is required only for `perspective_plastic`, and it must be local and bounded:

```text
mu_t   = clip(c_pe * abs(pe_t) +
              c_ar * arousal_t +
              c_bb * abs(bb_model_t - setpoint), 0, mu_max)

open_t = sigmoid(b_open + v_open · p_t)

o_tilde = obs_gated.copy()
o_tilde[-1] = 0.0              # no plasticity on bias column

ΔW_{t+1} = ΔW_t                                        if lesion_plasticity
           clip((1-λ_w) * ΔW_t +
                η_w * mu_t * open_t * outer(z_t, o_tilde),
                -Δ_max, Δ_max)                         otherwise
```

Required properties:

* `perspective`: same kernel, but `ΔW` frozen at zero.
* `perspective_plastic`: same kernel, with the plastic update enabled.
* `lesion_feedback`: zero both `W_rec @ z_{t-1}` and `W_pz @ p_{t-1}`.
* `lesion_integrator`: remove carryover terms.
* `lesion_perspective`: clamp `p_t = 0` and `open_t = sigmoid(b_open)`.
* `lesion_plasticity`: freeze `ΔW`.

Do **not** train these weights with SGD. Initialize fixed matrices from an architecture seed, then treat architecture seed as an experimental factor.

## Model family B: GW-lite comparator

This comparator should be a real broadcast bottleneck, not “another latent with a different name.” The point is to give the paper a clean rival mechanism. Recent GW work gives you a good computational template: low-capacity multimodal integration plus explicit top-down selection, especially under corruption. The Cogitate Nature paper is the methodological model here: build assays that discriminate theories rather than letting one model dominate every axis. ([arXiv][6])

Use this state:

```python
state = {
    "model_family": "gw_lite",
    "workspace": np.ndarray,        # shape (w_dim,)
    "selector_logits": np.ndarray,  # shape (n_modalities,)
    "selector_weights": np.ndarray, # shape (n_modalities,)
    "Ns": float,
    "motor": float,
    "efference": float,
    "precision_eff": float,
    "g_eff": float,
    "bb_true": float,
    "bb_model": float,
    "bb_pred": float,
    "pe": float,
    "valence": float,
    "arousal": float,
    "lesion_selector": False,
    "lesion_workspace": False,
    "lesion_affect": False,
}
```

Use four modalities: `touch`, `smell`, `vision`, `efference`.

Kernel:

```text
x_touch = [I_touch, sign(I_touch)]
x_smell = [I_smell, sign(I_smell)]
x_vision = [I_vision, sign(I_vision)]
x_eff = [Ne_{t-1}, 1.0]

e_m,t = tanh(E_m @ x_m)

sal_t = [abs(I_touch), abs(I_smell), abs(I_vision), Ne_{t-1}, arousal_t]

logits_t = b_sel + U_sal @ sal_t + U_ctx @ workspace_{t-1}

a_t = uniform                                   if lesion_selector
      softmax(logits_t / tau_sel)               otherwise

broadcast_t = Σ_m a_t[m] * e_m,t

w_prop = tanh(W_ww @ workspace_{t-1} + broadcast_t)

workspace_t = broadcast_t                       if lesion_workspace
              (1-α_w) workspace_{t-1} + α_w w_prop  otherwise

Ns_t    = clamp01(sigmoid(r_ns · workspace_t + b_ns))
motor_t = clamp01(sigmoid(r_m  · workspace_t + b_m))
Ne_t    = d_e * Ne_{t-1} + (1-d_e) * abs(motor_t)
```

Rules:

* same affect/body-budget update as the perspective model
* no slow perspective latent
* no local plasticity
* parameter count within 20% of `perspective_plastic`
* export `selector_weights_*` per step for plots

## Assays

Your current paper already has navigation, familiarity-controlled novelty competition, exploratory play, pain-tail, and causal lesions. Keep only a narrow continuity subset from that suite and put the new page budget into three primary dissociation assays. The session papers are short, so you want a compact “continuity + new double dissociation” story, not a giant kitchen-sink benchmark section. ([arXiv][1])

### 1) Hysteresis probe assay

This is the signature assay for the perspective model.

Implement a deterministic observation-stream probe, not just an environment rollout.

Protocol:

* choose 4 canonical observation motifs derived from existing gridworld/corridor sensors
* ramp a corruption / cue-gain parameter `λ` from `0 → 1 → 0`
* feed the exact same motif schedule on the up-ramp and down-ramp
* run pure step updates without action choice
* compute hysteresis in full latent space, not just on a plotted axis

Primary metric:

```text
H = mean_t || φ_up(λ_t) - φ_down(λ_t) ||_2
```

with `φ = p_t` for perspective models and `φ = workspace_t` for GW-lite.

Secondary metrics:

* residue after return: `||p_post_return - p_pre||`
* growth-then-stabilization of `plasticity_open` and `||ΔW||_F`

Predictions:

* `perspective > humphrey_barrett > recon` on `H`
* `perspective_plastic > perspective` on residue / growth-then-stabilization
* `gw_lite` may show short-memory effects but should lag on `H`

### 2) Contextual fork / perceptual aliasing assay

Implement one new corridor-like environment: `ContextForkCorridor`.

Protocol:

* early cue sets hidden context `A` or `B`
* after a delay, the agent reaches a fork
* immediate local observation at the fork must be matched across contexts
* the correct branch depends on the earlier context
* test several delay lengths, for example `6, 12, 18`

Primary metrics:

* success rate at the fork versus delay
* latent separation at the matched fork observation:
  `R = ||mean(φ | ctx=A, fork) - mean(φ | ctx=B, fork)||_2`

Predictions:

* `perspective_plastic` best at long delays
* `perspective` next
* `gw_lite` may be competitive at short delays but should decay faster
* `humphrey_barrett` may retain brief context but should underperform at long delay

### 3) Multimodal conflict / corruption assay

Do this mostly as an adapter/wrapper problem, not a new environment.

Protocol:

* take existing gridworld and corridor tasks
* corrupt one modality in blocks: sign-flip, dropout, or structured noise
* include at least one conflict regime where vision and smell disagree
* log GW selector weights every step

Primary metric:

```text
robustness = success_rate - κ * mean_hazard_contacts
```

Secondary:

* selector entropy
* probability mass on the uncorrupted modality
* behavior degradation relative to clean condition

Predictions:

* `gw_lite` should win here
* `perspective_plastic` may still adapt somewhat
* `humphrey_barrett` should be less robust under structured corruption

### Continuity checks

Keep just two old checks:

* pain-tail
* familiarity-controlled scenic-vs-dull route choice

The purpose is not to re-run the whole prior paper. The purpose is to show the new models do not regress on the core signatures the current paper already established. ([arXiv][1])

## Statistics and rigor rules

Your current paper notes that some intervals remained wide and explicitly recommends more seed-level reporting and robustness checks. Build that discipline in from the start. ([arXiv][1])

Use these rules:

* two seed axes: `arch_seed` for fixed weight initialization, `env_seed` for task randomness
* `PROFILE=quick`: `4 arch_seeds × 4 env_seeds`
* `PROFILE=paper`: `4 arch_seeds × 12 env_seeds`
* no tuning on the primary assays
* tune only on a calibration suite:

  * bounded dynamics
  * no NaNs
  * no hard saturation >95% of steps
  * basic navigation competence floor
* one primary metric per assay; everything else exploratory
* report mean difference, bootstrap 95% CI, and Cliff’s delta
* compute representation distances in the **full latent space**
* use PCA/2D plots only for visualization, never as the primary statistic
* parameter-count match `perspective_plastic` and `gw_lite` within 20%

## File-level patch plan

The repo already has the right places for this: core dynamics in `core/`, assay code in `experiments/`, claims and tables in `analysis/`, and extensive regression tests in `tests/`. ([GitHub][7])

Add these new files:

```text
core/perspective_model.py
core/workspace_model.py
core/driver/perspective_dynamics.py
core/driver/perspective_forward.py
core/driver/workspace_dynamics.py
core/driver/workspace_forward.py
experiments/hysteresis_probe_assay.py
experiments/context_fork_assay.py
experiments/multimodal_conflict_assay.py
experiments/latent_viz.py
tests/test_perspective_dynamics.py
tests/test_workspace_dynamics.py
tests/test_history_dependence.py
tests/test_parameter_count_match.py
tests/test_new_model_lesions.py
```

Patch these existing files:

```text
experiments/evaluation_harness.py
experiments/gridworld_exp.py
experiments/corridor_exp.py
analysis/build_dissociation_table.py
analysis/paper_claims.py
run_experiments.sh
utils/model_naming.py
```

Implementation notes:

* in `gridworld_exp.py` and `corridor_exp.py`, extend `select_forward_model()` so new model ids route to the correct pure forward function
* in `evaluation_harness.py`, extend `build_model_network()` with new ids:

  * `perspective`
  * `perspective_plastic`
  * `gw_lite`
* keep builder objects exposing `.params`, `.affect`, and optional `.score_weights`
* keep network shells exposing `start_root(True)` and `net._ipsundrum_state`
* add a flattening utility so arrays become log columns like `z_0`, `p_0`, `workspace_0`, `sel_touch`

## Tests Codex must add

1. `perspective_step` is deterministic and non-mutating.
2. `workspace_step` is deterministic and non-mutating.
3. forward prediction matches online update for both new model families.
4. `α_p << α_z` produces slower average change in `p` than `z`.
5. plasticity lesion freezes `ΔW`.
6. perspective lesion kills hysteresis more than it kills corruption robustness.
7. selector lesion kills corruption robustness in `gw_lite`.
8. same observation after different histories yields larger latent separation in perspective models than in scalar baselines.
9. parameter counts are matched within tolerance.
10. old scalar-model tests still pass.

## Claims-as-code

Because the repo already has `analysis/paper_claims.py` and dissociation-table tooling, codify the new claims explicitly instead of leaving them in notebooks. ([GitHub][8])

Add boolean claim checks like:

* `claim_hysteresis_perspective_gt_scalar`
* `claim_plasticity_residue_gt_no_plastic`
* `claim_gw_conflict_robustness_gt_perspective`
* `claim_selector_lesion_selective`
* `claim_perspective_lesion_selective`
* `claim_continuity_no_regression_pain_tail`
* `claim_continuity_no_regression_familiarity_choice`

Each claim function should emit:

* estimate
* CI
* effect size
* direction-consistency across `arch_seed`
* pass/fail boolean

## Definition of done

Codex is done only when all of this is true:

* `PROFILE=quick ./run_experiments.sh` finishes end-to-end
* all existing tests and all new tests pass
* new model ids run in both gridworld and corridor code paths
* primary assay figures are generated
* latent logs are exported with flattened columns
* dissociation matrix is generated
* claims-as-code functions run without notebook intervention
* the primary double dissociation is visible:

  * perspective wins hysteresis / context-history assays
  * GW-lite wins corruption/conflict robustness
  * plasticity strengthens residue/history effects
  * lesions are selective

The main architectural reason this plan is strong is that it keeps the exact virtues of the current codebase — inspectability, minimal correspondence, causal lesionability, and shared policy scaffolding — while directly instantiating the paper’s own future-work line and upgrading the venue fit from “toy architecture with markers” to “synthetic agents as discriminating experiments on rival minimal mechanisms.” ([arXiv][1])

The one thing I would tell Codex not to improvise on is theory language: call the new model a **slow perspective latent** and the comparator **GW-lite**, keep the Butlin-style indicator framing, and keep every major result tied to a lesion. ([Cell][9])

[1]: https://arxiv.org/html/2602.23232v2 "https://arxiv.org/html/2602.23232v2"
[2]: https://github.com/xcellect/recips/blob/main/experiments/evaluation_harness.py "https://github.com/xcellect/recips/blob/main/experiments/evaluation_harness.py"
[3]: https://github.com/xcellect/recips/blob/main/core/driver/ipsundrum_dynamics.py "https://github.com/xcellect/recips/blob/main/core/driver/ipsundrum_dynamics.py"
[4]: https://arxiv.org/abs/2602.02902 "https://arxiv.org/abs/2602.02902"
[5]: https://raw.githubusercontent.com/xcellect/recips/main/core/driver/active_perception.py "https://raw.githubusercontent.com/xcellect/recips/main/core/driver/active_perception.py"
[6]: https://arxiv.org/abs/2502.21142 "https://arxiv.org/abs/2502.21142"
[7]: https://github.com/xcellect/recips "https://github.com/xcellect/recips"
[8]: https://github.com/xcellect/recips/blob/main/analysis/paper_claims.py "https://github.com/xcellect/recips/blob/main/analysis/paper_claims.py"
[9]: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613%2825%2900286-4 "https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613%2825%2900286-4"
