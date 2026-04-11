# Social Homeostatic Coupling Extension

## Overview

This document specifies the two-agent social extension added to the repository. The extension preserves the existing ReCoN/Ipsundrum recurrent-affective controller and planning scaffold, but augments each agent with an explicit homeostatic resource state and a social coupling channel that can route partner distress into self-regulation. The implementation is designed to support paper-ready mechanistic claims rather than reward-shaped benchmark performance.

The core scientific target is the following dissociation:

1. Self-homeostasis alone does not produce helping.
2. Perceiving partner need alone does not produce helping.
3. Helping appears when partner need becomes part of the controller's own homeostatic regulation.
4. Causal lesions of the coupling channel abolish helping.
5. Coupling sweeps reveal regime changes rather than a single monotone improvement curve.

The implementation follows those commitments directly. No external social reward term was added to the planner. No direct `help partner` bonus was introduced. The same affect-capable base controller is reused across social conditions, and the planner continues to score predicted self-state, now after coupling has altered the self homeostat.

## Design Principles

### Construct-validity constraints

The extension was implemented under the following non-negotiable rules.

- No external reward bonus for partner welfare in headline experiments.
- No direct prosocial action bonus in headline experiments.
- Same base controller family across `social_none`, `social_cognitive_direct`, `social_affective_direct`, and `social_full_direct`.
- Scenic/beauty shaping and novelty bonuses disabled in social tasks.
- Partner death does not terminate the actor's episode.
- Lesion and sham-lesion conditions implemented as first-class experimental conditions.
- Internal variables required for causal interpretation logged explicitly.

These constraints are important because the original repository already contained non-social shaping terms and recurrent-affective modulation. The social extension was engineered so the headline claim can be attributed to homeostatic coupling, not to an auxiliary reward specification.

### Engineering strategy

The repository already had a stable split between:

- core recurrent-affective dynamics,
- active-perception planning,
- environment adapters,
- experiment runners,
- analysis utilities,
- tests.

The extension therefore took the least invasive path:

- preserve `core/driver/active_perception.py` as the single action-scoring interface,
- preserve the existing Ipsundrum/ReCoN forward step for non-social tasks,
- add a social forward model that augments the predicted rollout state with partner homeostasis,
- map the explicit homeostat into the existing body-budget variables so the scorer does not need to be rewritten.

## Added Components

### New modules

The following modules were added.

- `core/social_homeostat.py`
- `core/social_forward.py`
- `core/social_model.py`
- `core/envs/social_foodshare.py`
- `core/envs/social_corridor.py`
- `experiments/social_foodshare.py`
- `experiments/social_corridor.py`
- `experiments/social_lesion_assay.py`
- `analysis/social_exact_solver.py`
- `analysis/social_summary.py`
- `analysis/social_claims.py`
- `tests/test_social_homeostat.py`
- `tests/test_social_forward.py`
- `tests/test_social_foodshare.py`
- `tests/test_social_lesions.py`

### Modified modules

The following existing modules were extended rather than forked.

- `core/driver/active_perception.py`
- `core/driver/ipsundrum_forward.py`
- `core/driver/recon_forward.py`
- `core/envs/__init__.py`
- `experiments/evaluation_harness.py`
- `run_experiments.sh`
- `utils/model_naming.py`

## Homeostatic Model

### State representation

Each social agent now carries an explicit physiological resource model:

```python
@dataclass
class HomeostatState:
    energy_true: float
    energy_model: float
    energy_pred: float
    distress_self: float
    distress_other_est: float
    distress_coupled: float
    pe: float
    valence: float
    arousal: float
    alive: bool
```

The distinction between `energy_true`, `energy_model`, and `energy_pred` was introduced for the same reason the original repository distinguished true body budget, predicted body budget, and prediction error: the planner should not act directly on a privileged scalar utility, but on an internal state that evolves under prediction and control.

### Parameters

The shared homeostat parameter block is defined in `core/social_homeostat.py`.

```python
@dataclass
class HomeostatParams:
    setpoint: float = 0.70
    basal_cost: float = 0.01
    move_cost: float = 0.005
    hazard_cost: float = 0.05
    eat_gain: float = 0.25
    pass_gain: float = 0.25
    k_homeo: float = 0.25
    k_pe: float = 0.50
    valence_scale: float = 0.50
    arousal_scale: float = 1.00
    death_threshold: float = 0.05
    death_patience: int = 5
```

The parameterization is intentionally simple. It is sufficient to generate self-preserving and rescue-like behavior while still being interpretable enough for lesion and sweep analyses.

### Social coupling parameters

```python
@dataclass
class SocialCouplingParams:
    lambda_affective: float = 0.0
    observe_partner_internal: bool = False
    observe_partner_expression: bool = False
    use_decoder: bool = False
    expression_noise_std: float = 0.0
    lesion_mode: str = "none"
```

The current implementation ships the direct-state conditions required for the primary result. Decoded-empathy fields are present for a later phase, but the current paper-grade experiments use the direct variants.

### Update equations

The implemented update follows the intended equations closely.

```text
energy_true[t+1] = clip(
    energy_true[t]
    - basal_cost
    - move_cost(action)
    - hazard_cost * hazard_contact
    + eat_gain * self_ate
    + pass_gain * received_food,
    0, 1
)

distress_self  = relu(setpoint - energy_model)
distress_other = relu(setpoint - other_energy_est)

distress_coupled = distress_self + lambda_affective * distress_other

control_u   = -k_homeo * distress_coupled
energy_pred = clip(energy_model + control_u, 0, 1)

pe = energy_true - energy_pred
energy_model[t+1] = clip(energy_model + k_pe * pe, 0, 1)

valence = 1 - clip(distress_coupled / valence_scale, 0, 1)
arousal = clip(arousal_scale * (abs(pe) + distress_coupled), 0, 1)
```

Two implementation details are important.

- Coupling is applied at the level of drive or homeostatic error, not as an added action reward.
- Valence and arousal computed from the coupled homeostat are then routed back into the existing affect-to-loop modulation pathway, preserving the original architecture's recurrent interpretation.

### Compatibility mapping

To keep the existing scorer intact, the explicit homeostat is mapped into the body-budget variables already used by the policy.

```text
bb_true  = energy_true  - setpoint
bb_model = energy_model - setpoint
bb_pred  = energy_pred  - setpoint
bb_err   = distress_coupled
```

This means the planner still scores predicted self-state via valence, arousal, `Ns`, and body-budget error, but those quantities now arise from an explicit and socializable homeostatic process.

## Social Conditions

The following direct-state conditions are implemented.

### `social_none`

- No partner observation.
- No partner expression channel.
- `lambda_affective = 0.0`.

This is the self-homeostasis baseline.

### `social_cognitive_direct`

- Direct partner internal state available.
- `lambda_affective = 0.0`.

This isolates partner-state access without affective coupling.

### `social_affective_direct`

- No explicit observation channel required.
- Partner distress enters self-regulation directly.
- `lambda_affective > 0`.

This is the core affective-coupling condition.

### `social_full_direct`

- Direct partner internal state available.
- `lambda_affective > 0`.

This combines direct social observation with affective coupling.

In the current results, `social_none` and `social_cognitive_direct` remain non-helping, while `social_affective_direct` and `social_full_direct` help reliably.

## Planner Integration

### Action scoring

The central planner remains the same active-perception scorer in `core/driver/active_perception.py`. The key extension is that the policy context now accepts `social_ctx`, and forward-model calls may receive optional rollout metadata.

The non-social forward models were left backward-compatible by allowing them to ignore the new optional arguments.

### Social rollout model

The social planner uses `core/social_forward.py`, which predicts:

- next self recurrent-affective state,
- next self explicit homeostatic state,
- next partner homeostatic state,
- local task transition features such as food transfer, movement, and hazard contact.

The action scorer remains self-directed. It never adds a direct partner-welfare term. Instead, predicted partner state can influence predicted self state through `distress_coupled`, which then changes valence, arousal, and `bb_err` before scoring.

### Sequence planning in corridor

The original planner was adequate for one-step toy choices but insufficient for multi-step food delivery. A social sequence planner was therefore added in `core/social_model.py`.

This planner:

- uses the same internal scorer,
- uses the same social forward model,
- performs memoized finite-horizon rollout over candidate action sequences,
- filters invalid actions such as `PASS` without food or `GET` away from the source.

This change was necessary to make helping in the corridor task emerge from a planned fetch-carry-deliver sequence rather than from single-step action repetition.

## Environments

### FoodShareToy

Implemented in `core/envs/social_foodshare.py`.

Properties:

- two agents: active possessor, passive partner,
- possessor actions: `EAT`, `PASS`, `STAY`,
- `EAT` benefits the possessor,
- `PASS` benefits the partner,
- partner death does not terminate the episode.

This environment is intentionally minimal and is paired with an exact solver in `analysis/social_exact_solver.py`. The exact solver verifies that `EAT` is optimal when `lambda_affective = 0`, and that `PASS` becomes optimal above a coupling threshold.

### SocialCorridorWorld

Implemented in `core/envs/social_corridor.py`.

Properties:

- active possessor and passive partner,
- linear corridor geometry,
- food source on one side and partner on the other,
- actions: `LEFT`, `RIGHT`, `GET`, `EAT`, `PASS`, `STAY`,
- helping requires a multi-step fetch-carry-deliver policy,
- beauty shaping is disabled.

A later extension introduced metabolic-load presets to expose regime structure:

- `low`
- `medium`
- `high`

These modify basal cost, movement cost, and food gain while leaving the control architecture unchanged.

## Lesions

The social coupling implementation supports the following lesion modes.

### `none`

No lesion.

### `sham`

A control lesion label with no functional effect.

### `coupling_off`

Disables the affective coupling channel by suppressing partner contribution to self-regulation.

### `shuffle_partner`

Feeds an incorrect partner trajectory into the coupling channel.

### `decoder_off`

Reserved for later decoded-empathy variants.

The lesion suite is implemented from the start rather than as a post hoc manipulation. In the current paper-scale results, `sham` preserves helping and both `coupling_off` and `shuffle_partner` abolish it.

## Logging

### Timestep-level logs

The social tasks write per-step episode logs including:

- seed and episode identity,
- task and condition,
- action,
- `energy_true`, `energy_model`, `energy_pred`,
- `distress_self`, `distress_other_est`, `distress_coupled`, `pe`,
- `valence`, `arousal`, `Ns`, `internal`, `efference`, `g_eff`, `precision_eff`,
- transfer and lesion markers.

### Episode summaries

Episode summaries include:

- `help_rate_when_partner_distressed`
- `partner_recovery_rate`
- `mutual_viability`
- `rescue_latency`
- `self_cost_of_help`
- `episode_joint_longevity`

The corridor sweeps also record `metabolic_load` to support regime-map analysis.

## Analysis Utilities

### Exact solver

`analysis/social_exact_solver.py` provides a brute-force toy solver over food-sharing states. This serves as an analytic anchor and guards against overinterpreting simulation-only results.

### Summary statistics

`analysis/social_summary.py` implements:

- bootstrap confidence intervals,
- Cliff's delta,
- grouped summary tables.

### Claims-as-code

`analysis/social_claims.py` builds paper-facing artifacts from saved CSV outputs and writes:

- `headline_summary.csv`
- `lesion_summary.csv`
- `coupling_sweep.csv`
- `claims.json`

The current artifact directory is `results/social-paper/`.

## Tests

The shipped test suite covers the core mechanistic and causal requirements.

- `lambda_affective = 0` reduces to self-only homeostasis.
- Increasing partner distress increases `distress_coupled`.
- `PASS` reduces predicted partner distress.
- Sham lesion leaves trajectories unchanged.
- Partner death does not terminate the actor's episode.
- The exact toy solver is selfish at zero coupling.
- Coupling lesions reduce helping relative to sham.
- Adjacent baseline tests for active perception and Ipsundrum dynamics remain green.

## Paper-Scale Results Summary

### Headline dissociation

The paper-scale reruns show:

#### FoodShareToy

- `social_none`: no helping
- `social_cognitive_direct`: no helping
- `social_affective_direct`: helping
- `social_full_direct`: helping

#### SocialCorridorWorld

- `social_none`: no helping
- `social_cognitive_direct`: no helping
- `social_affective_direct`: helping
- `social_full_direct`: helping

This gives the intended dissociation: observing partner need alone is insufficient, but coupling partner need into self-regulation is sufficient.

### Lesion results

At paper scale, in both tasks:

- sham lesions preserve helping,
- `coupling_off` abolishes helping,
- `shuffle_partner` abolishes helping.

This supports a causal rather than purely correlational interpretation of the coupling channel.

### Coupling regime map

The final paper-scale coupling sweep now includes metabolic-load bands.

Observed regimes in `results/social-lesions/coupling_sweep.csv`:

- low load: no help at `lambda = 0.0`, reliable helping at intermediate coupling, and loss of helping again at `lambda = 1.0`
- medium load: low viability and no helping across the sweep
- high load: lower viability still and no helping across the sweep

This does not yet produce a classic intermediate optimum in joint viability within the low-load band. However, it does produce a nontrivial regime structure rather than the original flat saturation curve, including a transition from non-helping to helping and then an over-coupled collapse of helping at the largest tested `lambda`.

## Reproducibility Notes

### Profiles

The social experiments support:

- `quick`: smoke-test scale,
- `paper`: 64 seeds per condition.

### Primary output locations

- `results/social-foodshare/`
- `results/social-corridor/`
- `results/social-lesions/`
- `results/social-paper/`

### Recommended commands

To regenerate the social tasks directly:

```bash
PYTHONPATH=. python3 -m experiments.social_foodshare
PYTHONPATH=. python3 -m experiments.social_corridor
PYTHONPATH=. python3 -m experiments.social_lesion_assay
```

To rebuild paper-facing social summaries from saved outputs:

```bash
PYTHONPATH=. python3 -m analysis.social_claims
```

### Validation command

The regression suite used during development was:

```bash
PYTHONPATH=. python3 -m pytest -q \
  tests/test_social_homeostat.py \
  tests/test_social_forward.py \
  tests/test_social_foodshare.py \
  tests/test_social_lesions.py \
  tests/test_active_perception.py \
  tests/test_ipsundrum_dynamics.py
```

## Limitations and Remaining Work

The extension is complete enough for a strong two-task paper, but a few items remain clearly separated as future work rather than hidden limitations.

- Decoded-empathy conditions are scaffolded but not yet implemented as headline experiments.
- The current regime map shows distinct helping and viability bands, but not yet a clean intermediate maximum in joint viability.
- A fully symmetric two-active-agent care world remains a stretch extension.
- Additional paper outputs such as publication-ready plots or LaTeX tables have not yet been added for the social branch.

## Suggested Paper Framing

The cleanest claim supported by the present implementation is:

> In these minimal recurrent agents, helping does not arise from perceiving another's need alone; it arises when another's need becomes part of the controller's own self-regulation.

That sentence matches the actual design and the actual causal interventions now implemented in the repository.
