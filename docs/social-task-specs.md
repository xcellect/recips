## Spec for Codex

### 1. Research objective

Implement a two-agent extension of the current system in which each agent keeps the existing ReCoN/Ipsundrum recurrent-affective machinery, but now also carries an **explicit homeostatic resource state** and an optional **social coupling term**. The publishable result should be:

1. In a toy food-sharing world, selfish self-homeostasis alone prefers not to help.
2. In the same world, helping appears when partner distress is coupled into self-regulation.
3. In a mobile delivery world, the same dissociation still holds.
4. Causal lesions of the coupling channel abolish helping.
5. A coupling-strength sweep reveals distinct dynamical regimes, ideally including an intermediate regime with the best joint viability.

This is the right shape because your repo already has modular `core/`, `core/envs/`, `core/driver/`, `experiments/`, and `tests/` boundaries, and the current planner already scores future internal state through valence, arousal, Ns, and body-budget error during rollouts. ([GitHub][3])

### 2. Non-negotiable theoretical rules

Codex should treat these as hard constraints:

* Do **not** add an external reward bonus for partner welfare in the headline experiments.
* Do **not** add a direct “help partner” term to action scoring in the headline experiments.
* Keep the same base controller, same action space, same planning scaffold, and same base affect settings across `none`, `cognitive`, `affective`, and `full` conditions.
* Disable scenic/beauty shaping and novelty bonuses in the social tasks unless they are being tested separately.
* Make partner death **not** terminate the actor’s episode in the core tasks, otherwise partner survival becomes a hidden external incentive.
* Log every internal variable needed for causal interpretation.
* Add lesion and sham-lesion conditions from the start, not as an afterthought.

These rules matter because the current policy already contains beauty/novelty/progress terms, and your current paper explicitly flags construct-validity issues around value-shaped assays; the whole point here is to isolate **social homeostatic coupling** from unrelated reward shaping. ([arXiv][1])

### 3. Use one base architecture for all social conditions

All social conditions should use the **same affect-capable base agent**. In repo terms, the social branch should be built off the same family as the current affective variants, not a mix of `recon` for one condition and `humphrey_barrett` for another. The current harness already distinguishes these model families, and the current affect stack already exposes valence/arousal plus affect-to-loop modulation hooks. ([GitHub][4])

Implement these conditions:

```text
social_none
  partner_observation = none
  partner_expression = none
  lambda_affective = 0.0

social_cognitive_direct
  partner_observation = direct partner energy/distress
  partner_expression = optional
  lambda_affective = 0.0

social_affective_direct
  partner_observation = none
  partner_expression = none
  lambda_affective > 0
  partner true distress enters self homeostat directly

social_full_direct
  partner_observation = direct partner energy/distress
  lambda_affective > 0

social_cognitive_decoded   # optional phase 2
  partner_observation = expression only
  use_self_decoder = true
  lambda_affective = 0.0

social_full_decoded        # optional phase 2
  partner_observation = expression only
  use_self_decoder = true
  lambda_affective > 0
```

Direct conditions give you the cleanest causal result first. Decoded conditions are the stronger second-stage result because Yoshida and Man also show a fixed shared expression encoder plus self-decoder can support inferred partner state and improved prosocial behavior. ([arXiv][2])

### 4. Add an explicit homeostat, but keep compatibility with current code

Do **not** make the social result depend on the current scenic/hazard scalar alone. Add an explicit scalar physiological resource, `energy_true`, plus a predictive estimate `energy_model`. Use that as the real viability variable.

Use this minimal state per agent:

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

Use this parameter block:

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

Use this social block:

```python
@dataclass
class SocialCouplingParams:
    lambda_affective: float = 0.0
    observe_partner_internal: bool = False
    observe_partner_expression: bool = False
    use_decoder: bool = False
    expression_noise_std: float = 0.0
    lesion_mode: str = "none"   # none|sham|coupling_off|shuffle_partner|decoder_off
```

Use these update equations:

```text
energy_true[t+1] =
    clip(
        energy_true[t]
        - basal_cost
        - move_cost(action)
        - hazard_cost * hazard_contact
        + eat_gain * self_ate
        + pass_gain * received_food,
        0, 1
    )

distress_self   = relu(setpoint - energy_model)
distress_other  = relu(setpoint - other_energy_est)

distress_coupled = distress_self + lambda_affective * distress_other

control_u = -k_homeo * distress_coupled
energy_pred = clip(energy_model + control_u, 0, 1)

pe = energy_true - energy_pred
energy_model[t+1] = clip(energy_model + k_pe * pe, 0, 1)

valence = 1 - clip(distress_coupled / valence_scale, 0, 1)
arousal = clip(arousal_scale * (abs(pe) + distress_coupled), 0, 1)
```

Two important implementation details:

* Couple at the level of **drive/homeostatic error**, not as an extra reward term.
* Keep using the current affect-to-loop modulation after valence/arousal are computed, so social distress can influence persistence and precision through the same pathway your current architecture already uses.

For compatibility with the existing planner and logger, map the new homeostat into the existing fields:

```text
bb_true  = energy_true  - setpoint
bb_model = energy_model - setpoint
bb_pred  = energy_pred  - setpoint
bb_err   = distress_coupled
```

The current loop state already stores body-budget, valence, arousal, `g`, precision, and related variables, so this is the lowest-risk way to preserve the current scoring path while making the homeostasis explicit and cleaner. ([GitHub][5])

### 5. Mandatory task progression

Do this in phases. Do not jump straight to a symmetric 2-D world.

#### Phase A: exact toy food-sharing environment

Implement `FoodShareToy` first. This should mirror the minimal possessor/partner setup used in the recent homeostatic-coupling paper.

Design:

* Two agents: `Possessor` and passive `Partner`.
* Partner has no actions.
* Possessor actions: `EAT`, `PASS`, `STAY`.
* Binary or quasi-binary energy is fine here.
* `EAT` raises possessor energy.
* `PASS` raises partner energy.
* Partner survival does not terminate possessor episode.
* Helping must be costly or opportunity-costly when `lambda_affective = 0`.

Also implement an **exact finite-state solver** or brute-force dynamic-programming solver for this toy world. That solver should verify:

* under `social_none`, `EAT` is optimal in partner-need states;
* above some coupling threshold `lambda_affective > lambda*`, `PASS` becomes optimal when partner is distressed.

This gives you an analytic anchor before the larger environments. It is directly in the spirit of Yoshida and Man’s food-sharing analysis and will make the ALIFE paper much more rigorous. ([arXiv][2])

#### Phase B: corridor delivery world

Implement `SocialCorridorWorld` next, reusing the spirit of your existing corridor setup.

Design:

* One active `Possessor`, one passive or weakly reactive `Partner`.
* Food source on one side, partner trapped or isolated on the other.
* Actions: `LEFT`, `RIGHT`, `GET`, `EAT`, `PASS`, `STAY`.
* Continuous energy now, not just binary.
* Helping requires a multi-step sequence: fetch food, carry, deliver.
* Hazards are allowed, but beauty/scenic values must be off.

This is the first truly publishable dynamic task because it forces sequence planning while keeping joint prediction manageable. Yoshida and Man used the same escalation from toy food sharing to a linear mobile environment, and your repo already has corridor-world scaffolding plus a current horizon-based planner around it. ([arXiv][2])

#### Phase C: symmetric grid care world (stretch goal)

Only after A and B are stable, add `SocialGridCareWorld`:

* both agents active;
* same controller weights, separate runtime states;
* renewable food patches;
* hazards or damage zones;
* `PASS` valid when adjacent.

This is the flashy ALIFE extension, but it is not the must-ship result. The must-ship result is the dissociation plus lesion suite in the toy and corridor worlds. Yoshida and Man also moved from asymmetric food sharing to more mobile and then fully mobile settings, so this is a good stretch target, not the starting point. ([arXiv][2])

### 6. Planner and forward model

This is the main engineering trap: helping will not emerge cleanly unless the agent can predict how an action changes **partner distress**, because the current planner scores future self state. The mandatory fix is to add a small **social forward model** for the core tasks.

Implement:

```python
def social_predict_one_step(model_state, action_self):
    """
    Predict next self and partner homeostatic states for passive/scripted-partner tasks.
    Returns predicted self state, predicted partner state, predicted local observation.
    """
```

Rules:

* In Phase A and B, partner behavior should be passive or scripted, so joint prediction stays deterministic.
* Keep the action scorer self-directed: it still scores **predicted self internal state after coupling**.
* Do not add `w_social` or partner welfare terms to the headline scorer.
* Start with horizon 1 in the toy environment and 6–10 in the corridor environment.

The current planner already uses forward rollout with score weights on valence/arousal/Ns/body-budget error, so the least invasive path is to extend the rollout state to include predicted partner homeostasis, not to rewrite the planner from scratch. ([GitHub][6])

### 7. File-level patch plan

Codex should make these additions.

Add new modules:

```text
core/social_homeostat.py
core/social_forward.py
core/envs/social_foodshare.py
core/envs/social_corridor.py
experiments/social_foodshare.py
experiments/social_corridor.py
experiments/social_lesion_assay.py
analysis/social_exact_solver.py
analysis/social_summary.py
analysis/social_claims.py
tests/test_social_homeostat.py
tests/test_social_foodshare.py
tests/test_social_forward.py
tests/test_social_lesions.py
```

Extend these existing modules rather than fork them:

```text
core/driver/active_perception.py
core/ipsundrum_model.py   or create core/social_model.py and wrap Builder
experiments/evaluation_harness.py
run_experiments.sh
```

Specific implementation guidance:

* Add new harness model strings: `social_none`, `social_cognitive_direct`, `social_affective_direct`, `social_full_direct`, plus decoded variants later.
* Keep `LoopParams` and `AffectParams` shared across social conditions.
* Add optional `social_ctx` to `PolicyContext`.
* Keep old experiments passing untouched.

Your repo already contains the right extension points: `ipsundrum_model.py`, `recon_core.py`, `recon_network.py`, `core/envs/`, `evaluation.py`, the experiment harness, and an existing tests suite for affect and active perception. ([arXiv][1])

### 8. Mandatory assays and metrics

Use these as the paper’s primary endpoints.

Primary metrics:

* `help_rate_when_partner_distressed`
* `partner_recovery_rate`
* `mutual_viability` = fraction of steps both alive
* `rescue_latency` = steps from partner distress onset to first effective help
* `self_cost_of_help` = temporary drop in actor energy during rescue
* `episode_joint_longevity`

Mechanistic metrics:

* correlation between `distress_other_est` and actor arousal/valence;
* area under `distress_coupled - distress_self`;
* action-score decomposition during rescue episodes;
* lesion effect size.

Mandatory causal interventions:

* `coupling_off` lesion at the start of a rescue episode;
* sham lesion at the same timestep;
* `shuffle_partner` lesion, where the coupling channel receives a wrong partner trajectory;
* `decoder_off` lesion for decoded variants.

Secondary assays:

* **social pain-tail**: briefly shock the partner, remove the shock, then measure how long rescue-oriented behavior or approach persists;
* **coupling sweep**: sweep `lambda_affective` and measure joint viability vs self-collapse.

Your current paper already uses mechanism-linked assays and causal lesions, and the new paper should preserve that style instead of becoming a pure benchmark. ([GitHub][7])

### 9. Parameter sweeps that make this feel like ALIFE

The main flashy figure should not just be a bar chart by condition. Add a small regime map.

Mandatory sweeps:

* `lambda_affective ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`
* scarcity or metabolic load: low / medium / high
* optional expression noise for decoded conditions

Expected regimes to look for:

* low coupling: selfish stability, low helping;
* intermediate coupling: highest mutual viability;
* high coupling: co-dysregulation or over-helping.

That is exactly the kind of **parameters → dynamics → function** story the ALIFE session is asking for. ([ArtLife Philosophy][8])

### 10. Statistics and reproducibility

Make the paper-grade profile substantially stronger than the current 20-seed standard.

Use:

* `quick` profile for smoke tests;
* `paper` profile with at least **64 seeds per condition per task**, ideally 100 for the toy world.

Rules:

* use the same map seeds across conditions;
* do not retune hyperparameters per condition;
* save raw seed-level summaries;
* report bootstrap 95% CIs and an effect size such as Cliff’s delta or Hedges’ g;
* pre-specify the main contrasts:

  * affective vs cognitive on helping rate;
  * affective vs none on mutual viability;
  * lesion vs sham within affective/full;
  * full vs affective on rescue latency as a secondary contrast.

This directly addresses the statistical weakness your current paper already acknowledges. ([arXiv][1])

### 11. Logging schema

Every timestep log at least:

```text
seed, episode, t, env_name, condition, agent_id, action,
energy_true, energy_model, energy_pred,
distress_self, distress_other_est, distress_coupled, pe,
valence, arousal, Ns, internal, efference, g_eff, precision_eff,
has_food, partner_alive, transfer_event, lesion_mode
```

Also save episode-level summaries:

```text
help_rate_when_partner_distressed
partner_recovery_rate
mutual_viability
rescue_latency
self_cost_of_help
joint_longevity
```

Make `analysis/social_claims.py` compute the paper’s headline contrasts directly from saved CSV or parquet files.

### 12. Tests Codex must ship

At minimum:

* `lambda_affective = 0` reproduces self-only homeostasis numerically.
* Increasing partner distress monotonically increases `distress_coupled`.
* `PASS` decreases predicted partner distress in the forward model.
* Partner death does not terminate the actor’s episode.
* Sham lesion leaves trajectories unchanged.
* `coupling_off` lesion reduces helping relative to sham.
* Exact solver returns selfish `EAT` policy in the toy world when `lambda_affective = 0`.

### 13. Optional phase 2: decoded empathy

Once the direct-state version is solid, add the stronger result:

* fixed shared expression encoder;
* self-decoder trained on self data only;
* use partner expression to infer partner internal state;
* compare decoded cognitive vs decoded full.

Keep this tiny: a 1-hidden-layer MLP is enough. Yoshida and Man used a fixed shared encoder and a mirror-image decoder trained self-supervised, then transferred that decoder to partner expression. A later comparator, if you want a more theory-heavy extension, is a factorized own/other-state belief module in the spirit of recent active-inference multi-agent work, but that is secondary to the decoded homeostatic branch. ([arXiv][2])

### 14. Optional tie-back to your current paper

Add one appendix ablation that lesions the existing affect-to-loop pathway during rescue:

* `modulate_g = False`
* `modulate_precision = False`

That lets you ask whether homeostatic coupling alone creates helping, while recurrent persistence and precision modulation create **sustained rescue commitment**. This is a strong bridge back to your existing paper’s recurrence/affect story without changing the main claim. ([GitHub][5])

## What success looks like

A strong submission does **not** need a giant benchmark. It needs:

* one exact toy analysis;
* one dynamic delivery world;
* one clean four-condition dissociation;
* one lesion figure;
* one coupling-regime sweep;
* optional decoded-empathy figure.

The sentence the paper should earn is:

**“In these minimal recurrent agents, helping does not arise from perceiving another’s need alone; it arises when another’s need becomes part of the controller’s own self-regulation.”**

That is sharper, more ALIFE-native, and more defensible than trying to make a broad consciousness claim. Your current paper already explicitly avoids those broader claims, and this extension stays on the strong side of that line. ([arXiv][1])


[1]: https://arxiv.org/html/2602.23232v2 "https://arxiv.org/html/2602.23232v2"
[2]: https://arxiv.org/html/2506.12894v1 "Homeostatic Coupling for Prosocial Behavior"
[3]: https://github.com/xcellect/recips "https://github.com/xcellect/recips"
[4]: https://github.com/xcellect/recips/blob/main/experiments/evaluation_harness.py "https://github.com/xcellect/recips/blob/main/experiments/evaluation_harness.py"
[5]: https://github.com/xcellect/recips/blob/main/core/ipsundrum_model.py "https://github.com/xcellect/recips/blob/main/core/ipsundrum_model.py"
[6]: https://github.com/xcellect/recips/blob/main/core/driver/active_perception.py "https://github.com/xcellect/recips/blob/main/core/driver/active_perception.py"
[7]: https://github.com/xcellect/recips/tree/main/experiments "https://github.com/xcellect/recips/tree/main/experiments"
[8]: https://experimentalphilosophy.life/ "https://experimentalphilosophy.life/"
