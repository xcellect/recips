from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _relu(x: float) -> float:
    return max(0.0, float(x))


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
    below_threshold_steps: int = 0


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


@dataclass
class SocialCouplingParams:
    lambda_affective: float = 0.0
    observe_partner_internal: bool = False
    observe_partner_expression: bool = False
    use_decoder: bool = False
    expression_noise_std: float = 0.0
    lesion_mode: str = "none"


@dataclass
class SocialObservation:
    other_energy_est: Optional[float] = None
    other_expression: Optional[float] = None
    shuffled_energy_est: Optional[float] = None


@dataclass
class HomeostatTransition:
    state: HomeostatState
    debug: Dict[str, float] = field(default_factory=dict)


def initial_homeostat(energy: float, params: Optional[HomeostatParams] = None) -> HomeostatState:
    p = params or HomeostatParams()
    energy = _clip01(energy)
    distress = _relu(p.setpoint - energy)
    valence = 1.0 - _clip01(distress / max(1e-9, p.valence_scale))
    arousal = _clip01(p.arousal_scale * distress)
    return HomeostatState(
        energy_true=energy,
        energy_model=energy,
        energy_pred=energy,
        distress_self=distress,
        distress_other_est=0.0,
        distress_coupled=distress,
        pe=0.0,
        valence=valence,
        arousal=arousal,
        alive=energy > p.death_threshold,
        below_threshold_steps=0,
    )


def infer_partner_energy(
    social: SocialCouplingParams,
    obs: SocialObservation,
    *,
    rng=None,
) -> Optional[float]:
    lesion = str(getattr(social, "lesion_mode", "none"))
    if lesion == "coupling_off":
        return None
    if lesion == "shuffle_partner":
        return obs.shuffled_energy_est
    if social.observe_partner_internal and obs.other_energy_est is not None:
        return _clip01(obs.other_energy_est)
    if (not social.observe_partner_internal) and (not social.observe_partner_expression) and obs.other_energy_est is not None and float(social.lambda_affective) > 0.0:
        return _clip01(obs.other_energy_est)
    if social.observe_partner_expression and obs.other_expression is not None:
        if lesion == "decoder_off" and social.use_decoder:
            return None
        noise = 0.0
        if social.expression_noise_std > 0.0 and rng is not None:
            noise = float(rng.normal(0.0, social.expression_noise_std))
        expr = _clip01(obs.other_expression + noise)
        return expr
    return None


def step_homeostat(
    state: HomeostatState,
    params: HomeostatParams,
    social: SocialCouplingParams,
    *,
    move_amount: float = 0.0,
    hazard_contact: float = 0.0,
    self_ate: float = 0.0,
    received_food: float = 0.0,
    partner_observation: Optional[SocialObservation] = None,
    rng=None,
) -> HomeostatTransition:
    obs = partner_observation or SocialObservation()
    energy_true = _clip01(
        state.energy_true
        - params.basal_cost
        - params.move_cost * float(move_amount)
        - params.hazard_cost * float(hazard_contact)
        + params.eat_gain * float(self_ate)
        + params.pass_gain * float(received_food)
    )

    other_energy_est = infer_partner_energy(social, obs, rng=rng)
    distress_self_now = _relu(params.setpoint - state.energy_model)
    distress_other_now = 0.0 if other_energy_est is None else _relu(params.setpoint - other_energy_est)
    lambda_eff = 0.0 if str(social.lesion_mode) == "coupling_off" else float(social.lambda_affective)
    distress_coupled_now = distress_self_now + lambda_eff * distress_other_now

    control_u = -params.k_homeo * distress_coupled_now
    energy_pred = _clip01(state.energy_model + control_u)
    pe = energy_true - energy_pred
    energy_model = _clip01(state.energy_model + params.k_pe * pe)

    distress_self = _relu(params.setpoint - energy_model)
    distress_other = 0.0 if other_energy_est is None else _relu(params.setpoint - other_energy_est)
    distress_coupled = distress_self + lambda_eff * distress_other
    valence = 1.0 - _clip01(distress_coupled / max(1e-9, params.valence_scale))
    arousal = _clip01(params.arousal_scale * (abs(pe) + distress_coupled))

    below = int(state.below_threshold_steps)
    if energy_true <= params.death_threshold:
        below += 1
    else:
        below = 0
    alive = below < int(params.death_patience)

    next_state = HomeostatState(
        energy_true=energy_true,
        energy_model=energy_model,
        energy_pred=energy_pred,
        distress_self=distress_self,
        distress_other_est=distress_other,
        distress_coupled=distress_coupled,
        pe=pe,
        valence=valence,
        arousal=arousal,
        alive=alive,
        below_threshold_steps=below,
    )
    return HomeostatTransition(
        state=next_state,
        debug={
            "control_u": float(control_u),
            "other_energy_est": float(other_energy_est) if other_energy_est is not None else float("nan"),
        },
    )


def social_state_to_net_dict(state: HomeostatState, params: HomeostatParams) -> Dict[str, float]:
    out = asdict(state)
    out.update(
        {
            "bb_true": float(state.energy_true - params.setpoint),
            "bb_model": float(state.energy_model - params.setpoint),
            "bb_pred": float(state.energy_pred - params.setpoint),
            "bb_err": float(state.distress_coupled),
        }
    )
    return out


def homeostat_from_net_dict(state: Dict[str, float], params: HomeostatParams) -> HomeostatState:
    return HomeostatState(
        energy_true=float(state.get("energy_true", params.setpoint)),
        energy_model=float(state.get("energy_model", params.setpoint)),
        energy_pred=float(state.get("energy_pred", params.setpoint)),
        distress_self=float(state.get("distress_self", 0.0)),
        distress_other_est=float(state.get("distress_other_est", 0.0)),
        distress_coupled=float(state.get("distress_coupled", 0.0)),
        pe=float(state.get("pe", 0.0)),
        valence=float(state.get("valence", 1.0)),
        arousal=float(state.get("arousal", 0.0)),
        alive=bool(state.get("alive", True)),
        below_threshold_steps=int(state.get("below_threshold_steps", 0)),
    )
