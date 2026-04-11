from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from core.driver.active_perception import ActionEval, ActivePerceptionPolicy, EnvAdapter, score_internal
from core.ipsundrum_model import AffectParams, Builder, LoopParams
from core.social_forward import SocialForwardContext, predict_one_step_social
from core.social_homeostat import HomeostatParams, SocialCouplingParams, initial_homeostat, social_state_to_net_dict


SOCIAL_ACTIONS = ("EAT", "PASS", "STAY", "LEFT", "RIGHT", "GET")


@dataclass
class SocialAgentConfig:
    model_name: str
    homeostat: HomeostatParams
    social: SocialCouplingParams
    horizon: int
    score_weights: Dict[str, float]


class SocialEnvAdapter(EnvAdapter):
    pass


class SocialAgent:
    def __init__(
        self,
        env: Any,
        config: SocialAgentConfig,
        seed: int = 0,
        *,
        partner_params: Optional[HomeostatParams] = None,
    ) -> None:
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.y = 0
        self.heading = 0
        self.x = int(getattr(env.state, "possessor_x", 0))
        self.has_food = bool(getattr(env.state, "has_food", False))
        self.partner_params = partner_params or config.homeostat
        self._value_cache: Dict[Tuple[Any, ...], Tuple[float, str]] = {}

        loop = LoopParams(
            g=1.0,
            h=1.0,
            internal_decay=0.6,
            fatigue=0.02,
            nonlinearity="linear",
            saturation=True,
            sensor_bias=0.5,
            divisive_norm=0.8,
            efference_decay=0.7,
        )
        aff = AffectParams(
            enabled=True,
            setpoint=config.homeostat.setpoint,
            valence_scale=max(1e-6, config.homeostat.valence_scale),
            arousal_scale=config.homeostat.arousal_scale,
            modulate_g=True,
            k_g_arousal=0.8,
            k_g_unpleasant=0.8,
            modulate_precision=True,
            precision_base=1.0,
            k_precision_arousal=0.5,
        )
        self.builder = Builder(params=loop, affect=aff)
        self.net, _ = self.builder.stage_D(efference_threshold=0.05)
        self.net.start_root(True)
        self.score_weights = dict(config.score_weights)

        self.self_homeostat = initial_homeostat(getattr(env.state, "possessor_energy", config.homeostat.setpoint), config.homeostat)
        self.partner_homeostat = initial_homeostat(getattr(env.state, "partner_energy", config.homeostat.setpoint), self.partner_params)
        self.net._ipsundrum_state.update(social_state_to_net_dict(self.self_homeostat, config.homeostat))  # type: ignore[attr-defined]
        self.net._ipsundrum_state["partner_state"] = social_state_to_net_dict(self.partner_homeostat, self.partner_params)  # type: ignore[attr-defined]
        self.net._ipsundrum_state["x"] = float(self.x)  # type: ignore[attr-defined]
        self.net._ipsundrum_state["has_food"] = 1.0 if self.has_food else 0.0  # type: ignore[attr-defined]

        actions: Sequence[str] = tuple(a for a in SOCIAL_ACTIONS if a in getattr(env, "actions", SOCIAL_ACTIONS))

        def compute_I_affect(_env: Any, _y: int, _x: int, _heading: int):
            return 0.0, 0.0, 0.0, 0.0

        def eval_action(_env: Any, _y: int, x: int, heading: int, action: str) -> ActionEval:
            pred_x = x
            if action == "LEFT":
                pred_x = max(0, x - 1)
            elif action == "RIGHT":
                pred_x = min(max(0, getattr(env, "length", x + 1) - 1), x + 1)
            return ActionEval(y=0, x=pred_x, heading=heading, pred_y=0, pred_x=pred_x, pred_heading=heading, bumped=False)

        self.adapter = SocialEnvAdapter(actions=actions, compute_I_affect=compute_I_affect, eval_action=eval_action)
        self.policy = ActivePerceptionPolicy(self.adapter, forward_model=predict_one_step_social)

    def _social_ctx(self) -> SocialForwardContext:
        return SocialForwardContext(
            env_model=self.env,
            homeostat_params=self.config.homeostat,
            social_params=self.config.social,
            partner_homeostat_params=self.partner_params,
        )

    def _candidate_actions(self, state: Dict[str, Any]) -> Sequence[str]:
        actions = list(self.adapter.actions)
        if hasattr(self.env, "food_source_x"):
            pos = int(round(float(state.get("x", 0.0))))
            has_food = bool(round(float(state.get("has_food", 0.0))))
            partner_x = int(getattr(self.env, "partner_x", pos))
            filtered = []
            for action in actions:
                if action == "EAT" and not has_food:
                    continue
                if action == "GET" and (has_food or pos != int(getattr(self.env, "food_source_x", pos))):
                    continue
                if action == "PASS" and (not has_food or abs(pos - partner_x) > 1):
                    continue
                filtered.append(action)
            actions = filtered or ["STAY"]
        return actions

    def _state_key(self, state: Dict[str, Any], depth: int) -> Tuple[Any, ...]:
        partner = state.get("partner_state", {})
        return (
            depth,
            round(float(state.get("energy_true", 0.0)), 3),
            round(float(state.get("energy_model", 0.0)), 3),
            round(float(state.get("x", 0.0)), 3),
            int(round(float(state.get("has_food", 0.0)))),
            round(float(partner.get("energy_true", 0.0)), 3),
            round(float(partner.get("energy_model", 0.0)), 3),
        )

    def _rollout_value(self, state: Dict[str, Any], depth: int) -> Tuple[float, str]:
        key = self._state_key(state, depth)
        if key in self._value_cache:
            return self._value_cache[key]
        best_score = -1e18
        best_action = "STAY"
        for action in self._candidate_actions(state):
            pred = predict_one_step_social(
                state,
                self.builder.params,
                self.builder.affect,
                0.0,
                rng=self.rng,
                social_ctx=self._social_ctx(),
                action=action,
            )
            score = score_internal(pred, self.builder.affect, current_I=0.0, predicted_I=0.0, w_epistemic=0.0, **self.score_weights)
            if depth > 1:
                future_score, _ = self._rollout_value(pred, depth - 1)
                score += 0.95 * future_score
            if score > best_score:
                best_score = float(score)
                best_action = str(action)
        out = (best_score, best_action)
        self._value_cache[key] = out
        return out

    def choose_action(self) -> str:
        self._value_cache.clear()
        _, best_action = self._rollout_value(dict(getattr(self.net, "_ipsundrum_state", {})), self.config.horizon)
        return best_action
