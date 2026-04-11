from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


FOODSHARE_ACTIONS = ("EAT", "PASS", "STAY")


@dataclass
class FoodShareState:
    possessor_energy: float
    partner_energy: float
    t: int = 0
    done: bool = False


class FoodShareToy:
    actions = FOODSHARE_ACTIONS
    use_beauty_term = False
    hazard_penalty = 0.0
    stay_penalty = 0.0
    forward_prior = 0.0
    turn_cost = 0.0
    bump_penalty = 0.0

    def __init__(self, horizon: int = 1, initial_possessor: float = 0.55, initial_partner: float = 0.20) -> None:
        self.horizon = int(horizon)
        self.initial_possessor = float(initial_possessor)
        self.initial_partner = float(initial_partner)
        self.state = FoodShareState(self.initial_possessor, self.initial_partner)

    def reset(self) -> FoodShareState:
        self.state = FoodShareState(self.initial_possessor, self.initial_partner)
        return self.state

    def step(self, action: str) -> FoodShareState:
        self.state = FoodShareState(
            possessor_energy=self.state.possessor_energy,
            partner_energy=self.state.partner_energy,
            t=self.state.t + 1,
            done=(self.state.t + 1) >= self.horizon,
        )
        return self.state

    def predict_transition(self, self_state: Dict[str, float], partner_state: Dict[str, float], action: str, eval_info: Any = None) -> Dict[str, float]:
        _ = eval_info
        return {
            "move_amount_self": 0.0,
            "hazard_contact_self": 0.0,
            "self_ate": 1.0 if action == "EAT" else 0.0,
            "received_food_self": 0.0,
            "partner_ate": 0.0,
            "received_food_partner": 1.0 if action == "PASS" else 0.0,
            "hazard_contact_partner": 0.0,
            "other_energy_observed": float(partner_state.get("energy_true", 0.0)),
            "other_expression": float(partner_state.get("distress_self", 0.0)),
            "shuffled_other_energy": float(self_state.get("energy_true", 0.0)),
            "transfer_event": 1.0 if action == "PASS" else 0.0,
        }

    def eval_action(self, action: str) -> Tuple[int, int, int]:
        _ = action
        return 0, 0, 0
