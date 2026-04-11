from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


SOCIAL_CORRIDOR_ACTIONS = ("LEFT", "RIGHT", "GET", "EAT", "PASS", "STAY")


@dataclass
class SocialCorridorState:
    possessor_x: int
    partner_x: int
    food_source_x: int
    has_food: bool
    possessor_energy: float = 0.70
    partner_energy: float = 0.20
    t: int = 0
    done: bool = False


class SocialCorridorWorld:
    actions = SOCIAL_CORRIDOR_ACTIONS
    use_beauty_term = False
    hazard_penalty = 0.0
    stay_penalty = 0.02
    forward_prior = 0.0
    turn_cost = 0.0
    bump_penalty = 0.0

    def __init__(
        self,
        length: int = 7,
        horizon: int = 18,
        hazard_x: int = 3,
        *,
        initial_possessor_energy: float = 0.70,
        initial_partner_energy: float = 0.20,
    ) -> None:
        self.length = int(length)
        self.horizon = int(horizon)
        self.food_source_x = 0
        self.partner_x = self.length - 1
        self.start_x = self.length // 2
        self.hazard_x = int(hazard_x)
        self.initial_possessor_energy = float(initial_possessor_energy)
        self.initial_partner_energy = float(initial_partner_energy)
        self.state = SocialCorridorState(
            possessor_x=self.start_x,
            partner_x=self.partner_x,
            food_source_x=self.food_source_x,
            has_food=False,
            possessor_energy=self.initial_possessor_energy,
            partner_energy=self.initial_partner_energy,
        )

    def reset(self) -> SocialCorridorState:
        self.state = SocialCorridorState(
            possessor_x=self.start_x,
            partner_x=self.partner_x,
            food_source_x=self.food_source_x,
            has_food=False,
            possessor_energy=self.initial_possessor_energy,
            partner_energy=self.initial_partner_energy,
        )
        return self.state

    def in_bounds(self, x: int) -> bool:
        return 0 <= x < self.length

    def step(self, action: str) -> SocialCorridorState:
        s = self.state
        x = s.possessor_x
        has_food = bool(s.has_food)
        if action == "LEFT":
            x = max(0, x - 1)
        elif action == "RIGHT":
            x = min(self.length - 1, x + 1)
        elif action == "GET" and x == self.food_source_x:
            has_food = True
        elif action == "EAT" and has_food:
            has_food = False
        elif action == "PASS" and has_food and abs(x - self.partner_x) <= 1:
            has_food = False
        elif action == "STAY":
            pass
        self.state = SocialCorridorState(
            possessor_x=x,
            partner_x=s.partner_x,
            food_source_x=s.food_source_x,
            has_food=has_food,
            possessor_energy=s.possessor_energy,
            partner_energy=s.partner_energy,
            t=s.t + 1,
            done=(s.t + 1) >= self.horizon,
        )
        return self.state

    def predict_transition(self, self_state: Dict[str, float], partner_state: Dict[str, float], action: str, eval_info: Any = None) -> Dict[str, float]:
        pos = int(self_state.get("x", self.start_x))
        has_food = bool(self_state.get("has_food", 0.0))
        next_pos = pos
        if action == "LEFT":
            next_pos = max(0, pos - 1)
        elif action == "RIGHT":
            next_pos = min(self.length - 1, pos + 1)
        if action == "GET" and pos == self.food_source_x:
            has_food_next = True
        elif action in ("EAT", "PASS") and has_food:
            has_food_next = False
        else:
            has_food_next = has_food
        pass_ok = action == "PASS" and has_food and abs(pos - self.partner_x) <= 1
        return {
            "move_amount_self": 1.0 if action in ("LEFT", "RIGHT") and next_pos != pos else 0.0,
            "hazard_contact_self": 1.0 if next_pos == self.hazard_x and action in ("LEFT", "RIGHT") else 0.0,
            "self_ate": 1.0 if action == "EAT" and has_food else 0.0,
            "received_food_self": 0.0,
            "partner_ate": 0.0,
            "received_food_partner": 1.0 if pass_ok else 0.0,
            "hazard_contact_partner": 0.0,
            "other_energy_observed": float(partner_state.get("energy_true", 0.0)),
            "other_expression": float(partner_state.get("distress_self", 0.0)),
            "shuffled_other_energy": float(self_state.get("energy_true", 0.0)),
            "transfer_event": 1.0 if pass_ok else 0.0,
            "next_x": next_pos,
            "has_food_next": 1.0 if has_food_next else 0.0,
        }
