import pytest

import experiments.gridworld_exp as gw
from experiments.evaluation_harness import EvalAgent
from core.envs.gridworld import GridWorld
from core.driver.active_perception import ActivePerceptionPolicy


class CapturePolicy(ActivePerceptionPolicy):
    def __init__(self, adapter, forward_model):
        super().__init__(adapter, forward_model=forward_model)
        self.last_w_epistemic = None

    def choose_action(self, ctx, **kwargs):
        self.last_w_epistemic = kwargs.get("w_epistemic")
        return "stay"


def test_epistemic_weight_precedence():
    env = GridWorld(H=6, W=6, seed=0)
    agent = EvalAgent(env, model="recon", seed=0, start=(3, 3), heading=1, eps=0.0)
    agent.score_weights = {"w_epistemic": 0.0}

    policy = CapturePolicy(gw._GRIDWORLD_ADAPTER, forward_model=gw.select_forward_model(model="recon"))
    agent.policy = policy

    gw.choose_action_feelings(agent, horizon=1, w_epistemic=0.35)
    assert policy.last_w_epistemic == pytest.approx(0.35)

    gw.choose_action_feelings(agent, horizon=1, w_epistemic=None)
    assert policy.last_w_epistemic == pytest.approx(0.0)
