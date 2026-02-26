import pytest

from core.driver.env_adapters import gridworld_adapter
from core.envs.gridworld import GridWorld


def test_gridworld_forward_bump_and_penalty():
    env = GridWorld(H=6, W=6, seed=0)
    adapter = gridworld_adapter()

    y, x, heading = 0, 0, 0  # facing up at top-left boundary
    eval_info = adapter.eval_action(env, y, x, heading, "forward")

    assert eval_info.bumped is True
    assert adapter.bump_penalty_fn(env) > 0.0
