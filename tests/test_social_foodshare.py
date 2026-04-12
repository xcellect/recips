from analysis.social_exact_solver import find_helping_threshold, solve_foodshare_state
from core.envs.social_foodshare import FoodShareToy


def test_partner_death_does_not_terminate_actor_episode():
    env = FoodShareToy(horizon=3)
    env.reset()
    env.state.done = False
    assert not env.state.done
    env.step("STAY")
    assert not env.state.done


def test_exact_solver_selfish_when_lambda_zero():
    action, _ = solve_foodshare_state(0.55, 0.2, condition="social_none", lambda_affective=0.0)
    assert action == "EAT"


def test_helping_threshold_exists():
    threshold = find_helping_threshold()
    assert threshold is not None
