import core.driver.ipsundrum_forward as ipsundrum_forward
import experiments.gridworld_exp as gw


def test_recon_planning_does_not_call_ipsundrum_forward(monkeypatch):
    calls = {"n": 0}
    original = ipsundrum_forward.predict_one_step

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ipsundrum_forward, "predict_one_step", _spy)

    env = gw.GridWorld(H=6, W=6, seed=0)
    agent = gw.Agent(env, mode="recon", seed=0)
    agent.eps = 0.0

    for _ in range(5):
        agent.step()

    assert calls["n"] == 0
