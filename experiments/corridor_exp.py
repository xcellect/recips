import numpy as np
from collections import deque

from core.envs.corridor import CorridorWorld, gaussian_kernel, conv2_same
from core.ipsundrum_model import Builder, LoopParams, AffectParams
from core.driver.active_perception import (
    ActivePerceptionPolicy,
    PolicyContext,
    score_internal as score_internal,
)
from core.driver.env_adapters import corridor_adapter
from core.driver.ipsundrum_forward import predict_one_step as _predict_one_step
from core.driver.recon_forward import predict_one_step_recon as _predict_one_step_recon
from core.driver.sensory import compute_I_affect as _compute_I_affect
from typing import Optional
from utils.model_naming import canonical_model_id


# =========================
# Internal-feelings policy (same mechanism as gridworld)
# =========================

def compute_I_affect(env: CorridorWorld, y, x, heading):
    """
    Sensor fusion into signed scalar I in [-1,1].
    EXACT same structure as your gridworld: touch + smell + vision
    """
    return _compute_I_affect(env, y, x, heading)


def predict_one_step(
    state, loop: LoopParams, aff: AffectParams,
    I_ext, rng,
):
    return _predict_one_step(state, loop, aff, I_ext, rng)


_CORRIDOR_ADAPTER = corridor_adapter(compute_I_affect)

def select_forward_model(*, model=None, agent=None):
    name = model
    if name is None and agent is not None:
        name = getattr(agent, "mode", None) or getattr(agent, "model", None)
    if canonical_model_id(str(name)) == "recon":
        return _predict_one_step_recon
    return _predict_one_step


def choose_action_feelings(
    agent,
    horizon=6,
    curiosity=False,
    w_progress: Optional[float] = None,
    w_epistemic: Optional[float] = None,
    beauty_weight=1.0,
    use_beauty_term=None,
    w_valence=None,
    w_arousal=None,
    w_ns=None,
    w_bb_err=None,
    novelty_scale=None,
):
    """
    Active-inference style action selection (purely internal).
    """
    policy = getattr(agent, "policy", None)
    forward_model = select_forward_model(agent=agent)
    if (
        not isinstance(policy, ActivePerceptionPolicy)
        or policy.adapter is not _CORRIDOR_ADAPTER
        or policy.forward_model is not forward_model
    ):
        policy = ActivePerceptionPolicy(_CORRIDOR_ADAPTER, forward_model=forward_model)
        try:
            agent.policy = policy
        except Exception:
            pass
    efference_threshold = None
    try:
        if "P" in agent.net.nodes:
            efference_threshold = getattr(agent.net.get("P"), "efference_threshold", None)
    except Exception:
        efference_threshold = None
    ctx = PolicyContext(
        env=agent.env,
        y=agent.y,
        x=agent.x,
        heading=agent.heading,
        rng=agent.rng,
        loop=agent.b.params,
        aff=agent.b.affect,
        net_state=getattr(agent.net, "_ipsundrum_state", {}),
        efference_threshold=efference_threshold,
    )
    score_kwargs = {}
    agent_weights = getattr(agent, "score_weights", None)
    if isinstance(agent_weights, dict):
        score_kwargs.update(agent_weights)
    w_epistemic_final = (
        float(w_epistemic)
        if w_epistemic is not None
        else float(score_kwargs.pop("w_epistemic", 0.35))
    )
    w_progress_final = (
        float(w_progress)
        if w_progress is not None
        else float(score_kwargs.pop("w_progress", 0.0))
    )
    score_kwargs.pop("w_epistemic", None)
    score_kwargs.pop("w_progress", None)
    if w_valence is not None:
        score_kwargs["w_valence"] = w_valence
    if w_arousal is not None:
        score_kwargs["w_arousal"] = w_arousal
    if w_ns is not None:
        score_kwargs["w_ns"] = w_ns
    if w_bb_err is not None:
        score_kwargs["w_bb_err"] = w_bb_err
    if novelty_scale is not None:
        score_kwargs["novelty_scale"] = novelty_scale

    return policy.choose_action(
        ctx,
        horizon=horizon,
        curiosity=curiosity,
        w_progress=w_progress_final,
        w_epistemic=w_epistemic_final,
        beauty_weight=beauty_weight,
        use_beauty_term=use_beauty_term,
        **score_kwargs,
    )


# =========================
# Agent wrapper (same as gridworld, just uses CorridorWorld)
# =========================

class Agent:
    def __init__(self, env: CorridorWorld, mode="humphrey_barrett", seed=0, start=None, heading=2, horizon=6):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self.policy = ActivePerceptionPolicy(
            _CORRIDOR_ADAPTER,
            forward_model=select_forward_model(model=mode),
        )
        self.horizon = int(horizon)

        if start is None:
            self.y, self.x = 1, env.goal_x
        else:
            self.y, self.x = start
        self.heading = heading  # default: facing down toward goal

        # Loop params: keep signed I in [-1,1] meaningful in Ns ∈ [0,1] via sensor_bias.
        loop = LoopParams(
            g=1.0,
            h=1.0,
            internal_decay=0.6,
            fatigue=0.02,
            nonlinearity="linear",
            saturation=True,
            sensor_bias=0.5,
        )
        # optional if your LoopParams supports it
        if hasattr(loop, "divisive_norm"):
            loop.divisive_norm = 0.8

        if mode == "recon":
            b = Builder(params=loop, affect=AffectParams(enabled=False))
            net, _ = b.stage_B()
            b.score_weights = {
                "w_valence": 0.0,
                "w_arousal": 0.0,
                "w_ns": 0.0,
                "w_bb_err": 0.0,
                "w_epistemic": 0.0,
            }
        elif mode == "humphrey":
            b = Builder(params=loop, affect=AffectParams(enabled=False))
            net, _ = b.stage_D(efference_threshold=0.05)
        elif mode == "humphrey_barrett":
            aff = AffectParams(
                enabled=True,
                valence_scale=3.0,
                k_homeo=0.10,
                k_pe=0.50,
                demand_motor=0.20,
                demand_stim=0.30,
                modulate_g=True,
                k_g_arousal=0.8,
                k_g_unpleasant=0.8,
                modulate_precision=True,
                precision_base=1.0,
                k_precision_arousal=0.5,
            )
            b = Builder(params=loop, affect=aff)
            net, _ = b.stage_D(efference_threshold=0.05)
        else:
            raise ValueError(mode)

        self.b = b
        self.net = net
        self.score_weights = getattr(self.b, "score_weights", None)
        self.net.start_root(True)

        self.eps = 0.08  # small stochasticity to avoid dead ties
        self.recent = deque(maxlen=30)

        self.log = {k: [] for k in [
            "I_total","I_touch","I_smell","I_vision",
            "Ns","Ne","Nv","Na","alpha",
            "action","touch_count"
        ]}
        self._touch_count = 0

    def _read_state(self):
        st = getattr(self.net, "_ipsundrum_state", {})
        Ns = float(self.net.get("Ns").activation) if "Ns" in self.net.nodes else np.nan
        Ne = float(self.net.get("Ne").activation) if "Ne" in self.net.nodes else np.nan
        Nv = float(self.net.get("Nv").activation) if "Nv" in self.net.nodes else np.nan
        Na = float(self.net.get("Na").activation) if "Na" in self.net.nodes else np.nan
        alpha = float(st.get("alpha_eff", np.nan))
        return Ns, Ne, Nv, Na, alpha, st

    def step(self):
        I_total, I_touch, I_smell, I_vision = compute_I_affect(self.env, self.y, self.x, self.heading)

        if I_touch > 0.5:
            self._touch_count += 1

        # physiology update
        if hasattr(self.net, "_update_ipsundrum_sensor"):
            self.net._update_ipsundrum_sensor(I_total, rng=self.rng)  # type: ignore[attr-defined]
        else:
            self.net.set_sensor_value("Ns", float(np.clip(0.5 + 0.5*I_total, 0.0, 1.0)))

        self.net.step()
        Ns, Ne, Nv, Na, alpha, st = self._read_state()

        # choose action via internal feelings
        if self.rng.random() < self.eps:
            action = self.rng.choice(["forward","turn_left","turn_right"])
        else:
            action = choose_action_feelings(self, horizon=self.horizon)

        # move
        self.y, self.x, self.heading = self.env.step(self.y, self.x, action, self.heading)
        self.recent.append((self.y, self.x))

        # log
        self.log["I_total"].append(I_total)
        self.log["I_touch"].append(I_touch)
        self.log["I_smell"].append(I_smell)
        self.log["I_vision"].append(I_vision)

        self.log["Ns"].append(Ns)
        self.log["Ne"].append(Ne)
        self.log["Nv"].append(Nv)
        self.log["Na"].append(Na)
        self.log["alpha"].append(alpha)
        self.log["action"].append(action)
        self.log["touch_count"].append(self._touch_count)

        return action


def run_animation():
    from experiments.viz_utils.corridor_viz import run_animation as _run_animation

    return _run_animation()


if __name__ == "__main__":
    run_animation()
