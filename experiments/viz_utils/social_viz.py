from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from analysis.social_exact_solver import condition_social_params
from core.envs.social_corridor import SocialCorridorWorld
from core.envs.social_foodshare import FoodShareToy
from core.social_forward import SocialForwardContext, predict_one_step_social
from core.social_homeostat import HomeostatParams, homeostat_from_net_dict
from core.social_model import SocialAgent, SocialAgentConfig
from utils.plot_style import apply_times_style


apply_times_style()


SCORE_WEIGHTS = {
    "w_valence": 2.0,
    "w_arousal": -1.2,
    "w_ns": -0.8,
    "w_bb_err": -0.4,
}


@dataclass
class SocialEpisodeFrame:
    t: int
    action: str
    self_energy: float
    partner_energy: float
    self_distress: float
    partner_distress: float
    distress_coupled: float
    has_food: bool
    transfer_event: bool
    possessor_x: int = 0
    partner_x: int = 0
    food_source_x: int = 0
    hazard_x: int = -1
    condition: str = ""
    task_name: str = ""


def _agent_config(condition: str, lambda_affective: float, *, horizon: int, homeostat: HomeostatParams) -> SocialAgentConfig:
    return SocialAgentConfig(
        model_name=condition,
        homeostat=homeostat,
        social=condition_social_params(condition, lambda_affective),
        horizon=horizon,
        score_weights=dict(SCORE_WEIGHTS),
    )


def simulate_foodshare(condition: str, *, lambda_affective: float = 1.0, seed: int = 0) -> List[SocialEpisodeFrame]:
    env = FoodShareToy(horizon=1)
    env.reset()
    lam = lambda_affective if ("affective" in condition or "full" in condition) else 0.0
    cfg = _agent_config(condition, lam, horizon=1, homeostat=HomeostatParams())
    agent = SocialAgent(env, cfg, seed=seed)
    social_ctx = SocialForwardContext(env_model=env, homeostat_params=agent.config.homeostat, social_params=agent.config.social)
    action = agent.choose_action()
    pred = predict_one_step_social(
        getattr(agent.net, "_ipsundrum_state", {}),
        agent.builder.params,
        agent.builder.affect,
        0.0,
        rng=np.random.default_rng(seed),
        social_ctx=social_ctx,
        action=action,
    )
    self_state = homeostat_from_net_dict(pred, agent.config.homeostat)
    partner_state = homeostat_from_net_dict(pred["partner_state"], agent.partner_params)
    return [
        SocialEpisodeFrame(
            t=0,
            action=action,
            self_energy=self_state.energy_true,
            partner_energy=partner_state.energy_true,
            self_distress=self_state.distress_self,
            partner_distress=partner_state.distress_self,
            distress_coupled=self_state.distress_coupled,
            has_food=True,
            transfer_event=action == "PASS",
            condition=condition,
            task_name="FoodShareToy",
        )
    ]


def simulate_corridor(condition: str, *, lambda_affective: float = 1.0, seed: int = 0, horizon: int = 24) -> List[SocialEpisodeFrame]:
    env = SocialCorridorWorld(horizon=horizon)
    env.reset()
    lam = lambda_affective if ("affective" in condition or "full" in condition) else 0.0
    cfg = _agent_config(condition, lam, horizon=10, homeostat=HomeostatParams())
    agent = SocialAgent(env, cfg, seed=seed)
    frames: List[SocialEpisodeFrame] = []
    for t in range(horizon):
        action = agent.choose_action() if homeostat_from_net_dict(getattr(agent.net, "_ipsundrum_state", {}), agent.config.homeostat).alive else "STAY"
        social_ctx = SocialForwardContext(env_model=env, homeostat_params=agent.config.homeostat, social_params=agent.config.social)
        pred = predict_one_step_social(
            getattr(agent.net, "_ipsundrum_state", {}),
            agent.builder.params,
            agent.builder.affect,
            0.0,
            rng=np.random.default_rng(seed * 1000 + t),
            social_ctx=social_ctx,
            action=action,
        )
        transition = env.predict_transition(getattr(agent.net, "_ipsundrum_state", {}), getattr(agent.net, "_ipsundrum_state", {})["partner_state"], action)
        env.step(action)
        agent.net._ipsundrum_state.clear()  # type: ignore[attr-defined]
        agent.net._ipsundrum_state.update(pred)  # type: ignore[attr-defined]
        agent.x = int(transition.get("next_x", agent.x))
        agent.has_food = bool(transition.get("has_food_next", agent.has_food))
        agent.net._ipsundrum_state["x"] = float(agent.x)  # type: ignore[attr-defined]
        agent.net._ipsundrum_state["has_food"] = 1.0 if agent.has_food else 0.0  # type: ignore[attr-defined]
        self_state = homeostat_from_net_dict(pred, agent.config.homeostat)
        partner_state = homeostat_from_net_dict(pred["partner_state"], agent.partner_params)
        frames.append(
            SocialEpisodeFrame(
                t=t,
                action=action,
                self_energy=self_state.energy_true,
                partner_energy=partner_state.energy_true,
                self_distress=self_state.distress_self,
                partner_distress=partner_state.distress_self,
                distress_coupled=self_state.distress_coupled,
                has_food=bool(agent.has_food),
                transfer_event=bool(transition.get("transfer_event", 0.0) > 0.0),
                possessor_x=agent.x,
                partner_x=env.partner_x,
                food_source_x=env.food_source_x,
                hazard_x=env.hazard_x,
                condition=condition,
                task_name="SocialCorridorWorld",
            )
        )
    return frames


def _energy_panel(ax, frame: SocialEpisodeFrame) -> None:
    ax.clear()
    ax.set_title(f"t={frame.t}  action={frame.action}", fontsize=11)
    vals = [frame.self_energy, frame.partner_energy]
    labels = ["Possessor", "Partner"]
    colors = ["#0f4c81", "#c84c09"]
    ax.bar(labels, vals, color=colors, width=0.6)
    ax.axhline(0.70, color="#444444", linestyle="--", linewidth=1)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Energy")
    ax.text(0.02, 0.95, f"coupled distress={frame.distress_coupled:.2f}", transform=ax.transAxes, va="top", fontsize=10)
    ax.text(0.02, 0.87, f"self distress={frame.self_distress:.2f}  partner distress={frame.partner_distress:.2f}", transform=ax.transAxes, va="top", fontsize=10)


def _draw_foodshare_world(ax, frame: SocialEpisodeFrame) -> None:
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    poss = Circle((2.4, 3.0), 0.65, color="#0f4c81")
    part = Circle((7.6, 3.0), 0.65, color="#c84c09")
    ax.add_patch(poss)
    ax.add_patch(part)
    ax.text(2.4, 1.9, "Possessor", ha="center", fontsize=10)
    ax.text(7.6, 1.9, "Partner", ha="center", fontsize=10)
    ax.text(2.4, 3.0, f"{frame.self_energy:.2f}", ha="center", va="center", color="white", fontsize=11)
    ax.text(7.6, 3.0, f"{frame.partner_energy:.2f}", ha="center", va="center", color="white", fontsize=11)
    food_x = 5.0 if frame.transfer_event else 2.4
    ax.add_patch(Rectangle((food_x - 0.2, 4.0), 0.4, 0.4, color="#d4a017"))
    if frame.transfer_event:
        ax.add_patch(FancyArrowPatch((3.2, 4.2), (6.8, 4.2), arrowstyle="->", mutation_scale=16, linewidth=2, color="#d4a017"))
    ax.text(5.0, 5.0, "PASS" if frame.transfer_event else "EAT", ha="center", fontsize=12, weight="bold")


def _draw_corridor_world(ax, frame: SocialEpisodeFrame, length: int = 7) -> None:
    ax.clear()
    ax.set_xlim(-0.5, length - 0.5)
    ax.set_ylim(-0.4, 1.4)
    ax.set_yticks([])
    ax.set_xticks(range(length))
    for x in range(length):
        color = "#f3efe6"
        if x == frame.hazard_x:
            color = "#f6c5b8"
        if x == frame.food_source_x:
            color = "#d9ead3"
        ax.add_patch(Rectangle((x - 0.5, -0.1), 1.0, 0.8, facecolor=color, edgecolor="#999999"))
    ax.text(frame.food_source_x, -0.22, "food", ha="center", fontsize=9)
    ax.text(frame.hazard_x, -0.22, "hazard", ha="center", fontsize=9)
    ax.add_patch(Circle((frame.partner_x, 0.28), 0.16, color="#c84c09"))
    ax.add_patch(Circle((frame.possessor_x, 0.28), 0.16, color="#0f4c81"))
    if frame.has_food:
        ax.add_patch(Rectangle((frame.possessor_x - 0.08, 0.58), 0.16, 0.14, color="#d4a017"))
    if frame.transfer_event:
        ax.add_patch(FancyArrowPatch((frame.possessor_x, 0.68), (frame.partner_x, 0.68), arrowstyle="->", mutation_scale=16, linewidth=2, color="#d4a017"))
    ax.text(0.02, 0.95, f"action={frame.action}", transform=ax.transAxes, va="top", fontsize=10)
    ax.text(0.02, 0.84, f"self={frame.self_energy:.2f}  partner={frame.partner_energy:.2f}", transform=ax.transAxes, va="top", fontsize=10)


def save_foodshare_compare_gif(out_path: str, *, fps: int = 1) -> None:
    left = simulate_foodshare("social_none", seed=0)
    right = simulate_foodshare("social_affective_direct", seed=0)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={"height_ratios": [2.0, 1.3]})

    def update(_frame_idx: int):
        _draw_foodshare_world(axes[0, 0], left[0])
        _draw_foodshare_world(axes[0, 1], right[0])
        _energy_panel(axes[1, 0], left[0])
        _energy_panel(axes[1, 1], right[0])
        axes[0, 0].set_title("self-only baseline", fontsize=12)
        axes[0, 1].set_title("affective coupling", fontsize=12)
        fig.suptitle("FoodShareToy: helping appears only when partner distress is coupled", fontsize=14)
        return []

    ani = FuncAnimation(fig, update, frames=1, interval=1000, blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def save_corridor_compare_gif(out_path: str, *, fps: int = 2, horizon: int = 24, final_hold: int = 6) -> None:
    left = simulate_corridor("social_none", seed=0, horizon=horizon)
    right = simulate_corridor("social_affective_direct", seed=0, horizon=horizon)
    n = max(len(left), len(right)) + final_hold
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), gridspec_kw={"height_ratios": [1.7, 1.3]})

    def update(frame_idx: int):
        li = left[min(frame_idx, len(left) - 1)]
        ri = right[min(frame_idx, len(right) - 1)]
        _draw_corridor_world(axes[0, 0], li)
        _draw_corridor_world(axes[0, 1], ri)
        _energy_panel(axes[1, 0], li)
        _energy_panel(axes[1, 1], ri)
        axes[0, 0].set_title("self-only baseline", fontsize=12)
        axes[0, 1].set_title("affective coupling", fontsize=12)
        fig.suptitle("SocialCorridorWorld: coupled agent fetches, carries, and passes food", fontsize=14)
        return []

    ani = FuncAnimation(fig, update, frames=n, interval=500, blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def build_social_gifs(outdir: str = "paper-site/static/media") -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    food = out / "social-foodshare-compare.gif"
    corridor = out / "social-corridor-compare.gif"
    save_foodshare_compare_gif(str(food))
    save_corridor_compare_gif(str(corridor))
    return {"foodshare": str(food), "corridor": str(corridor)}


if __name__ == "__main__":
    build_social_gifs()
