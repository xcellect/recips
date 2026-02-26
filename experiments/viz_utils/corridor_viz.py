import matplotlib as mpl
from IPython.display import HTML
mpl.rcParams["animation.html"] = "jshtml"

import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Circle

from utils.plot_style import apply_times_style


apply_times_style()


def run_animation(horizon: int = 6):
    from experiments.corridor_exp import CorridorWorld, Agent

    env = CorridorWorld(H=18, W=18, seed=0)

    start = (1, env.goal_x)  # near top center
    agents = [
        Agent(env, mode="recon",    seed=0, start=start, heading=2, horizon=horizon),
        Agent(env, mode="humphrey", seed=1, start=start, heading=2, horizon=horizon),
        Agent(env, mode="humphrey_barrett", seed=2, start=start, heading=2, horizon=horizon),
    ]
    titles = ["Recon baseline (Stage B)", "Ipsundrum", "Ipsundrum+affect"]

    # Display field: beauty - hazard, with walls set darker for visibility
    bg = env.beauty - env.hazard
    bg_disp = bg.copy()
    bg_disp[env.blocked] = -1.2

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 3, height_ratios=[3.0, 1.4, 1.4], hspace=0.35, wspace=0.30, top=0.95, bottom=0.08)

    ax_env = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_ns  = fig.add_subplot(gs[1, :2])
    ax_hud = fig.add_subplot(gs[2, :2])

    ims, dots, cones, smell_circles = [], [], [], []
    goal_marks = []
    info_texts = []

    def make_cone_patch(ax):
        p = Polygon([[0, 0], [0, 0], [0, 0]], closed=True, alpha=0.18)
        ax.add_patch(p)
        return p

    def make_smell_circle(ax, r=3.0):
        c = Circle((0, 0), r, fill=False, alpha=0.4)
        ax.add_patch(c)
        return c

    for i, ax in enumerate(ax_env):
        im = ax.imshow(bg_disp, vmin=-1.2, vmax=1.0, extent=[-0.5, env.W-0.5, env.H-0.5, -0.5])
        ims.append(im)

        dot, = ax.plot([], [], marker="o", markersize=9)
        dots.append(dot)

        cones.append(make_cone_patch(ax))
        smell_circles.append(make_smell_circle(ax, r=3.0))

        # goal marker
        gmark, = ax.plot([env.goal_x], [env.goal_y], marker="*", markersize=12)
        goal_marks.append(gmark)

        ax.set_title(titles[i], fontsize=10, pad=5)
        ax.set_xlim(-0.5, env.W-0.5)
        ax.set_ylim(env.H-0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

        info_text = ax.text(
            0.5,
            -0.06,
            "",
            fontsize=8,
            ha="center",
            va="top",
            transform=ax.transAxes,
        )
        info_texts.append(info_text)

    # HUD: Ns traces
    ax_ns.set_title("Ns(t): persistence differs across models")
    ax_ns.set_xlim(0, 200)
    ax_ns.set_ylim(0, 1.05)
    lines_ns = [ax_ns.plot([], [], label=titles[i])[0] for i in range(3)]
    ax_ns.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    # HUD: affect model internals
    ax_hud.set_title("Ipsundrum+affect feelings & regime: Nv/Na/alpha")
    ax_hud.set_xlim(0, 200)
    ax_hud.set_ylim(0, 2.5)
    line_nv, = ax_hud.plot([], [], label="Nv (valence)")
    line_na, = ax_hud.plot([], [], label="Na (arousal)")
    line_alpha, = ax_hud.plot([], [], label="alpha_eff")
    ax_hud.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    def cone_polygon(x, y, heading, radius=5, fov_deg=70):
        dy0, dx0 = [(-1, 0), (0, 1), (1, 0), (0, -1)][heading]
        ang0 = math.atan2(dy0, dx0)
        fov = math.radians(fov_deg)
        a1 = ang0 - 0.5 * fov
        a2 = ang0 + 0.5 * fov
        pts = [(x, y)]
        for aa in np.linspace(a1, a2, 9):
            pts.append((x + radius * math.cos(aa), y + radius * math.sin(aa)))
        return pts

    tmax = 220

    def update(frame):
        for i, ag in enumerate(agents):
            action = ag.step()

            dots[i].set_data([ag.x], [ag.y])
            cones[i].set_xy(cone_polygon(ag.x, ag.y, ag.heading, radius=5, fov_deg=70))
            smell_circles[i].center = (ag.x, ag.y)

            ns = ag.log["Ns"][-1]
            i_t = ag.log["I_touch"][-1]
            i_s = ag.log["I_smell"][-1]
            i_v = ag.log["I_vision"][-1]
            tc = ag.log["touch_count"][-1]
            info_texts[i].set_text(
                f"act={action}  Ns={ns:.2f}  touch={i_t:.2f} (hits={tc})\n"
                f"smell={i_s:.2f}  vision={i_v:.2f}  heading={ag.heading}"
            )

        # Ns traces
        xs = np.arange(len(agents[0].log["Ns"]))
        for i in range(3):
            ys = np.asarray(agents[i].log["Ns"], dtype=float)
            lines_ns[i].set_data(xs, ys)
        ax_ns.set_xlim(max(0, len(xs) - 200), max(200, len(xs)))

        # Affect HUD (agent 2)
        full = agents[2]
        xs2 = np.arange(len(full.log["Ns"]))
        nv = np.asarray(full.log["Nv"], dtype=float)
        na = np.asarray(full.log["Na"], dtype=float)
        al = np.asarray(full.log["alpha"], dtype=float)

        line_nv.set_data(xs2, nv)
        line_na.set_data(xs2, na)
        line_alpha.set_data(xs2, al)
        ax_hud.set_xlim(max(0, len(xs2) - 200), max(200, len(xs2)))

        return (*dots, *cones, *lines_ns, line_nv, line_na, line_alpha)

    ani = FuncAnimation(fig, update, frames=tmax, interval=120, blit=False)
    return HTML(ani.to_jshtml())


if __name__ == "__main__":
    run_animation()
