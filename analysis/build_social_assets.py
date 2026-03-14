"""Build a Twitter-ready social asset pack from current paper artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "results" / "social-assets" / "twitter-thread"
CLAIMS_PATH = ROOT / "results" / "paper" / "claims.json"
FAM_PATH = ROOT / "results" / "familiarity" / "episodes_improved.csv"
PLAY_TRACE_PATH = ROOT / "results" / "exploratory-play" / "exploratory_play_clarified_trace_paper.csv"
PAIN_SUMMARY_PATH = ROOT / "results" / "pain-tail" / "summary.csv"
LESION_TRACES_PATH = ROOT / "results" / "lesion" / "ns_traces.npz"

COLORS = {
    "Recon": "#1f77b4",
    "ReCoN": "#1f77b4",
    "Ipsundrum": "#ff7f0e",
    "Ipsundrum+affect": "#2ca02c",
}
INK = "#0f172a"
SUBTLE = "#475569"
GRID = "#d8dee9"
PAPER = "#fcfcfb"
PANEL = "#f6f8fb"
ACCENT = "#d62728"
BLUE = "#1f77b4"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": SUBTLE,
            "ytick.color": SUBTLE,
        }
    )


def ensure_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_claims() -> dict:
    with CLAIMS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def claim(claims: dict, claim_id: str) -> float:
    return float(claims[claim_id]["value"])


def rounded_box(
    ax,
    xy: Tuple[float, float],
    width: float,
    height: float,
    *,
    fc: str = "white",
    ec: str = GRID,
    lw: float = 1.5,
    radius: float = 0.03,
    alpha: float = 1.0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def pill(ax, x: float, y: float, text: str, color: str, *, fontsize: int = 16) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.25", fc=color, ec=color),
    )


def label_box(ax, x: float, y: float, text: str, *, fc: str = "white", ec: str = GRID, color: str = INK, fontsize: int = 12) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.15", fc=fc, ec=ec),
    )


def draw_callout_stack(
    ax,
    *,
    box_y: float,
    box_h: float,
    title: str,
    rows: List[dict],
    fc: str,
    ec: str,
    box_x: float = 0.02,
    box_w: float = 0.96,
    top_pad: float = 0.07,
    bottom_pad: float = 0.07,
) -> None:
    rounded_box(ax, (box_x, box_y), box_w, box_h, fc=fc, ec=ec, lw=1.5, radius=0.03)
    ys = np.linspace(box_y + box_h - top_pad, box_y + bottom_pad, len(rows) + 1)
    ax.text(0.50, ys[0], title, ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
    for y, row in zip(ys[1:], rows):
        kwargs = {
            "fontsize": row.get("fontsize", 12.5),
            "color": row.get("color", SUBTLE),
            "fontweight": row.get("fontweight", "normal"),
            "linespacing": row.get("linespacing", 1.15),
            "ha": "center",
            "va": "center",
        }
        ax.text(0.50, y, row["text"], **kwargs)


def draw_checkerboard(ax, width: int, height: int) -> None:
    checker = (np.indices((height, width)).sum(axis=0) % 2).astype(float)
    ax.imshow(
        checker,
        cmap=matplotlib.colors.ListedColormap(["#f7fafc", "#eef3f8"]),
        interpolation="nearest",
        extent=[-0.5, width - 0.5, height - 0.5, -0.5],
        zorder=0,
    )
    for pos in np.arange(-0.5, width, 1.0):
        ax.axvline(pos, color="#e6ecf2", linewidth=0.6, zorder=1)
    for pos in np.arange(-0.5, height, 1.0):
        ax.axhline(pos, color="#e6ecf2", linewidth=0.6, zorder=1)


def corridor_agents(horizon: int = 6):
    from experiments.corridor_exp import Agent, CorridorWorld

    env = CorridorWorld(H=18, W=18, seed=0)
    start = (1, env.goal_x)
    agents = [
        Agent(env, mode="recon", seed=0, start=start, heading=2, horizon=horizon),
        Agent(env, mode="humphrey", seed=1, start=start, heading=2, horizon=horizon),
        Agent(env, mode="humphrey_barrett", seed=2, start=start, heading=2, horizon=horizon),
    ]
    return env, agents


def cone_polygon(x: float, y: float, heading: int, radius: float = 4.5, fov_deg: float = 72.0) -> List[Tuple[float, float]]:
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    dy0, dx0 = dirs[heading]
    ang0 = np.arctan2(dy0, dx0)
    fov = np.radians(fov_deg)
    a1 = ang0 - 0.5 * fov
    a2 = ang0 + 0.5 * fov
    pts = [(x, y)]
    for aa in np.linspace(a1, a2, 9):
        pts.append((x + radius * np.cos(aa), y + radius * np.sin(aa)))
    return pts


def build_hero_gif(out_path: Path, *, frames: int = 120, fps: int = 10) -> None:
    env, agents = corridor_agents(horizon=6)
    titles = ["Recon", "Ipsundrum", "Ipsundrum+affect"]
    bg = env.beauty - env.hazard
    bg_disp = bg.copy()
    bg_disp[env.blocked] = -1.2

    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    gs = fig.add_gridspec(1, 3, left=0.05, right=0.95, top=0.82, bottom=0.17, wspace=0.16)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    dots = []
    trails = []
    cones = []
    banners = []
    goal_marks = []
    positions = [[] for _ in agents]
    footer = fig.text(
        0.5,
        0.08,
        "recurrence -> persistence\naffect-coupled control -> stable preference + caution",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=INK,
        alpha=0.0,
        bbox=dict(boxstyle="round,pad=0.6,rounding_size=0.2", fc="white", ec=GRID, lw=1.5),
    )
    footer.get_bbox_patch().set_alpha(0.0)
    fig.text(0.5, 0.93, "Same task. Different internals.", ha="center", fontsize=26, fontweight="bold")
    fig.text(
        0.5,
        0.892,
        "ReCoN / Ipsundrum / Ipsundrum+affect in the same corridor rollout",
        ha="center",
        fontsize=14,
        color=SUBTLE,
    )
    fig.text(0.5, 0.12, "same task", ha="center", fontsize=13, color=SUBTLE)
    fig.text(0.5, 0.096, "different internal loop", ha="center", fontsize=13, color=SUBTLE)

    for ax, title in zip(axes, titles):
        ax.imshow(bg_disp, vmin=-1.2, vmax=1.0, extent=[-0.5, env.W - 0.5, env.H - 0.5, -0.5], zorder=0)
        ax.scatter([env.goal_x], [env.goal_y], marker="*", s=260, color="#f5c542", edgecolor="#916000", linewidth=1.1, zorder=5)
        ax.set_xlim(-0.5, env.W - 0.5)
        ax.set_ylim(env.H - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        banner = ax.text(
            0.5,
            1.06,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.38,rounding_size=0.18", fc=COLORS[title], ec=COLORS[title]),
        )
        banners.append(banner)
        trail, = ax.plot([], [], color=COLORS[title], linewidth=3.0, alpha=0.85, zorder=3)
        dot, = ax.plot([], [], marker="o", markersize=11, color=COLORS[title], markeredgecolor="white", markeredgewidth=2, zorder=4)
        cone = Polygon([[0, 0], [0, 0], [0, 0]], closed=True, facecolor=COLORS[title], edgecolor="none", alpha=0.18, zorder=2)
        ax.add_patch(cone)
        trails.append(trail)
        dots.append(dot)
        cones.append(cone)

    def update(frame_idx: int):
        for idx, agent in enumerate(agents):
            agent.step()
            positions[idx].append((agent.x, agent.y))
            tail = positions[idx][-16:]
            xs = [p[0] for p in tail]
            ys = [p[1] for p in tail]
            trails[idx].set_data(xs, ys)
            dots[idx].set_data([agent.x], [agent.y])
            cones[idx].set_xy(cone_polygon(agent.x, agent.y, agent.heading))

        if frame_idx >= frames - 28:
            alpha = min(1.0, (frame_idx - (frames - 28)) / 10.0)
        else:
            alpha = 0.0
        footer.set_alpha(alpha)
        footer.get_bbox_patch().set_alpha(0.92 * alpha)
        return [*trails, *dots, *cones, footer]

    animation = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    animation.save(out_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def build_architecture_card(out_path: Path) -> None:
    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.05, 0.92, "Same substrate. Three interventions.", fontsize=27, fontweight="bold")
    ax.text(
        0.05,
        0.875,
        "Start from an inspectable ReCoN agent, add recurrence, add affect,\nthen lesion feedback + integration at t = 3.",
        fontsize=13.5,
        color=SUBTLE,
        va="top",
    )

    y0 = 0.215
    height = 0.545
    margin_x = 0.04
    gap_x = 0.055
    width = (1 - 2 * margin_x - 2 * gap_x) / 3
    xs = [margin_x + i * (width + gap_x) for i in range(3)]
    titles = ["ReCoN", "Ipsundrum", "Ipsundrum+affect"]
    subtitles = [
        "Transparent\nsensorimotor script",
        "Add recurrent persistence\naround salience",
        "Add valence / arousal\ninto control",
    ]

    def block(x: float, y: float, w: float, h: float, text: str, fc: str = "white", ec: str = GRID) -> None:
        rounded_box(ax, (x, y), w, h, fc=fc, ec=ec, lw=1.3, radius=0.02)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12, fontweight="bold", color=INK)

    def draw_stage(x0: float, title: str, subtitle: str) -> dict:
        rounded_box(ax, (x0, y0), width, height, fc=PANEL, ec="#d6dde6", lw=1.8)
        pill(ax, x0 + width / 2, y0 + height - 0.055, title, COLORS[title], fontsize=16)
        ax.text(
            x0 + width / 2,
            y0 + height - 0.122,
            subtitle,
            ha="center",
            va="center",
            fontsize=11.5,
            color=SUBTLE,
            linespacing=1.28,
        )
        inner_x = x0 + 0.03
        inner_w = width - 0.06
        content_top = y0 + height - 0.180
        box_h = 0.060
        gap_y = 0.030
        slots = [content_top - box_h - i * (box_h + gap_y) for i in range(4)]
        return {
            "x0": x0,
            "inner_x": inner_x,
            "inner_w": inner_w,
            "center_x": x0 + width / 2,
            "box_h": box_h,
            "gap_y": gap_y,
            "slots": slots,
        }

    def stack_boxes(stage: dict, texts: List[str], *, start_slot: int = 0, fills: List[str] | None = None, edges: List[str] | None = None) -> List[Tuple[float, float, float, float]]:
        fills = fills or ["white"] * len(texts)
        edges = edges or [GRID] * len(texts)
        rects = []
        for idx, (text, fc, ec) in enumerate(zip(texts, fills, edges)):
            y = stage["slots"][start_slot + idx]
            rect = (stage["inner_x"], y, stage["inner_w"], stage["box_h"])
            rects.append(rect)
            block(*rect, text, fc=fc, ec=ec)
        return rects

    def arrow_between(upper: Tuple[float, float, float, float], lower: Tuple[float, float, float, float], *, color: str = INK, lw: float = 2.0) -> None:
        x = upper[0] + upper[2] / 2
        ax.add_patch(
            FancyArrowPatch(
                (x, upper[1]),
                (x, lower[1] + lower[3]),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=lw,
                color=color,
            )
        )

    stages = [draw_stage(x0, title, subtitle) for x0, title, subtitle in zip(xs, titles, subtitles)]

    # ReCoN
    recon = stages[0]
    recon_rects = stack_boxes(
        recon,
        ["Sensors", "Inspectable\nstate machine", "Action choice"],
    )
    arrow_between(recon_rects[0], recon_rects[1])
    arrow_between(recon_rects[1], recon_rects[2])

    # Ipsundrum
    ips = stages[1]
    ips_rects = stack_boxes(
        ips,
        ["Sensors", "Salience  Ns", "Planner / control"],
    )
    arrow_between(ips_rects[0], ips_rects[1])
    arrow_between(ips_rects[1], ips_rects[2])
    ax.add_patch(
        FancyArrowPatch(
            (ips["x0"] + width - 0.04, ips_rects[2][1] + ips_rects[2][3] * 0.36),
            (ips["x0"] + width - 0.04, ips_rects[1][1] + ips_rects[1][3] * 0.54),
            connectionstyle="arc3,rad=0.92",
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.8,
            color=COLORS["Ipsundrum"],
        )
    )

    # Ipsundrum + affect
    affect = stages[2]
    affect_rects = stack_boxes(
        affect,
        ["Sensors", "Salience  Ns", "Planner / control", "Valence /\narousal"],
        fills=["white", "white", "white", "#eff8ef"],
        edges=[GRID, GRID, GRID, "#cfe6cf"],
    )
    arrow_between(affect_rects[0], affect_rects[1])
    arrow_between(affect_rects[1], affect_rects[2])
    ax.add_patch(
        FancyArrowPatch(
            (affect["center_x"], affect_rects[3][1] + affect_rects[3][3]),
            (affect["center_x"], affect_rects[2][1]),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.3,
            color=COLORS["Ipsundrum+affect"],
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (affect["x0"] + width - 0.04, affect_rects[2][1] + affect_rects[2][3] * 0.58),
            (affect["x0"] + width - 0.04, affect_rects[1][1] + affect_rects[1][3] * 0.56),
            connectionstyle="arc3,rad=0.90",
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.8,
            color=COLORS["Ipsundrum+affect"],
        )
    )

    # Across-stage arrows
    bridge_y = recon_rects[1][1] + recon_rects[1][3] / 2
    ax.add_patch(
        FancyArrowPatch(
            (xs[0] + width + 0.01, bridge_y),
            (xs[1] - 0.01, bridge_y),
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=2.6,
            color="#94a3b8",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (xs[1] + width + 0.01, bridge_y),
            (xs[2] - 0.01, bridge_y),
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=2.6,
            color="#94a3b8",
        )
    )

    # Lesion callout
    rounded_box(ax, (0.275, 0.055), 0.45, 0.085, fc="#fff5f5", ec="#f1b8b8", lw=1.8, radius=0.02)
    ax.text(0.50, 0.097, "Lesion @ t = 3: cut feedback + integration", ha="center", va="center", fontsize=14, fontweight="bold", color=ACCENT)
    ax.add_patch(
        FancyArrowPatch(
            (0.44, 0.14),
            (ips["x0"] + width - 0.06, ips_rects[2][1] + ips_rects[2][3] * 0.5),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            linestyle="--",
            color=ACCENT,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.57, 0.14),
            (affect["center_x"], affect_rects[3][1] + affect_rects[3][3] + 0.012),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            linestyle="--",
            color=ACCENT,
        )
    )

    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def build_novelty_card(out_path: Path, claims: dict) -> None:
    models = ["ReCoN", "Ipsundrum", "Ipsundrum+affect"]
    values = [
        claim(claims, "fam_delta_scenic_entry_recon"),
        claim(claims, "fam_delta_scenic_entry_humphrey"),
        claim(claims, "fam_delta_scenic_entry_hb"),
    ]
    scenic_less_novel = claim(claims, "fam_median_delta_novelty_hb_scenic")

    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    ax = fig.add_axes([0.08, 0.20, 0.52, 0.55])
    note_ax = fig.add_axes([0.65, 0.18, 0.28, 0.56])
    note_ax.set_axis_off()

    fig.text(0.08, 0.90, "Novelty seeking is not preference.", fontsize=29, fontweight="bold")
    fig.text(0.08, 0.855, "After familiarity control, only Ipsundrum+affect keeps choosing scenic.", fontsize=14, color=SUBTLE)
    fig.text(
        0.08,
        0.823,
        "Δ scenic-entry = P(scenic | dull familiar) - P(scenic | scenic familiar)",
        fontsize=12.5,
        color=SUBTLE,
    )

    x = np.arange(len(models))
    bars = ax.bar(x, values, color=[COLORS[m] for m in models], width=0.58)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=14)
    ax.set_ylabel("Δ scenic-entry", fontsize=14)
    ax.set_ylim(0, 0.085)
    ax.set_yticks(np.linspace(0, 0.08, 5))
    ax.grid(axis="y", color="#e3e8ef", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.003,
            f"+{value:.02f}",
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color=INK,
        )

    top_box_y = 0.55
    top_box_h = 0.31
    draw_callout_stack(
        note_ax,
        box_y=top_box_y,
        box_h=top_box_h,
        title="What changes?",
        rows=[
            {"text": "ReCoN + Ipsundrum move with novelty.", "fontsize": 12.8, "color": SUBTLE},
            {"text": "Ipsundrum+affect barely shifts.", "fontsize": 12.8, "color": COLORS["Ipsundrum+affect"], "fontweight": "bold"},
        ],
        fc="white",
        ec="#dce4ee",
    )

    bottom_box_y = 0.15
    bottom_box_h = 0.30
    draw_callout_stack(
        note_ax,
        box_y=bottom_box_y,
        box_h=bottom_box_h,
        title="Scenic was less novel.",
        rows=[
            {"text": f"median Δ novelty = {scenic_less_novel:.02f}", "fontsize": 13.3, "color": INK},
            {"text": "Still chose scenic\ndespite lower novelty.", "fontsize": 12.6, "color": SUBTLE, "linespacing": 1.15},
        ],
        fc="#f0f9f0",
        ec="#cfe6cf",
    )

    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def extract_play_runs() -> Dict[str, pd.DataFrame]:
    trace = pd.read_csv(PLAY_TRACE_PATH)
    selected = {
        "ReCoN": ("Recon_curiosity", 0),
        "Ipsundrum": ("Ipsundrum_curiosity", 0),
        "Ipsundrum+affect": ("Ipsundrum_Affect_curiosity", 0),
    }
    runs: Dict[str, pd.DataFrame] = {}
    for label, (model_name, seed) in selected.items():
        sub = trace[(trace["model"] == model_name) & (trace["seed"] == seed)].copy()
        sub = sub.sort_values("step")
        runs[label] = sub
    return runs


def scan_positions(run: pd.DataFrame) -> Iterable[Tuple[int, int, int]]:
    prev_x = run["x"].shift(1)
    prev_y = run["y"].shift(1)
    turning = run["action"].isin(["turn_left", "turn_right"])
    inplace = turning & (run["x"] == prev_x) & (run["y"] == prev_y)
    scans = run[inplace]
    if scans.empty:
        return []
    counts = scans.groupby(["x", "y"], observed=False).size().reset_index(name="count")
    return list(counts.itertuples(index=False, name=None))


def build_play_card(out_path: Path, claims: dict) -> None:
    runs = extract_play_runs()
    w = int(max(run["x"].max() for run in runs.values()) + 1)
    h = int(max(run["y"].max() for run in runs.values()) + 1)

    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    gs = fig.add_gridspec(1, 3, left=0.05, right=0.77, top=0.78, bottom=0.14, wspace=0.15)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    badge_ax = fig.add_axes([0.79, 0.14, 0.19, 0.64])
    badge_ax.set_axis_off()

    fig.text(0.05, 0.90, "Same arena. Very different play style.", fontsize=29, fontweight="bold")
    fig.text(0.05, 0.855, "The affect-coupled agent doesn't just wander; it scans.", fontsize=14, color=SUBTLE)

    for ax, (label, run) in zip(axes, runs.items()):
        draw_checkerboard(ax, w, h)
        xs = run["x"].to_numpy()
        ys = run["y"].to_numpy()
        ax.plot(xs, ys, color=COLORS[label], linewidth=2.5, alpha=0.9, zorder=3)
        ax.scatter(xs[0], ys[0], s=90, color="white", edgecolor=INK, linewidth=1.8, zorder=4)
        ax.scatter(xs[-1], ys[-1], s=110, marker="X", color=ACCENT, edgecolor="white", linewidth=1.0, zorder=4)

        scans = list(scan_positions(run))
        if scans:
            scan_x = [x for x, _, _ in scans]
            scan_y = [y for _, y, _ in scans]
            scan_s = [20 + 16 * count for _, _, count in scans]
            ax.scatter(scan_x, scan_y, s=scan_s, facecolor="white", edgecolor=COLORS[label], linewidth=1.8, alpha=0.95, zorder=5)

        ax.set_title(label, fontsize=17, fontweight="bold", color=COLORS[label], pad=12)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    stat_box_y = 0.52
    stat_box_h = 0.34
    draw_callout_stack(
        badge_ax,
        box_y=stat_box_y,
        box_h=stat_box_h,
        title="scan events",
        rows=[
            {"text": f"{claim(claims, 'play_scan_events_hb'):.1f}", "fontsize": 31, "color": COLORS["Ipsundrum+affect"], "fontweight": "bold"},
            {"text": f"vs {claim(claims, 'play_scan_events_recon'):.1f} in ReCoN", "fontsize": 12.8, "color": SUBTLE},
            {"text": "Returns to local scan pockets.", "fontsize": 12.3, "color": SUBTLE},
        ],
        fc="white",
        ec="#dce4ee",
        top_pad=0.065,
        bottom_pad=0.065,
    )

    cycle_box_y = 0.08
    cycle_box_h = 0.34
    draw_callout_stack(
        badge_ax,
        box_y=cycle_box_y,
        box_h=cycle_box_h,
        title="structured local\ninvestigation",
        rows=[
            {"text": f"cycle score = {claim(claims, 'play_cycle_score_hb'):.1f}", "fontsize": 13.3, "color": INK},
            {"text": "Turn-in-place\nscan pockets visible.", "fontsize": 12.0, "color": SUBTLE, "linespacing": 1.15},
        ],
        fc="#f0f9f0",
        ec="#cfe6cf",
    )

    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def build_pain_card(out_path: Path, claims: dict) -> None:
    summary = pd.read_csv(PAIN_SUMMARY_PATH)
    model_ids = ["Recon", "Ipsundrum", "Ipsundrum+affect"]
    labels = ["ReCoN", "Ipsundrum", "Ipsundrum+affect"]
    values = [float(summary.loc[summary["model"] == model, "mean_tail_duration"].iloc[0]) for model in model_ids]

    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    ax = fig.add_axes([0.12, 0.22, 0.50, 0.50])
    note_ax = fig.add_axes([0.67, 0.14, 0.27, 0.64])
    note_ax.set_axis_off()

    fig.text(0.08, 0.90, "Persistence alone isn't enough.", fontsize=29, fontweight="bold")
    fig.text(
        0.08,
        0.855,
        "Only affect-coupled control turns persistence into lingering caution after the hazard is gone.",
        fontsize=14,
        color=SUBTLE,
    )

    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=[COLORS[m] for m in labels], height=0.56)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13.5)
    ax.tick_params(axis="y", pad=2)
    ax.set_xlabel("Planned caution tail duration (steps)", fontsize=14)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", color="#e3e8ef", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    for bar, value in zip(bars, values):
        ax.text(
            value + 2.0,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}",
            va="center",
            fontsize=20,
            fontweight="bold",
            color=INK,
        )

    tail_box_y = 0.58
    tail_box_h = 0.26
    draw_callout_stack(
        note_ax,
        box_y=tail_box_y,
        box_h=tail_box_h,
        title="tail duration",
        rows=[
            {"text": f"{claim(claims, 'pain_tail_duration_hb'):.0f} vs {claim(claims, 'pain_tail_duration_recon'):.0f}", "fontsize": 29, "color": COLORS["Ipsundrum+affect"], "fontweight": "bold"},
            {"text": "Hazard already removed.", "fontsize": 12.4, "color": SUBTLE},
        ],
        fc="#f0f9f0",
        ec="#cfe6cf",
        top_pad=0.06,
        bottom_pad=0.06,
    )

    why_box_y = 0.16
    why_box_h = 0.34
    draw_callout_stack(
        note_ax,
        box_y=why_box_y,
        box_h=why_box_h,
        title="Why this matters",
        rows=[
            {"text": "Ipsundrum shows internal persistence.", "fontsize": 13.0, "color": SUBTLE},
            {"text": "Behavior snaps back fast.\nAffect coupling keeps caution lingering.", "fontsize": 12.2, "color": COLORS["Ipsundrum+affect"], "fontweight": "bold", "linespacing": 1.1},
        ],
        fc="white",
        ec="#dce4ee",
    )

    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def lesion_mean_traces(npz_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    traces = np.load(npz_path)
    grouped: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for key in traces.files:
        model_name, rest = key.split("_seed", 1)
        _, condition = rest.split("_", 1)
        grouped.setdefault(model_name, {}).setdefault(condition, []).append(np.asarray(traces[key], dtype=float))

    means: Dict[str, Dict[str, np.ndarray]] = {}
    for model_name, cond_map in grouped.items():
        means[model_name] = {}
        for condition, arrays in cond_map.items():
            means[model_name][condition] = np.vstack(arrays).mean(axis=0)
    return means


def build_lesion_gif(out_path: Path, claims: dict, *, lesion_t: int = 3, show_until: int = 38, fps: int = 10) -> None:
    means = lesion_mean_traces(LESION_TRACES_PATH)
    sham = means["Ipsundrum+affect"]["sham"][:show_until]
    lesion = means["Ipsundrum+affect"]["lesion"][:show_until]
    x = np.arange(show_until)
    auc_drop_pct = [
        claim(claims, "lesion_auc_drop_recon"),
        claim(claims, "lesion_auc_drop_pct_humphrey"),
        claim(claims, "lesion_auc_drop_pct_hb"),
    ]

    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    ax_line = fig.add_axes([0.08, 0.20, 0.50, 0.56])
    ax_bar = fig.add_axes([0.65, 0.30, 0.25, 0.34])
    ax_note = fig.add_axes([0.64, 0.10, 0.28, 0.11])
    ax_note.set_axis_off()
    fig.text(0.08, 0.90, "Cut the loop, and the signature collapses.", fontsize=28, fontweight="bold")
    fig.text(0.08, 0.855, "Lesion feedback + integration at t = 3, then watch the mean Ns trace fall back toward baseline.", fontsize=14, color=SUBTLE)

    sham_line, = ax_line.plot([], [], color=BLUE, linewidth=3.0, label="Sham")
    lesion_line, = ax_line.plot([], [], color=ACCENT, linewidth=3.0, linestyle="--", label="Lesion")
    ax_line.axvline(lesion_t, color="#8b5cf6", linestyle=":", linewidth=2)
    ax_line.axhline(0.5, color="#cbd5e1", linestyle="--", linewidth=1.2)
    ax_line.set_xlim(0, show_until - 1)
    ax_line.set_ylim(0.48, 1.02)
    ax_line.set_xlabel("Time step", fontsize=14)
    ax_line.set_ylabel("Mean Ns", fontsize=14)
    ax_line.grid(color="#e3e8ef", linewidth=1.0)
    ax_line.legend(loc="upper right", frameon=False, fontsize=13)
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)
    label_box(ax_line, 9.5, 0.985, "lesion @ t=3", fc="#f5f3ff", ec="#ddd6fe", color="#6d28d9", fontsize=12)

    models = ["ReCoN", "Ipsundrum", "Ipsundrum+affect"]
    bar_x = np.arange(len(models))
    bars = ax_bar.bar(bar_x, auc_drop_pct, color=[COLORS[m] for m in models], width=0.58)
    ax_bar.set_xticks(bar_x)
    ax_bar.set_xticklabels(models, rotation=18, ha="right", fontsize=12)
    ax_bar.set_ylabel("AUC drop (%)", fontsize=13)
    ax_bar.set_ylim(0, 32)
    ax_bar.grid(axis="y", color="#e3e8ef", linewidth=1.0)
    ax_bar.set_axisbelow(True)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.set_title("Causal effect on persistence", fontsize=14, pad=10)

    for bar, value in zip(bars, auc_drop_pct):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=INK,
        )

    badge = fig.text(
        0.79,
        0.74,
        f"AUC drop: {claim(claims, 'lesion_auc_drop_pct_hb'):.1f}%",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color=COLORS["Ipsundrum+affect"],
        bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.2", fc="#f0f9f0", ec="#cfe6cf"),
    )
    rounded_box(ax_note, (0.02, 0.10), 0.96, 0.78, fc="white", ec="#dce4ee", lw=1.3, radius=0.03)
    ax_note.text(0.08, 0.50, "ReCoN stays unchanged;\nthe ipsundrum variants drop.", fontsize=12.8, color=SUBTLE, va="center", linespacing=1.25)

    hold_frames = 12

    def update(frame_idx: int):
        upto = min(frame_idx, show_until - 1)
        xs = x[: upto + 1]
        sham_line.set_data(xs, sham[: upto + 1])
        lesion_line.set_data(xs, lesion[: upto + 1])
        return [sham_line, lesion_line, badge]

    animation = FuncAnimation(fig, update, frames=show_until + hold_frames, interval=100, blit=False)
    animation.save(out_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def write_manifest(claims: dict) -> None:
    manifest = f"""# Twitter Thread Asset Pack

This folder contains a six-asset social pack built directly from the current `recips` repo and paper results.

Order:
1. `01_hero_same-task-different-internals.gif`
2. `02_architecture_same-substrate-three-interventions.png`
3. `03_novelty-seeking-is-not-preference.png`
4. `04_same-arena-very-different-play-style.png`
5. `05_persistence-alone-isnt-enough.png`
6. `06_cut-the-loop-and-the-signature-collapses.gif`

Key paper-linked numbers used on the cards:
- Familiarity control: ReCoN delta scenic-entry = {claim(claims, "fam_delta_scenic_entry_recon"):.02f}
- Familiarity control: Ipsundrum delta scenic-entry = {claim(claims, "fam_delta_scenic_entry_humphrey"):.02f}
- Familiarity control: Ipsundrum+affect delta scenic-entry = {claim(claims, "fam_delta_scenic_entry_hb"):.02f}
- Affect novelty note: median delta novelty = {claim(claims, "fam_median_delta_novelty_hb_scenic"):.02f}
- Exploratory play: affect scan events = {claim(claims, "play_scan_events_hb"):.1f}
- Exploratory play: ReCoN scan events = {claim(claims, "play_scan_events_recon"):.1f}
- Exploratory play: affect cycle score = {claim(claims, "play_cycle_score_hb"):.1f}
- Pain-tail duration: affect = {claim(claims, "pain_tail_duration_hb"):.0f}
- Pain-tail duration: ReCoN = {claim(claims, "pain_tail_duration_recon"):.0f}
- Lesion AUC drop: affect = {claim(claims, "lesion_auc_drop_pct_hb"):.1f}%

Primary sources in repo:
- `docs/paper-3-v9.tex`
- `results/paper/claims.json`
- `results/familiarity/episodes_improved.csv`
- `results/exploratory-play/exploratory_play_clarified_trace_paper.csv`
- `results/pain-tail/summary.csv`
- `results/lesion/ns_traces.npz`

Generation:
- Re-run with `python3 analysis/build_social_assets.py`
"""
    (OUT_DIR / "README.md").write_text(manifest, encoding="utf-8")


def main() -> None:
    style()
    ensure_dir()
    claims = read_claims()

    build_hero_gif(OUT_DIR / "01_hero_same-task-different-internals.gif")
    build_architecture_card(OUT_DIR / "02_architecture_same-substrate-three-interventions.png")
    build_novelty_card(OUT_DIR / "03_novelty-seeking-is-not-preference.png", claims)
    build_play_card(OUT_DIR / "04_same-arena-very-different-play-style.png", claims)
    build_pain_card(OUT_DIR / "05_persistence-alone-isnt-enough.png", claims)
    build_lesion_gif(OUT_DIR / "06_cut-the-loop-and-the-signature-collapses.gif", claims)
    write_manifest(claims)
    print(f"Social assets written to {OUT_DIR}")


if __name__ == "__main__":
    main()
