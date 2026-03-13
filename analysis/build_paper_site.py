"""Build a static project page from existing paper artifacts."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from core.ipsundrum_model import Builder
from core.recon_core import NodeKind
from experiments.viz_utils import corridor_viz, gridworld_viz
from utils.model_naming import MODEL_DISPLAY_ORDER, canonical_model_display

ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "paper-site"
STATIC_DIR = SITE_DIR / "static"
MEDIA_DIR = STATIC_DIR / "media"
DATA_DIR = STATIC_DIR / "data"

ARXIV_PAPER = {
    "title": "ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Control and Mechanism-Linked Consciousness Indicator Assays",
    "authors": ["Aishik Sanyal"],
    "authors_display": "Aishik Sanyal",
    "author_url": "https://xcellect.com",
    "arxiv_id": "arXiv:2602.23232",
    "arxiv_id_versioned": "arXiv:2602.23232v2",
    "abs_url": "https://arxiv.org/abs/2602.23232",
    "pdf_url": "https://arxiv.org/pdf/2602.23232.pdf",
    "doi": "10.48550/arXiv.2602.23232",
    "doi_url": "https://doi.org/10.48550/arXiv.2602.23232",
    "subject": "Artificial Intelligence (cs.AI)",
    "submitted": "Submitted on 26 Feb 2026",
    "revised": "last revised 1 Mar 2026 (this version, v2)",
    "comments": "Accepted at AAAI 2026 Spring Symposium - Machine Consciousness: Integrating Theory, Technology, and Philosophy",
    "abstract": (
        "Indicator-based approaches to machine consciousness recommend mechanism-linked evidence triangulated across tasks, "
        "supported by architectural inspection and causal intervention. Inspired by Humphrey's ipsundrum hypothesis, we "
        "implement ReCoN-Ipsundrum, an inspectable agent that extends a ReCoN state machine with a recurrent persistence "
        "loop over sensory salience Ns and an optional affect proxy reporting valence/arousal. Across fixed-parameter "
        "ablations (ReCoN, Ipsundrum, Ipsundrum+affect), we operationalize Humphrey's qualiaphilia (preference for sensory "
        "experience for its own sake) as a familiarity-controlled scenic-over-dull route choice. We find a novelty "
        "dissociation: non-affect variants are novelty-sensitive (Delta scenic-entry = 0.07). Affect coupling is stable "
        "(Delta scenic-entry = 0.01) even when scenic is less novel (median Delta novelty approx. -0.43). In reward-free "
        "exploratory play, the affect variant shows structured local investigation (scan events 31.4 vs. 0.9; cycle score "
        "7.6). In a pain-tail probe, only the affect variant sustains prolonged planned caution (tail duration 90 vs. 5). "
        "Lesioning feedback+integration selectively reduces post-stimulus persistence in ipsundrum variants (AUC drop 27.62, "
        "27.9%) while leaving ReCoN unchanged. These dissociations link recurrence to persistence and affect-coupled control "
        "to preference stability, scanning, and lingering caution, illustrating how indicator-like signatures can be "
        "engineered and why mechanistic and causal evidence should accompany behavioral markers."
    ),
    "bibtex": (
        "@misc{sanyal2026reconipsundrum,\n"
        "  title         = {ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Control and Mechanism-Linked Consciousness Indicator Assays},\n"
        "  author        = {Aishik Sanyal},\n"
        "  year          = {2026},\n"
        "  eprint        = {2602.23232},\n"
        "  archivePrefix = {arXiv},\n"
        "  primaryClass  = {cs.AI},\n"
        "  doi           = {10.48550/arXiv.2602.23232},\n"
        "  url           = {https://arxiv.org/abs/2602.23232},\n"
        "  note          = {Accepted at AAAI 2026 Spring Symposium - Machine Consciousness: Integrating Theory, Technology, and Philosophy}\n"
        "}"
    ),
}


def ensure_dirs() -> None:
    for path in (MEDIA_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def claim_value(claims: dict, claim_id: str):
    return claims[claim_id]["value"]


def model_order(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    out = df.copy()
    out[model_col] = out[model_col].map(canonical_model_display)
    order_lookup = {name: idx for idx, name in enumerate(MODEL_DISPLAY_ORDER)}
    out["_sort_key"] = out[model_col].map(lambda value: order_lookup.get(value, len(order_lookup)))
    out = out.sort_values(["_sort_key", model_col]).drop(columns=["_sort_key"])
    return out


def normalize_records(df: pd.DataFrame, model_col: str = "model") -> List[dict]:
    out = df.copy()
    if model_col in out.columns:
        out[model_col] = out[model_col].map(canonical_model_display)
    records = json.loads(out.to_json(orient="records"))
    return records


def build_stage_graph(net, add_aux_edges: bool = True, add_reverse_edges: bool = True) -> nx.DiGraph:
    graph = nx.DiGraph()
    aux_nodes = {"Ni", "Nv", "Na"}

    for nid, node in net.nodes.items():
        if node.kind == NodeKind.SCRIPT:
            state = node.script_state.name
        else:
            state = getattr(node, "terminal_state", None)
            state = state.name if state is not None else ""
        graph.add_node(nid, kind=node.kind.name, state=state)

    for nid, node in net.nodes.items():
        for child in getattr(node, "children", []):
            if child in net.nodes:
                graph.add_edge(nid, child, etype="sub")
                if add_reverse_edges:
                    graph.add_edge(child, nid, etype="sur")

        successor = getattr(node, "successor", None)
        predecessor = getattr(node, "predecessor", None)
        if successor is not None and successor in net.nodes:
            graph.add_edge(nid, successor, etype="por")
        if add_reverse_edges and predecessor is not None and predecessor in net.nodes:
            graph.add_edge(nid, predecessor, etype="ret")

    if add_aux_edges and "Root" in net.nodes:
        for nid in aux_nodes:
            if nid in net.nodes and getattr(net.get(nid), "parent", None) is None:
                graph.add_edge("Root", nid, etype="aux")

    return graph


def hierarchy_positions(graph: nx.DiGraph, root: str = "Root") -> Dict[str, Tuple[float, float]]:
    aux_nodes = {"Ni", "Nv", "Na"}
    hierarchy = nx.DiGraph()
    hierarchy.add_nodes_from(graph.nodes())
    hierarchy.add_edges_from(
        (u, v)
        for u, v, data in graph.edges(data=True)
        if data.get("etype") == "sub"
    )

    depth = {root: 0}
    queue = [root]
    while queue:
        current = queue.pop(0)
        for child in hierarchy.successors(current):
            if child not in depth:
                depth[child] = depth[current] + 1
                queue.append(child)

    for node in graph.nodes():
        if node not in depth:
            depth[node] = 1 if node in aux_nodes else 999

    levels: Dict[int, List[str]] = {}
    for node, layer in depth.items():
        levels.setdefault(layer, []).append(node)

    positions: Dict[str, Tuple[float, float]] = {}
    max_depth = max(levels) if levels else 0
    for layer in sorted(levels):
        nodes = sorted(levels[layer])
        for idx, node in enumerate(nodes, start=1):
            x = idx / (len(nodes) + 1)
            y = 1.0 - (layer / (max_depth + 1 if max_depth >= 0 else 1))
            positions[node] = (x, y)
    return positions


def node_color(kind: str) -> str:
    colors = {
        "SCRIPT": "#1f3c5f",
        "SENSOR": "#d8a31a",
        "ACTUATOR": "#b4572e",
        "AFFECT": "#9f3d5f",
        "INTERNAL": "#356859",
    }
    return colors.get(kind, "#4b5563")


def draw_stage_panel(ax, net, title: str) -> None:
    graph = build_stage_graph(net)
    pos = hierarchy_positions(graph, root="Root") if "Root" in graph.nodes else nx.spring_layout(graph, seed=0)
    labels = {
        node: f"{node}\n{graph.nodes[node]['kind']}"
        for node in graph.nodes
    }

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=1450,
        node_color=[node_color(graph.nodes[node]["kind"]) for node in graph.nodes],
        edgecolors="#f8f4ed",
        linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, font_color="#f8f4ed", ax=ax)

    sub_edges = {(u, v) for u, v, d in graph.edges(data=True) if d.get("etype") == "sub"}
    por_edges = {(u, v) for u, v, d in graph.edges(data=True) if d.get("etype") == "por"}
    sur_edges = {(u, v) for u, v, d in graph.edges(data=True) if d.get("etype") == "sur"}
    ret_edges = {(u, v) for u, v, d in graph.edges(data=True) if d.get("etype") == "ret"}
    aux_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("etype") == "aux"]

    shared_sub = {tuple(sorted((u, v))) for (u, v) in sub_edges if (v, u) in sur_edges}
    shared_por = {tuple(sorted((u, v))) for (u, v) in por_edges if (v, u) in ret_edges}
    edge_kw = dict(arrows=True, arrowsize=14, min_source_margin=8, min_target_margin=8, ax=ax)
    if shared_sub:
        nx.draw_networkx_edges(graph, pos, edgelist=sorted(shared_sub), arrowstyle="<|-|>", width=1.2, edge_color="#d9e2ec", **edge_kw)
    if shared_por:
        nx.draw_networkx_edges(graph, pos, edgelist=sorted(shared_por), arrowstyle="<|-|>", width=1.0, style="dashed", edge_color="#efb366", **edge_kw)
    if aux_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=aux_edges, arrows=False, style="dotted", width=1.0, edge_color="#7fc8a9", ax=ax)

    ax.set_title(title, color="#f8f4ed", fontsize=11, pad=10)
    ax.set_axis_off()


def render_stage_strip(out_path: Path) -> None:
    builder = Builder()
    stages = [
        ("Stage A", builder.stage_A()[0]),
        ("Stage B", builder.stage_B()[0]),
        ("Stage C", builder.stage_C(cycles_required=10)[0]),
        ("Stage D", builder.stage_D(efference_threshold=0.05)[0]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#06131f")
    for ax, (title, net) in zip(axes.flat, stages):
        ax.set_facecolor("#102133")
        draw_stage_panel(ax, net, title)
    fig.suptitle("Mechanism scaffold: recurrent persistence accumulates across stages", color="#f8f4ed", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def lesion_trace_means(npz_path: Path) -> Dict[str, Dict[str, List[float]]]:
    traces = np.load(npz_path)
    grouped: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for key in traces.files:
        model_name, rest = key.split("_seed", 1)
        _, condition = rest.split("_", 1)
        grouped.setdefault(model_name, {}).setdefault(condition, []).append(np.asarray(traces[key], dtype=float))

    means: Dict[str, Dict[str, List[float]]] = {}
    for model_name, cond_map in grouped.items():
        display_name = canonical_model_display(model_name)
        means[display_name] = {}
        for condition, arrays in cond_map.items():
            stacked = np.vstack(arrays)
            means[display_name][condition] = stacked.mean(axis=0).round(6).tolist()
    return means


def render_lesion_gif(out_path: Path, mean_traces: Dict[str, Dict[str, List[float]]], lesion_t: int = 3) -> None:
    models = [name for name in MODEL_DISPLAY_ORDER if name in mean_traces]
    arrays = {
        model: {
            condition: np.asarray(values, dtype=float)
            for condition, values in mean_traces[model].items()
        }
        for model in models
    }
    length = min(len(arrays[model]["sham"]) for model in models if "sham" in arrays[model])
    frame_points = np.arange(length, dtype=int)

    fig, axes = plt.subplots(1, len(models), figsize=(13.5, 4.4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#071019")
    line_map = {}

    for ax, model in zip(axes, models):
        ax.set_facecolor("#0f2231")
        sham_line, = ax.plot([], [], color="#f4efe6", linewidth=2.4, label="Sham")
        lesion_line, = ax.plot([], [], color="#e6613f", linewidth=2.4, linestyle="--", label="Lesion")
        marker = ax.axvline(lesion_t, color="#ffcf6d", linewidth=1.2, linestyle=":")
        ax.axvspan(lesion_t, length - 1, color="#ffcf6d", alpha=0.08)
        ax.set_title(model, color="#f8f4ed", fontsize=11)
        ax.set_xlim(0, length - 1)
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.15, color="#f8f4ed")
        ax.tick_params(colors="#c7d2da", labelsize=8)
        ax.spines["bottom"].set_color("#365366")
        ax.spines["left"].set_color("#365366")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        line_map[model] = (sham_line, lesion_line, marker)

    axes[0].set_ylabel("Mean Ns", color="#f8f4ed")
    fig.suptitle("Causal lesion probe: sham vs lesion mean traces", color="#f8f4ed", fontsize=14, y=0.98)
    axes[-1].legend(loc="upper right", fontsize=8, frameon=False, labelcolor="#f8f4ed")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def update(frame_idx: int):
        upto = int(frame_points[frame_idx])
        artists = []
        xs = np.arange(upto + 1)
        for model in models:
            sham_line, lesion_line, marker = line_map[model]
            sham_line.set_data(xs, arrays[model]["sham"][: upto + 1])
            lesion_line.set_data(xs, arrays[model]["lesion"][: upto + 1])
            marker.set_xdata([lesion_t, lesion_t])
            artists.extend([sham_line, lesion_line, marker])
        return artists

    animation = FuncAnimation(fig, update, frames=len(frame_points), interval=90, blit=False)
    animation.save(out_path, writer=PillowWriter(fps=10), dpi=100)
    plt.close(fig)


def export_media() -> Dict[str, str]:
    copied = {
        "play_summary": ("results/publication-figures/fig1_play.png", "fig-play-summary.png"),
        "familiarity_summary": ("results/publication-figures/fig2.png", "fig-familiarity-summary.png"),
        "familiarity_supp": ("results/publication-figures/fig2_supp.png", "fig-familiarity-supp.png"),
        "lesion_summary": ("results/publication-figures/fig3.png", "fig-lesion-summary.png"),
        "play_trajectories": ("results/exploratory-play/fig_exploratory_play_trajectories_paper.png", "play-trajectories.png"),
        "play_heatmaps": ("results/exploratory-play/fig_exploratory_play_clarified_heatmaps_paper.png", "play-heatmaps.png"),
        "play_dwell": ("results/exploratory-play/fig_exploratory_play_dwell_paper.png", "play-dwell.png"),
        "familiarity_internal": ("results/familiarity/familiarity_internal_20260219_001714.png", "familiarity-internal.png"),
        "pain_tail": ("results/pain-tail/results.png", "pain-tail.png"),
        "qualiaphilia": ("results/qualiaphilia/results.png", "qualiaphilia.png"),
    }
    asset_paths: Dict[str, str] = {}
    for key, (src_rel, dst_name) in copied.items():
        src = ROOT / src_rel
        dst = MEDIA_DIR / dst_name
        if src.exists():
            copy_file(src, dst)
            asset_paths[key] = f"static/media/{dst_name}"

    stage_dst = MEDIA_DIR / "stage-strip.png"
    render_stage_strip(stage_dst)
    asset_paths["stage_strip"] = "static/media/stage-strip.png"

    corridor_dst = MEDIA_DIR / "corridor-compare.gif"
    corridor_viz.save_animation_gif(str(corridor_dst), horizon=6, frames=200, fps=10, dpi=100, figsize=(13.2, 7.8))
    asset_paths["corridor_gif"] = "static/media/corridor-compare.gif"

    gridworld_dst = MEDIA_DIR / "gridworld-compare.gif"
    gridworld_viz.save_animation_gif(str(gridworld_dst), horizon=10, frames=200, fps=10, dpi=100, figsize=(13.2, 7.8))
    asset_paths["gridworld_gif"] = "static/media/gridworld-compare.gif"

    lesion_means = lesion_trace_means(ROOT / "results/lesion/ns_traces.npz")
    lesion_dst = MEDIA_DIR / "lesion-traces.gif"
    render_lesion_gif(lesion_dst, lesion_means, lesion_t=3)
    asset_paths["lesion_gif"] = "static/media/lesion-traces.gif"
    return asset_paths


def build_site_data(asset_paths: Dict[str, str]) -> dict:
    claims = read_json(ROOT / "results/paper/claims.json")
    config = read_json(ROOT / "results/config_metadata.json")

    play_df = model_order(pd.read_csv(ROOT / "results/exploratory-play/final_viewpoints.csv"))
    familiarity_df = model_order(pd.read_csv(ROOT / "results/familiarity/summary.csv"))
    goal_grid_df = model_order(pd.read_csv(ROOT / "results/goal-directed/gridworld_summary.csv"))
    goal_corridor_df = model_order(pd.read_csv(ROOT / "results/goal-directed/corridor_summary.csv"))
    lesion_means = lesion_trace_means(ROOT / "results/lesion/ns_traces.npz")

    headline_metrics = [
        {
            "label": "Scenic choice after scenic familiarization",
            "value": round(float(claim_value(claims, "fam_post_scenic_entry_hb_scenic")), 2),
            "suffix": "",
            "context": "Ipsundrum+affect keeps choosing scenic even when scenic becomes less novel.",
        },
        {
            "label": "Exploratory scan events",
            "value": round(float(claim_value(claims, "play_scan_events_hb")), 1),
            "suffix": "",
            "context": "Large jump in scan structure over Recon.",
        },
        {
            "label": "Pain-tail duration",
            "value": int(round(float(claim_value(claims, "pain_tail_duration_hb")))),
            "suffix": " steps",
            "context": "Post-stimulus persistence stays active long after the hazard pulse disappears.",
        },
        {
            "label": "Lesion AUC drop",
            "value": round(float(claim_value(claims, "lesion_auc_drop_hb")), 2),
            "suffix": "",
            "context": "Causal disruption appears only once the recurrent machinery matters.",
        },
        {
            "label": "Cycle score",
            "value": round(float(claim_value(claims, "play_cycle_score_hb")), 1),
            "suffix": "",
            "context": "Revisitation stays structured rather than collapsing into random wandering.",
        },
    ]

    play_metric_labels = {
        "unique_viewpoints": "Unique viewpoints",
        "neutral_sensory_entropy": "Neutral texture entropy",
        "viewpoint_entropy": "Viewpoint entropy",
        "scan_events": "Scan events",
        "cycle_score": "Cycle score",
        "hazard_contacts": "Hazard contacts",
        "boundary_hugging_fraction": "Boundary hugging",
        "dwell_p90": "Dwell p90",
        "occupancy_entropy": "Occupancy entropy",
        "mean_alpha_eff": "Mean alpha_eff",
    }

    familiarity_metric_labels = {
        "scenic_rate_entry": "Scenic entry rate",
        "split_delta_novelty": "Delta novelty",
        "mean_valence_scenic": "Mean valence (scenic)",
        "mean_arousal_scenic": "Mean arousal (scenic)",
    }

    goal_metric_labels = {
        "success_rate": "Success rate",
        "mean_hazards": "Mean hazards",
        "mean_time": "Mean time",
        "mean_Ns": "Mean Ns",
        "mean_alpha": "Mean alpha",
    }

    gallery = [
        {
            "title": "Exploratory dwell distribution",
            "src": asset_paths.get("play_dwell", ""),
            "caption": "Dwell statistics provide an additional check that the scan signature is not high-entropy dithering.",
        },
        {
            "title": "Familiarity internal probe",
            "src": asset_paths.get("familiarity_internal", ""),
            "caption": "Internal valence and arousal traces support the familiarity-control interpretation reported in the paper.",
        },
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper": ARXIV_PAPER,
        "headline_metrics": headline_metrics,
        "claims": claims,
        "config": {
            "profile": config.get("profile", "paper"),
            "headline_seeds": int(claim_value(claims, "headline_seeds")),
            "goal_directed_seeds": int(claim_value(claims, "goal_directed_seeds")),
            "familiarity_post_repeats": config.get("familiarity_post_repeats"),
            "lesion_time": config.get("lesion", {}).get("lesion_time", 3),
        },
        "links": {
            "paper_pdf": ARXIV_PAPER["pdf_url"],
            "colab": "https://colab.research.google.com/github/xcellect/recips/blob/main/playground.ipynb",
            "repo": "https://github.com/xcellect/recips",
        },
        "assets": {
            "hero": asset_paths.get("corridor_gif", ""),
            "gridworld": asset_paths.get("gridworld_gif", ""),
            "lesion": asset_paths.get("lesion_gif", ""),
            "stage_strip": asset_paths.get("stage_strip", ""),
            "play_summary": asset_paths.get("play_summary", ""),
            "play_trajectories": asset_paths.get("play_trajectories", ""),
            "play_heatmaps": asset_paths.get("play_heatmaps", ""),
            "familiarity_summary": asset_paths.get("familiarity_summary", ""),
            "familiarity_supp": asset_paths.get("familiarity_supp", ""),
            "lesion_summary": asset_paths.get("lesion_summary", ""),
            "pain_tail": asset_paths.get("pain_tail", ""),
            "qualiaphilia": asset_paths.get("qualiaphilia", ""),
        },
        "play": {
            "points": normalize_records(play_df),
            "metric_labels": play_metric_labels,
            "default_x": "scan_events",
            "default_y": "hazard_contacts",
        },
        "familiarity": {
            "summary": normalize_records(familiarity_df),
            "metric_labels": familiarity_metric_labels,
            "default_metric": "scenic_rate_entry",
            "default_condition": "scenic",
        },
        "goal_directed": {
            "summary": normalize_records(pd.concat([goal_grid_df, goal_corridor_df], ignore_index=True)),
            "metric_labels": goal_metric_labels,
            "default_task": "gridworld",
            "default_metric": "success_rate",
        },
        "lesion": {
            "lesion_t": config.get("lesion", {}).get("lesion_time", 3),
            "mean_traces": lesion_means,
        },
        "gallery": gallery,
    }


def write_site_data(payload: dict) -> None:
    with (DATA_DIR / "site-data.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    ensure_dirs()
    asset_paths = export_media()
    payload = build_site_data(asset_paths)
    write_site_data(payload)
    print(f"Site assets written to {SITE_DIR}")


if __name__ == "__main__":
    main()
