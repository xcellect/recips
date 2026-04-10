"""Build paper-facing summary tables for the ALife mechanism-decomposition extension."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def pick_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these paths exist: {paths}")


def df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    header = "| " + " | ".join(str(col) for col in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for row in df.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join([header, divider, *rows])


def summarize_familiarity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    post = df[df["phase"] == "post"].copy()
    post = post[post["familiarize_side"].isin(["scenic", "dull", "both"])].copy()
    summary = (
        post.groupby(["model", "familiarize_side"], as_index=False, observed=False)
        .agg(
            scenic_rate_entry=("scenic_rate_entry", "mean"),
            scenic_time_share_barrier=("scenic_time_share_barrier", "mean"),
            split_delta_novelty=("split_delta_novelty", "mean"),
            mean_valence_scenic=("mean_valence_scenic", "mean"),
            mean_valence_dull=("mean_valence_dull", "mean"),
            mean_arousal_scenic=("mean_arousal_scenic", "mean"),
            mean_arousal_dull=("mean_arousal_dull", "mean"),
            n=("n", "sum"),
        )
    )
    return summary


def summarize_play(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [
        "model",
        "unique_viewpoints",
        "scan_events",
        "turn_in_place_rate",
        "revisit_rate",
        "cycle_score",
        "occupancy_entropy",
        "dwell_p90",
        "mean_valence",
        "mean_arousal",
        "mean_Ns",
        "mean_precision_eff",
        "mean_alpha_eff",
    ]
    return df[keep].copy()


def summarize_lesion(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    grouped = (
        df.groupby(["model", "condition"], as_index=False, observed=False)
        .agg(
            post_lesion_auc=("post_lesion_auc", "mean"),
            post_lesion_halflife=("post_lesion_halflife", "mean"),
            immediate_slope=("immediate_slope", "mean"),
            ns_at_lesion=("ns_at_lesion", "mean"),
            n=("seed", "count"),
        )
    )
    pivot = grouped.pivot(index="model", columns="condition")
    out = pd.DataFrame({"model": pivot.index})
    for metric in ["post_lesion_auc", "post_lesion_halflife", "immediate_slope", "ns_at_lesion", "n"]:
        if (metric, "sham") in pivot.columns:
            out[f"{metric}_sham"] = pivot[(metric, "sham")].values
        if (metric, "lesion") in pivot.columns:
            out[f"{metric}_lesion"] = pivot[(metric, "lesion")].values
    if "post_lesion_auc_sham" in out.columns and "post_lesion_auc_lesion" in out.columns:
        out["auc_drop_sham_minus_lesion"] = (
            out["post_lesion_auc_sham"] - out["post_lesion_auc_lesion"]
        )
    return out


def summarize_pain_tail(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [
        "model",
        "mean_tail_duration",
        "std_tail_duration",
        "mean_ns_auc_above_baseline",
        "std_ns_auc_above_baseline",
        "mean_arousal_integral",
        "mean_turn_rate_tail",
        "mean_forward_rate_tail",
        "n",
    ]
    df = df[df["n"] > 0].copy()
    return df[keep]


def render_markdown(
    output_path: Path,
    *,
    root_dir: Path,
    familiarity_src: Path,
    play_src: Path,
    lesion_src: Path,
    pain_src: Path,
    familiarity: pd.DataFrame,
    play: pd.DataFrame,
    lesion: pd.DataFrame,
    pain: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# ALife Extension Engineering Notes")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "Mechanism-decomposition extension for the ALife paper: split affect coupling into policy readout versus recurrent-loop modulation and compare the resulting behavior across familiarity control, exploratory play, lesion, and pain-tail assays."
    )
    lines.append("")
    lines.append("## Code Changes")
    lines.append("")
    lines.append("- Added `--models` CLI support to `experiments/pain_tail_assay.py` so the ablation can be run directly without a wrapper.")
    lines.append("- Extended `run_experiments.sh` so the ablation batch now covers familiarity, exploratory play, lesion, and pain-tail.")
    lines.append("- Added `analysis/alife_mechanism_decomposition.py` to compile paper-facing summary tables and markdown notes from the raw assay outputs.")
    lines.append("")
    lines.append("## Assay Sources")
    lines.append("")
    lines.append(f"- Familiarity summary source: `{familiarity_src}`")
    lines.append(f"- Exploratory play summary source: `{play_src}`")
    lines.append(f"- Lesion episodes source: `{lesion_src}`")
    lines.append(f"- Pain-tail summary source: `{pain_src}`")
    lines.append("")
    lines.append("## Commands Run")
    lines.append("")
    lines.append("```bash")
    lines.append("MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python -m experiments.familiarity_control --seeds 10 --post_repeats 5 --models humphrey_barrett,humphrey_barrett_readout_only,humphrey_barrett_modulation_only --outdir results/extensions/alife_mech_decomp/familiarity")
    lines.append("MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python -m experiments.exploratory_play --profile quick --seeds 10 --steps 200 --config_set ablation --tag alife_extension_quick --outdir results/extensions/alife_mech_decomp/exploratory-play")
    lines.append("MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python -m experiments.lesion_causal --seeds 10 --post_window 150 --models humphrey_barrett,humphrey_barrett_readout_only,humphrey_barrett_modulation_only --outdir results/extensions/alife_mech_decomp/lesion")
    lines.append("MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python -m experiments.pain_tail_assay --seeds 10 --post_steps 200 --models humphrey_barrett,humphrey_barrett_readout_only,humphrey_barrett_modulation_only --outdir results/extensions/alife_mech_decomp/pain-tail")
    lines.append("MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python -m analysis.alife_mechanism_decomposition --root_dir results/extensions/alife_mech_decomp --outdir results/extensions/alife_mech_decomp/summary")
    lines.append("```")
    lines.append("")
    lines.append("## High-Signal Readouts")
    lines.append("")
    lines.append("### Familiarity")
    lines.append("")
    lines.append(df_to_markdown_table(familiarity))
    lines.append("")
    lines.append("### Exploratory Play")
    lines.append("")
    lines.append(df_to_markdown_table(play))
    lines.append("")
    lines.append("### Lesion")
    lines.append("")
    lines.append(df_to_markdown_table(lesion))
    lines.append("")
    lines.append("### Pain-Tail")
    lines.append("")
    lines.append(df_to_markdown_table(pain))
    lines.append("")
    lines.append("## Fast Interpretation Notes")
    lines.append("")
    lines.append("- Familiarity is the cleanest route-choice assay for readout sensitivity because it directly tracks scenic commitment after structured exposure.")
    lines.append("- Lesion AUC drop isolates persistence in the recurrent machinery; if the lesion effect remains strong in modulation-only, persistence is not purely a policy-readout effect.")
    lines.append("- Exploratory-play metrics are mixed by design: scanning, revisit structure, and dwell depend on several policy paths rather than one scalar affect score.")
    lines.append("- Pain-tail should be treated cautiously here because the current summary mixes large duration variance with near-zero Ns half-life under the baseline-corrected metric.")
    lines.append("")
    lines.append("## Output Inventory")
    lines.append("")
    for path in sorted(root_dir.rglob("*")):
        if path.is_file():
            lines.append(f"- `{path}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ALife mechanism-decomposition extension outputs")
    parser.add_argument("--root_dir", type=Path, default=Path("results/extensions/alife_mech_decomp"))
    parser.add_argument("--outdir", type=Path, default=Path("results/extensions/alife_mech_decomp/summary"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    familiarity_src = pick_existing(
        [
            args.root_dir / "familiarity" / "summary.csv",
            Path("results/ablations/familiarity/summary.csv"),
        ]
    )
    play_src = pick_existing(
        [
            args.root_dir / "exploratory-play" / "summary_alife_extension_quick.csv",
            Path("results/ablations/exploratory-play/summary_ablation_paper.csv"),
        ]
    )
    lesion_src = pick_existing([args.root_dir / "lesion" / "episodes_extended.csv"])
    pain_src = pick_existing([args.root_dir / "pain-tail" / "summary.csv"])

    familiarity = summarize_familiarity(familiarity_src)
    play = summarize_play(play_src)
    lesion = summarize_lesion(lesion_src)
    pain = summarize_pain_tail(pain_src)

    familiarity.to_csv(args.outdir / "familiarity_mechanism_summary.csv", index=False)
    play.to_csv(args.outdir / "exploratory_play_mechanism_summary.csv", index=False)
    lesion.to_csv(args.outdir / "lesion_mechanism_summary.csv", index=False)
    pain.to_csv(args.outdir / "pain_tail_mechanism_summary.csv", index=False)

    render_markdown(
        args.outdir / "ENGINEERING_NOTES.md",
        root_dir=args.root_dir,
        familiarity_src=familiarity_src,
        play_src=play_src,
        lesion_src=lesion_src,
        pain_src=pain_src,
        familiarity=familiarity,
        play=play,
        lesion=lesion,
        pain=pain,
    )


if __name__ == "__main__":
    main()
