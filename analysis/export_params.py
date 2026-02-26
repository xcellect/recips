"""Export parameter/config tables for the paper."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from experiments.evaluation_harness import default_loop_params, default_affect_params


def _load_config(results_dir: str) -> Dict[str, Any]:
    path = os.path.join(results_dir, "config_metadata.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _format_value(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:.3f}".rstrip("0").rstrip(".")
    return str(val)


def build_params_rows(config: Dict[str, Any]) -> List[Tuple[str, Any, str]]:
    loop = default_loop_params()
    aff_on = default_affect_params(True)

    rows = [
        ("Loop gain g", loop.g, "Ipsundrum"),
        ("Motor gain h", loop.h, "Ipsundrum"),
        ("Internal decay d", loop.internal_decay, "Ipsundrum"),
        ("Sensor bias", loop.sensor_bias, "signed I_t in [-1,1]"),
        ("Divisive norm", getattr(loop, "divisive_norm", 0.0), "Ns saturation control"),
        ("Affect enabled", aff_on.enabled, "affect variants"),
        ("Valence scale", aff_on.valence_scale, "affect readout"),
        ("k_homeo", aff_on.k_homeo, "body-budget update"),
        ("k_pe", aff_on.k_pe, "body-budget update"),
        ("Demand motor", aff_on.demand_motor, "body-budget update"),
        ("Demand stim", aff_on.demand_stim, "body-budget update"),
        ("Modulate g", aff_on.modulate_g, "affect coupling"),
        ("k_g_arousal", aff_on.k_g_arousal, "affect coupling"),
        ("k_g_unpleasant", aff_on.k_g_unpleasant, "affect coupling"),
        ("Modulate precision", aff_on.modulate_precision, "affect coupling"),
        ("Precision base", aff_on.precision_base, "affect coupling"),
        ("k_precision_arousal", aff_on.k_precision_arousal, "affect coupling"),
        ("Score w_valence", 2.0, "internal score"),
        ("Score w_arousal", -1.2, "internal score"),
        ("Score w_ns", -0.8, "internal score"),
        ("Score w_bb_err", -0.4, "internal score"),
    ]

    if config:
        seeds = config.get("seeds", {})
        windows = config.get("window_lengths", {})
        horizons = config.get("horizons", {})
        rows.extend([
            ("Seeds (familiarity)", seeds.get("familiarity"), "config"),
            ("Seeds (exploratory play)", seeds.get("exploratory_play"), "config"),
            ("Seeds (lesion)", seeds.get("lesion"), "config"),
            ("Play steps", windows.get("exploratory_play_steps"), "config"),
            ("Familiarity steps", windows.get("familiarity_steps"), "config"),
            ("Lesion post window", windows.get("lesion_post_window"), "config"),
            ("Pain-tail post steps", windows.get("pain_tail_post_stimulus_steps"), "config"),
            ("Goal horizons", horizons.get("goal_directed"), "config"),
        ])

    return rows


def write_table_tex(rows: List[Tuple[str, Any, str]], out_path: str) -> None:
    lines = ["\\begin{tabular}{p{0.48\\linewidth}p{0.28\\linewidth}p{0.18\\linewidth}}",
             "\\toprule",
             "Parameter & Value & Notes \\",
             "\\midrule"]
    for name, value, note in rows:
        lines.append(f"{name} & {_format_value(value)} & {note} \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export parameter table")
    parser.add_argument("--outdir", type=str, default="results/paper", help="Output directory")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    args = parser.parse_args()

    config = _load_config(args.results_dir)
    rows = build_params_rows(config)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.DataFrame(rows, columns=["parameter", "value", "notes"])
    df.to_csv(os.path.join(args.outdir, "params_table.csv"), index=False)
    write_table_tex(rows, os.path.join(args.outdir, "params_table.tex"))


if __name__ == "__main__":
    main()
