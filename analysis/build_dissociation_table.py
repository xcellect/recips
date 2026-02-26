"""Build dissociation table (checkmarks/crosses) from claims."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple


def _load_claims(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _claim_mean_ci(claims: Dict[str, dict], key: str) -> Tuple[float, float, float]:
    entry = claims[key]
    mean = float(entry["value"])
    ci = entry.get("ci") or [float("nan"), float("nan")]
    return mean, float(ci[0]), float(ci[1])


def _ci_excludes_zero(ci_low: float, ci_high: float) -> bool:
    return ci_low > 0.0 or ci_high < 0.0


def build_table(claims: Dict[str, dict]) -> Dict[str, Dict[str, bool]]:
    # thresholds
    delta_threshold = 0.05
    scan_threshold = 1.0
    tail_threshold = 5.0

    models = ["recon", "humphrey", "humphrey_barrett"]

    # persistence in Ns
    recon_half, _, _ = _claim_mean_ci(claims, "pain_ns_half_life_recon")
    persistence = {}
    for model in models:
        if model == "recon":
            persistence[model] = False
            continue
        lesion_key = f"lesion_auc_drop_{'hb' if model == 'humphrey_barrett' else model}"
        if lesion_key not in claims:
            lesion_key = f"lesion_auc_drop_{model}"
        mean, ci_low, ci_high = _claim_mean_ci(claims, lesion_key)
        half_mean, _, _ = _claim_mean_ci(claims, f"pain_ns_half_life_{'hb' if model == 'humphrey_barrett' else model}")
        persistence[model] = _ci_excludes_zero(ci_low, ci_high) or (half_mean - recon_half >= tail_threshold)

    # valence-stable scenic preference
    valence_stable = {}
    for model in models:
        key = f"fam_delta_scenic_entry_{'hb' if model == 'humphrey_barrett' else model}"
        mean, ci_low, ci_high = _claim_mean_ci(claims, key)
        valence_stable[model] = (abs(mean) <= delta_threshold) and (ci_low >= -delta_threshold) and (ci_high <= delta_threshold)

    # structured scanning
    scan_means = {}
    scan_cis = {}
    for model in models:
        key = f"play_scan_events_{'hb' if model == 'humphrey_barrett' else model}"
        mean, ci_low, ci_high = _claim_mean_ci(claims, key)
        scan_means[model] = mean
        scan_cis[model] = (ci_low, ci_high)
    structured = {}
    for model in models:
        others = [m for m in models if m != model]
        max_other_mean = max(scan_means[m] for m in others)
        max_other_high = max(scan_cis[m][1] for m in others)
        ci_low = scan_cis[model][0]
        structured[model] = (scan_means[model] >= max_other_mean + scan_threshold) and (ci_low > max_other_high)

    # lingering planned caution
    tail_means = {}
    tail_cis = {}
    for model in models:
        key = f"pain_tail_duration_{'hb' if model == 'humphrey_barrett' else model}"
        mean, ci_low, ci_high = _claim_mean_ci(claims, key)
        tail_means[model] = mean
        tail_cis[model] = (ci_low, ci_high)
    caution = {}
    for model in models:
        others = [m for m in models if m != model]
        max_other_mean = max(tail_means[m] for m in others)
        max_other_high = max(tail_cis[m][1] for m in others)
        ci_low = tail_cis[model][0]
        caution[model] = (tail_means[model] >= max_other_mean + tail_threshold) and (ci_low > max_other_high)

    return {
        "persistence": persistence,
        "valence_stable": valence_stable,
        "structured_scan": structured,
        "lingering_caution": caution,
    }


def render_table(table: Dict[str, Dict[str, bool]]) -> str:
    models = ["recon", "humphrey", "humphrey_barrett"]
    headers = ["Recon", "Ipsundrum", "Ipsundrum+affect"]
    row_labels = {
        "persistence": "Persistence in $N^s$ (lesion / pain-tail half-life)",
        "valence_stable": "Valence-stable scenic preference (familiarity-controlled)",
        "structured_scan": "Structured local scanning (play scan events)",
        "lingering_caution": "Lingering planned caution (pain-tail tail duration)",
    }
    def mark(val: bool) -> str:
        return "\\ensuremath{\\checkmark}" if val else "\\ensuremath{\\times}"

    lines = [
        "\\begin{tabular}{p{0.33\\linewidth}ccc}",
        "\\toprule",
        f"Signature (assay) & {headers[0]} & {headers[1]} & {headers[2]} \\\\",
        "\\midrule",
    ]
    for key in ("persistence", "valence_stable", "structured_scan", "lingering_caution"):
        row = [row_labels[key]] + [mark(table[key][m]) for m in models]
        lines.append("{} & {} & {} & {} \\\\")
        lines[-1] = lines[-1].format(*row)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dissociation table")
    parser.add_argument("--outdir", type=str, default="results/paper", help="Output directory")
    parser.add_argument("--claims", type=str, default="results/paper/claims.json", help="Claims JSON path")
    args = parser.parse_args()

    claims = _load_claims(args.claims)
    table = build_table(claims)
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "dissociation_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render_table(table))


if __name__ == "__main__":
    main()
