"""Build dissociation table from assay-backed claims."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def _load_claims(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _claim_bool(claims: Dict[str, dict], key: str) -> bool:
    entry = claims.get(key, {})
    return bool(entry.get("value", False))


def _claim_positive(claims: Dict[str, dict], key: str) -> bool:
    entry = claims.get(key, {})
    meta = entry.get("meta", {}) or {}
    if "pass" in meta:
        return bool(meta.get("pass"))
    try:
        return float(entry.get("value", 0.0)) > 0.0
    except Exception:
        return False


def build_table(claims: Dict[str, dict]) -> Dict[str, Dict[str, bool]]:
    models = ["perspective", "perspective_plastic", "gw_lite", "humphrey_barrett", "humphrey", "recon"]
    table = {
        "hysteresis": {m: False for m in models},
        "contextual_memory": {m: False for m in models},
        "conflict_robustness": {m: False for m in models},
        "plastic_residue": {m: False for m in models},
        "selective_lesions": {m: False for m in models},
    }
    table["hysteresis"]["perspective"] = _claim_positive(claims, "claim_hysteresis_perspective_gt_scalar")
    table["hysteresis"]["perspective_plastic"] = _claim_positive(claims, "claim_hysteresis_perspective_gt_scalar")
    table["contextual_memory"]["perspective"] = _claim_positive(claims, "claim_hysteresis_perspective_gt_scalar")
    table["contextual_memory"]["perspective_plastic"] = _claim_positive(claims, "claim_context_delay_perspective_plastic_gt_gw")
    table["conflict_robustness"]["gw_lite"] = _claim_positive(claims, "claim_gw_conflict_robustness_gt_perspective")
    table["plastic_residue"]["perspective_plastic"] = _claim_positive(claims, "claim_plasticity_residue_gt_no_plastic")
    table["selective_lesions"]["perspective"] = _claim_bool(claims, "claim_perspective_lesion_selective")
    table["selective_lesions"]["gw_lite"] = _claim_bool(claims, "claim_selector_lesion_selective")
    return table


def render_table(table: Dict[str, Dict[str, bool]]) -> str:
    models: List[str] = ["perspective", "perspective_plastic", "gw_lite", "humphrey_barrett", "humphrey", "recon"]
    headers = ["Perspective", "Perspective+plastic", "GW-lite", "Ipsundrum+affect", "Ipsundrum", "Recon"]
    row_labels = {
        "hysteresis": "Hysteresis under cue ramp",
        "contextual_memory": "Delay-tolerant contextual fork memory",
        "conflict_robustness": "Robustness under multimodal conflict",
        "plastic_residue": "Plastic residue after return",
        "selective_lesions": "Selective lesion signature",
    }

    def mark(val: bool) -> str:
        return "\\ensuremath{\\checkmark}" if val else "\\ensuremath{\\times}"

    row_end = "\\\\"
    lines = [
        "\\begin{tabular}{p{0.32\\linewidth}cccccc}",
        "\\toprule",
        "Signature & " + " & ".join(headers) + " " + row_end,
        "\\midrule",
    ]
    for key in ("hysteresis", "contextual_memory", "conflict_robustness", "plastic_residue", "selective_lesions"):
        row = [row_labels[key]] + [mark(table[key][m]) for m in models]
        lines.append("{} & {} & {} & {} & {} & {} & {} ".format(*row) + row_end)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
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
