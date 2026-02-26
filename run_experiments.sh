#!/bin/bash
# Master script to run all evaluation tests with organized logging
set -euo pipefail

PROFILE="${PROFILE:-paper}"
if [[ "${PROFILE}" != "paper" && "${PROFILE}" != "quick" ]]; then
  echo "ERROR: PROFILE must be 'paper' or 'quick' (got '${PROFILE}')" >&2
  exit 1
fi

SEEDS_HEADLINE=5
GOAL_DIRECTED_SEEDS=10
QUALIAPHILIA_SEEDS=5
PAIN_TAIL_SEEDS="${PAIN_TAIL_SEEDS:-20}"
PAIN_POST_STEPS=50
if [[ "${PROFILE}" == "paper" ]]; then
  SEEDS_HEADLINE=20
  GOAL_DIRECTED_SEEDS=20
  QUALIAPHILIA_SEEDS=20
  PAIN_POST_STEPS=200
fi
EXPLORATORY_STEPS=200
FAMILIARITY_POST_REPEATS=5
LESION_POST_WINDOW=150
export PROFILE SEEDS_HEADLINE GOAL_DIRECTED_SEEDS QUALIAPHILIA_SEEDS PAIN_TAIL_SEEDS PAIN_POST_STEPS EXPLORATORY_STEPS FAMILIARITY_POST_REPEATS LESION_POST_WINDOW

echo "======================================================================"
echo "COMPLETE EVALUATION SUITE"
echo "======================================================================"
echo ""

# ReCoN strict-mode toggle (default strict)
if [[ -z "${RECON_STRICT+x}" ]]; then
  export RECON_STRICT=1
fi
RECON_MODE="${RECON_MODE:-strict}"
case "${RECON_STRICT,,}" in
  1|true|yes|on)
    RECON_MODE="strict"
    ;;
  0|false|no|off)
    RECON_MODE="compat"
    ;;
esac
echo "Recon mode: ${RECON_MODE}"
echo "Profile: ${PROFILE} (headline seeds=${SEEDS_HEADLINE})"
echo ""

# Clean and create directory structure
echo "Setting up directory structure..."
rm -rf results logs
mkdir -p results/{exploratory-play,familiarity,lesion,pain-tail,qualiaphilia,goal-directed,publication-figures,publication-figures/debug,ablations,paper}
mkdir -p logs

# Test 1: Goal-Directed Performance
echo ""
echo "======================================================================"
echo "TEST 1/8: Goal-Directed Performance (Corridor + Gridworld)"
echo "======================================================================"
python3 -m experiments.goal_directed_sweeps --seeds "${GOAL_DIRECTED_SEEDS}" --horizons "$(seq -s, 1 20)" --outdir results/goal-directed 2>&1 | tee logs/goal_directed.log
echo "✓ Completed. Log: logs/goal_directed.log"

# Test 2: Pain-Tail (Baseline)
echo ""
echo "======================================================================"
echo "TEST 2/8: Pain-Tail Assay (Baseline - Thick Moment)"
echo "======================================================================"
python3 -m experiments.pain_tail_assay --seeds "${PAIN_TAIL_SEEDS}" --post_steps "${PAIN_POST_STEPS}" --outdir results/pain-tail 2>&1 | tee logs/pain_tail.log
echo "✓ Completed. Log: logs/pain_tail.log"

# Test 3: Qualiaphilia (Baseline)
echo ""
echo "======================================================================"
echo "TEST 3/8: Qualiaphilia Assay (Baseline - Scenic Preference)"
echo "======================================================================"
python3 -m experiments.qualiaphilia_assay --seeds "${QUALIAPHILIA_SEEDS}" --outdir results/qualiaphilia 2>&1 | tee logs/qualiaphilia.log
echo "✓ Completed. Log: logs/qualiaphilia.log"

# Test 4: Exploratory Play
echo ""
echo "======================================================================"
echo "TEST 4/8: Exploratory Play (Viewpoints + Scan Metrics)"
echo "======================================================================"
python3 -m experiments.exploratory_play --profile "${PROFILE}" --seeds "${SEEDS_HEADLINE}" --steps "${EXPLORATORY_STEPS}" --outdir results/exploratory-play --config_set default --tag "${PROFILE}" 2>&1 | tee logs/exploratory_play.log
echo "✓ Completed. Log: logs/exploratory_play.log"

# Test 5: Qualiaphilia Familiarity Control
echo ""
echo "======================================================================"
echo "TEST 5/8: Qualiaphilia Familiarity Control (Advanced)"
echo "======================================================================"
python3 -m experiments.familiarity_control --seeds "${SEEDS_HEADLINE}" --post_repeats "${FAMILIARITY_POST_REPEATS}" --outdir results/familiarity 2>&1 | tee logs/familiarity.log
echo "✓ Completed. Log: logs/familiarity.log"

# Test 6: Qualiaphilia Familiarity Internal Control
echo ""
echo "======================================================================"
echo "TEST 6/8: Qualiaphilia Familiarity Internal Control"
echo "======================================================================"
PYTHONPATH=. python3 -m experiments.familiarity_internal 2>&1 | tee logs/familiarity_internal.log
echo "✓ Completed. Log: logs/familiarity_internal.log"

# Test 7: Causal Lesion Test
echo ""
echo "======================================================================"
echo "TEST 7/8: Pain-Tail Causal Lesion Test (Advanced)"
echo "======================================================================"
python3 -m experiments.lesion_causal --seeds "${SEEDS_HEADLINE}" --post_window "${LESION_POST_WINDOW}" --outdir results/lesion 2>&1 | tee logs/lesion.log
echo "✓ Completed. Log: logs/lesion.log"

# Test 8: Publication Figures
echo ""
echo "======================================================================"
echo "TEST 8/8: Creating Publication Figures"
echo "======================================================================"
python3 -m experiments.viz_utils.publication_figures 2>&1 | tee logs/figures.log
echo "✓ Completed. Log: logs/figures.log"

# Ablations: affect readout vs modulation
echo ""
echo "======================================================================"
echo "ABLATIONS: Affect Readout vs Modulation (Corridor + Play)"
echo "======================================================================"
python3 -m experiments.familiarity_control --seeds "${SEEDS_HEADLINE}" --post_repeats "${FAMILIARITY_POST_REPEATS}" --models Ipsundrum+affect,Ipsundrum+affect_readout_only,Ipsundrum+affect_modulation_only --outdir results/ablations/familiarity 2>&1 | tee logs/ablations_familiarity.log
python3 -m experiments.exploratory_play --profile "${PROFILE}" --seeds "${SEEDS_HEADLINE}" --steps "${EXPLORATORY_STEPS}" --outdir results/ablations/exploratory-play --config_set ablation --tag "ablation_${PROFILE}" 2>&1 | tee logs/ablations_play.log
echo "✓ Completed ablations. Logs: logs/ablations_familiarity.log, logs/ablations_play.log"

# Optional weight sweep (default OFF)
if [[ "${RUN_WEIGHT_SWEEP:-0}" == "1" ]]; then
  echo ""
  echo "======================================================================"
  echo "OPTIONAL: Weight Sweep"
  echo "======================================================================"
  python3 -m experiments.weight_sweep --outdir results/ablations/weight-sweep 2>&1 | tee logs/weight_sweep.log
  echo "✓ Completed. Log: logs/weight_sweep.log"
fi

# Save reproducibility metadata
python3 << 'PYEOF'
import json
import os
from experiments.evaluation_harness import MODEL_ORDER_DISPLAY

profile = os.environ.get("PROFILE", "paper")
seeds_headline = int(os.environ.get("SEEDS_HEADLINE", "5"))
goal_directed_seeds = int(os.environ.get("GOAL_DIRECTED_SEEDS", "10"))
pain_tail_seeds = int(os.environ.get("PAIN_TAIL_SEEDS", "20"))
qualiaphilia_seeds = int(os.environ.get("QUALIAPHILIA_SEEDS", "5"))
pain_post_steps = int(os.environ.get("PAIN_POST_STEPS", "50"))
exploratory_steps = int(os.environ.get("EXPLORATORY_STEPS", "200"))
familiarity_post_repeats = int(os.environ.get("FAMILIARITY_POST_REPEATS", "5"))
lesion_post_window = int(os.environ.get("LESION_POST_WINDOW", "150"))

config = {
    "profile": profile,
    "model_order": MODEL_ORDER_DISPLAY,
    "seeds": {
        "goal_directed": list(range(goal_directed_seeds)),
        "pain_tail": list(range(pain_tail_seeds)),
        "qualiaphilia_baseline": list(range(qualiaphilia_seeds)),
        "exploratory_play": list(range(seeds_headline)),
        "familiarity": list(range(seeds_headline)),
        "lesion": list(range(seeds_headline)),
        "familiarity_internal": list(range(5)),
    },
    "horizons": {
        "goal_directed": list(range(1, 21)),
        "exploratory_play": 3,
        "familiarity": 5,
        "familiarity_internal": 5,
    },
    "window_lengths": {
        "exploratory_play_steps": exploratory_steps,
        "familiarity_steps": 80,
        "familiarity_internal_steps": 80,
        "lesion_total_steps": 200,
        "lesion_post_window": lesion_post_window,
        "goal_directed_gridworld_steps": 250,
        "goal_directed_corridor_steps": 80,
        "pain_tail_post_stimulus_steps": pain_post_steps,
        "qualiaphilia_trial_steps": 80,
    },
    "familiarity_post_repeats": familiarity_post_repeats,
    "curiosity": {
        "exploratory_play": True,
        "familiarity": True,
        "goal_directed": False,
    },
    "sensory_entropy": {
        "bins_per_modality": 8,
        "modalities": ["touch", "smell_h", "smell_b", "vision"],
    },
    "lesion": {
        "lesion_time": 3,
        "flags": ["lesion_integrator", "lesion_feedback"],
        "immediacy_metric": "delta_ns5",
    },
    "bootstrap_ci": {
        "ci": 95,
        "n_boot": 2000,
        "seed": 0,
    },
}

os.makedirs("results", exist_ok=True)
with open("results/config_metadata.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
print("Saved results/config_metadata.json")
PYEOF

# Claims-as-code artifacts
python3 -m analysis.paper_claims --outdir results/paper 2>&1 | tee logs/claims.log
python3 -m analysis.export_params --outdir results/paper 2>&1 | tee logs/params.log
python3 -m analysis.build_dissociation_table --outdir results/paper 2>&1 | tee logs/dissociation.log

# Summary
echo ""
echo "======================================================================"
echo "ALL TESTS COMPLETE"
echo "======================================================================"
echo ""
echo "Baseline Assays:"
echo "  - results/pain-tail/episodes.csv, summary.csv"
echo "  - results/qualiaphilia/episodes.csv, summary.csv"
echo ""
echo "Goal-Directed Performance:"
echo "  - results/goal-directed/corridor_*.csv, gridworld_*.csv"
echo ""
echo "Advanced Consciousness Tests:"
echo "  - results/exploratory-play/final_viewpoints.csv"
echo "  - results/familiarity/episodes_improved.csv"
echo "  - results/familiarity/familiarity_internal_*.csv"
echo "  - results/lesion/episodes_extended.csv, ns_traces.npz"
echo ""
echo "Publication Figures:"
echo "  - results/publication-figures/fig1_play.png"
echo "  - results/publication-figures/fig2.png"
echo "  - results/publication-figures/fig3.png"
echo "  - results/publication-figures/debug/fig1_debug.png"
echo "  - results/publication-figures/debug/fig2_debug.png"
echo "  - results/publication-figures/debug/fig3_debug.png"
echo ""
echo "Reproducibility:"
echo "  - results/config_metadata.json"
echo "  - RESULTS_README.md"
echo ""
echo "Claims-as-Code:"
echo "  - results/paper/claims.json"
echo "  - results/paper/claims.tex"
echo "  - results/paper/claims.md"
echo "  - results/paper/params_table.tex"
echo "  - results/paper/params_table.csv"
echo "  - results/paper/dissociation_table.tex"
echo ""
echo "Ablations:"
echo "  - results/ablations/familiarity/episodes_improved.csv"
echo "  - results/ablations/familiarity/summary.csv"
echo "  - results/ablations/exploratory-play/exploratory_play_clarified_ablation_${PROFILE}.csv"
echo "  - results/ablations/exploratory-play/summary_ablation_${PROFILE}.csv"
echo ""
echo "Logs:"
echo "  - logs/goal_directed.log"
echo "  - logs/pain_tail.log (baseline)"
echo "  - logs/qualiaphilia.log (baseline)"
echo "  - logs/exploratory_play.log"
echo "  - logs/familiarity.log (advanced)"
echo "  - logs/lesion.log (advanced)"
echo "  - logs/figures.log"
echo "  - logs/ablations_familiarity.log"
echo "  - logs/ablations_play.log"
echo "  - logs/claims.log"
echo "  - logs/params.log"
echo "  - logs/dissociation.log"
echo ""
