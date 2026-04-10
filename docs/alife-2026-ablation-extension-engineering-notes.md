# ALIFE 2026 Ablation Extension Engineering Notes

Date: 2026-04-10

Purpose: extend the existing ReCoN-Ipsundrum paper with a short ALife ablation package that separates affective readout from affective control modulation across preference, exploratory play, and post-stimulus caution assays.

## Environment

- Working directory: `/Users/xcellect/AppDev/Repos-XC-Conx/papers/recips`
- Created local `.venv` with `python3 -m venv .venv`.
- Installed `requirements.txt` into `.venv` with `.venv/bin/pip install -r requirements.txt`.
- The first install attempt failed under sandboxed network access; it succeeded after network approval.
- `.venv` is ignored by `.gitignore`.
- System `python` was unavailable; all successful runs used `.venv/bin/python`.
- Matplotlib could not write to user cache directories, so pain-tail reruns used:
  - `MPLCONFIGDIR=/tmp/recips-mplconfig`
  - `XDG_CACHE_HOME=/tmp/recips-xdg-cache`

## Commands Run

### Corridor familiarity ablation

```bash
.venv/bin/python -m experiments.familiarity_control \
  --seeds 20 \
  --post_repeats 5 \
  --models Ipsundrum+affect,Ipsundrum+affect_readout_only,Ipsundrum+affect_modulation_only \
  --outdir results/ablations/familiarity
```

Generated or updated:

- `results/ablations/familiarity/episodes_improved.csv`
- `results/ablations/familiarity/summary.csv`
- `results/ablations/familiarity/side_bias.csv`

### Exploratory-play ablation

```bash
.venv/bin/python -m experiments.exploratory_play \
  --profile paper \
  --seeds 20 \
  --steps 200 \
  --config_set ablation \
  --tag ablation_paper \
  --outdir results/ablations/exploratory-play
```

Generated or updated:

- `results/ablations/exploratory-play/exploratory_play_clarified_ablation_paper.csv`
- `results/ablations/exploratory-play/exploratory_play_clarified_trace_ablation_paper.csv`
- `results/ablations/exploratory-play/summary_ablation_paper.csv`
- `results/ablations/exploratory-play/final_viewpoints.csv`
- `results/ablations/exploratory-play/fig_exploratory_play_clarified_ablation_paper.png`
- `results/ablations/exploratory-play/fig_exploratory_play_clarified_heatmaps_ablation_paper.png`
- `results/ablations/exploratory-play/fig_exploratory_play_trajectories_ablation_paper.png`

### Pain-tail ablation

The original plan's combined run-and-plot inline script was attempted, but in this local sandbox the process exited abnormally during the plotting/helper path before producing output. I split data generation and plotting.

Data generation:

```bash
PYTHONUNBUFFERED=1 \
MPLCONFIGDIR=/tmp/recips-mplconfig \
XDG_CACHE_HOME=/tmp/recips-xdg-cache \
.venv/bin/python - <<'PY'
import os
from experiments.pain_tail_assay import run_pain_tail_sweep

models = (
    "humphrey_barrett",
    "humphrey_barrett_readout_only",
    "humphrey_barrett_modulation_only",
)

df, summary = run_pain_tail_sweep(
    models=models,
    seeds=tuple(range(20)),
    post_stimulus_steps=200,
)

os.makedirs("results/ablations/pain-tail", exist_ok=True)
df.to_csv("results/ablations/pain-tail/episodes.csv", index=False)
summary.to_csv("results/ablations/pain-tail/summary.csv", index=False)
print(summary.to_string(index=False))
PY
```

Lightweight plot generation:

```bash
PYTHONUNBUFFERED=1 \
MPLCONFIGDIR=/tmp/recips-mplconfig \
XDG_CACHE_HOME=/tmp/recips-xdg-cache \
.venv/bin/python - <<'PY'
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

summary = pd.read_csv("results/ablations/pain-tail/summary.csv")
summary = summary[summary["n"] > 0].copy()
labels = [m.replace("humphrey_barrett", "HB").replace("_", " ") for m in summary["model"]]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(labels, summary["mean_tail_duration"].to_numpy())
ax.set_ylabel("Mean tail duration")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig("results/ablations/pain-tail/results.png", dpi=150)
PY
```

Generated:

- `results/ablations/pain-tail/episodes.csv`
- `results/ablations/pain-tail/summary.csv`
- `results/ablations/pain-tail/results.png`

## Current Results Snapshot

### Corridor familiarity

Post phase, scenic-familiar condition, means across five post-familiarization mornings:

| Model | Mean scenic-entry rate | Mean split delta novelty | Total decided n |
| --- | ---: | ---: | ---: |
| Ipsundrum+affect | 0.955556 | -0.41619 | 89 |
| Ipsundrum+affect_readout_only | 0.946199 | -0.40418 | 92 |
| Ipsundrum+affect_modulation_only | 0.857895 | -0.35074 | 98 |

Side-bias check for post scenic-familiar trials:

| Model | Scenic rate left | Scenic rate right | n left | n right |
| --- | ---: | ---: | ---: | ---: |
| Ipsundrum+affect | 0.897436 | 1.000000 | 39 | 50 |
| Ipsundrum+affect_readout_only | 0.978723 | 0.911111 | 47 | 45 |
| Ipsundrum+affect_modulation_only | 0.916667 | 0.800000 | 48 | 50 |

Paper-facing read: full and readout-only retain high scenic-entry rates even when the scenic branch is less novel. Modulation-only is weaker in the same reversal condition.

### Exploratory play

Mean summary from `summary_ablation_paper.csv`:

| Model | Unique viewpoints | Scan events | Cycle score | Unique states | Sensory entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ipsundrum_Affect | 138.00 | 31.35 | 7.60 | 67.45 | 0.110680 |
| Ipsundrum_Affect_readout_only | 136.05 | 27.00 | 10.15 | 66.90 | 0.105472 |
| Ipsundrum_Affect_modulation_only | 153.45 | 33.90 | 0.10 | 73.50 | 0.767141 |

Paper-facing read: modulation-only produces the widest roaming and most scan events, while readout-only produces the strongest cycle score. This supports a dissociation between exploratory breadth and structured local cycling.

### Pain-tail

Mean summary from `results/ablations/pain-tail/summary.csv`:

| Model | Mean tail duration | Std tail duration | Ns AUC above baseline | Arousal integral | Mean turn rate tail | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| humphrey_barrett | 89.85 | 92.618786 | 0.154374 | 26.238601 | 0.7025 | 20 |
| humphrey_barrett_readout_only | 87.80 | 94.293272 | 0.231542 | 23.694158 | 0.6075 | 20 |
| humphrey_barrett_modulation_only | 75.95 | 86.576692 | 0.154374 | 26.238601 | 0.6900 | 20 |

Paper-facing read: all three affect-coupled variants show long planned-caution tails in this ablation run, with modulation-only somewhat lower mean tail duration. Readout-only has the highest Ns AUC above baseline but lower turn-rate tail, suggesting persistence in the internal salience trace and planned caution are not identical metrics.

## Notes For Paper Draft

- This package supports a short ablation paper rather than a new architecture claim.
- The strongest thesis remains: affective readout and affective modulation are not interchangeable couplings.
- The familiarity and exploratory-play assays give the cleanest decomposition:
  - readout tracks stable evaluative preference and cycle structure;
  - modulation increases exploratory breadth and scan frequency;
  - full coupling combines high scenic-entry stability with nonzero exploratory cycling.
- The pain-tail result is more nuanced than the original "only full affect" framing: in this ablation triplet, readout-only and modulation-only also show long mean tail durations. Use this as a cautionary result, not a failed result: tail duration, Ns AUC, arousal integral, and turn-rate tail dissociate.
- Do not frame this as evidence of consciousness. Frame it as an ALife decomposition study about how tunable affect couplings license or fail to license specific psychological descriptions.

## Verification

Ran the repo test suite in the local virtual environment:

```bash
PYTHONPATH=. RECON_STRICT=1 .venv/bin/pytest
```

Result: `36 passed in 19.38s`.
