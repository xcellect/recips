# Files Manifest - ReCoN-Ipsundrum Evaluation Suite

**Last Updated**: 2026-01-30  

This repository is organized around a single “run everything” entrypoint, `run_experiments.sh`, which runs all assays, generates figures, exports “claims-as-code” artifacts for the paper, and records the run configuration in `results/config_metadata.json`.

---

## 0. Quickstart

```bash
./run_experiments.sh
```

Outputs are written to `results/` and logs to `logs/`.

> Note: `run_experiments.sh` deletes and recreates `results/` and `logs/` at startup.

---

## I. Master Script (`run_experiments.sh`)

### Profiles

The script supports two profiles via `PROFILE`:

- `PROFILE=paper` (default): higher seed counts for headline experiments.
- `PROFILE=quick`: faster smoke run with fewer headline seeds.

Defaults by profile:

| Variable | `paper` default | `quick` default |
|---|---:|---:|
| `SEEDS_HEADLINE` | 20 | 5 |
| `GOAL_DIRECTED_SEEDS` | 20 | 10 |
| `QUALIAPHILIA_SEEDS` | 20 | 5 |
| `PAIN_TAIL_SEEDS` | 20* | 20* |
| `PAIN_POST_STEPS` | 200 | 50 |
| `EXPLORATORY_STEPS` | 200 | 200 |
| `FAMILIARITY_POST_REPEATS` | 5 | 5 |
| `LESION_POST_WINDOW` | 150 | 150 |

\* `PAIN_TAIL_SEEDS` is overridable via the environment: `PAIN_TAIL_SEEDS=... ./run_experiments.sh`.

### ReCoN strictness toggle

`RECON_STRICT` controls strict vs compatibility behavior in the ReCoN state machine:

- Default: strict (`RECON_STRICT=1` if unset).
- Compatibility mode: `RECON_STRICT=0 ./run_experiments.sh`.

### Optional runs

- `RUN_WEIGHT_SWEEP=1 ./run_experiments.sh` runs `experiments.weight_sweep` into `results/ablations/weight-sweep/` and logs to `logs/weight_sweep.log`.

---

## II. Entrypoints executed by `run_experiments.sh`

### Goal-directed sweeps

- `python3 -m experiments.goal_directed_sweeps`
- Outputs: `results/goal-directed/{corridor,gridworld}_{episodes,summary}.csv` and plots (`*_success.png`, `*_hazards.png`, `*_time.png`).

### Baseline assays

- `python3 -m experiments.pain_tail_assay`
  - Outputs: `results/pain-tail/episodes.csv`, `results/pain-tail/summary.csv`, `results/pain-tail/results.png`.
- `python3 -m experiments.qualiaphilia_assay`
  - Outputs: `results/qualiaphilia/episodes.csv`, `results/qualiaphilia/summary.csv`, `results/qualiaphilia/results.png`.

### Advanced assays

- `python3 -m experiments.exploratory_play`
  - Outputs include `results/exploratory-play/final_viewpoints.csv` plus tagged CSVs/plots (e.g. `*_paper.csv`, `summary_paper.csv`, `fig_*.png`).
- `python3 -m experiments.familiarity_control`
  - Outputs: `results/familiarity/episodes_improved.csv`, `results/familiarity/summary.csv`, `results/familiarity/side_bias.csv`.
- `PYTHONPATH=. python3 -m experiments.familiarity_internal`
  - Outputs: timestamped `results/familiarity/familiarity_internal_*.csv` and `results/familiarity/familiarity_internal_*.png`.
- `python3 -m experiments.lesion_causal`
  - Outputs: `results/lesion/episodes_extended.csv`, `results/lesion/ns_traces.npz`.

### Publication figures

- `python3 -m experiments.viz_utils.publication_figures`
- Outputs: `results/publication-figures/fig1_play.png`, `fig2.png`, `fig2_supp.png`, `fig3.png` and debug versions in `results/publication-figures/debug/`.

### Ablations (affect readout vs modulation)

- Familiarity ablations: `results/ablations/familiarity/{episodes_improved.csv,summary.csv,side_bias.csv}`
- Exploratory-play ablations: `results/ablations/exploratory-play/*` (tagged `*_ablation_paper.*` + plots + `final_viewpoints.csv`)

---

## III. Analysis exports (“claims-as-code”)

These are produced at the end of `run_experiments.sh`:

- `python3 -m analysis.paper_claims --outdir results/paper`
  - Outputs: `results/paper/claims.json`, `results/paper/claims.tex`, `results/paper/claims.md`.
- `python3 -m analysis.export_params --outdir results/paper`
  - Outputs: `results/paper/params_table.tex`, `results/paper/params_table.csv`.
- `python3 -m analysis.build_dissociation_table --outdir results/paper`
  - Outputs: `results/paper/dissociation_table.tex`.

`docs/paper-3-v9.tex` includes `../results/paper/claims.tex` if present.

---

## IV. Code Layout

- `core/`: ReCoN primitives, Ipsundrum(+affect) dynamics, and environment/driver utilities.
- `experiments/`: Assays, harness, and figure generation.
- `analysis/`: CSV → paper artifact exporters (claims, params, dissociation table).
- `utils/`: Plot style + display naming helpers.
- `tests/`: Unit tests; run via `./run_pytest.sh`.

---

## V. Output Structure (`results/` + `logs/`)

Key outputs created by `run_experiments.sh`:

```
results/
├── config_metadata.json
├── goal-directed/...
├── pain-tail/...
├── qualiaphilia/...
├── exploratory-play/...
├── familiarity/...
├── lesion/...
├── publication-figures/...
├── ablations/...
└── paper/...
```

Logs (tee’d from each stage):

```
logs/
├── goal_directed.log
├── pain_tail.log
├── qualiaphilia.log
├── exploratory_play.log
├── familiarity.log
├── familiarity_internal.log
├── lesion.log
├── figures.log
├── ablations_familiarity.log
├── ablations_play.log
├── weight_sweep.log (only if RUN_WEIGHT_SWEEP=1)
├── claims.log
├── params.log
└── dissociation.log
```

---

## VI. Reproducibility metadata (factual run config)

`results/config_metadata.json` records the exact seed lists, horizons, window lengths, and bootstrap settings used for the last `run_experiments.sh` run in this workspace.

For example, the current `results/config_metadata.json` corresponds to `PROFILE=paper` with:

- Headline seeds: 20 (`SEEDS_HEADLINE`)
- Goal-directed horizons: 1–20 (`experiments.goal_directed_sweeps`)
- Pain-tail post window: 200 steps (`PAIN_POST_STEPS`)
- Familiarity post repeats: 5 (`FAMILIARITY_POST_REPEATS`)
- Lesion post window: 150 (`LESION_POST_WINDOW`)

---

## VII. Current metrics snapshot (from `results/paper/claims.json`)

These values are computed by `analysis.paper_claims` from the CSV outputs and will change if you rerun the suite.

| Area | Claim | Value |
|---|---|---:|
| Exploratory play | `play_scan_events_recon` | 0.95 |
| Exploratory play | `play_scan_events_hb` | 31.35 |
| Familiarity | `fam_delta_scenic_entry_recon` (dull − scenic) | 0.07 |
| Familiarity | `fam_delta_scenic_entry_hb` (dull − scenic) | 0.01 |
| Pain-tail | `pain_tail_duration_recon` | 5 |
| Pain-tail | `pain_tail_duration_hb` | 90 |
| Lesion | `lesion_auc_drop_recon` | 0.00 |
| Lesion | `lesion_auc_drop_hb` | 27.62 |
