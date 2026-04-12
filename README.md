<h1 align="center">ReCoN-Ipsundrum</h1>

<p align="center">
  An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Control and Mechanism-Linked Consciousness Indicator Assays
</p>

<p align="center">
  <a href="https://xcellect.com">Aishik Sanyal</a>
</p>

<p align="center">
  <a href="https://xcellect.github.io/recips/">Paper Website</a> |
  <a href="https://arxiv.org/pdf/2602.23232.pdf">Paper PDF</a> |
  <a href="https://arxiv.org/abs/2602.23232">ArXiv Page</a> |
  <a href="https://colab.research.google.com/github/xcellect/recips/blob/main/playground.ipynb">Colab Demo</a> (no setup required)
</p>

Welcome to the ReCoN-Ipsundrum codebase. This repository now supports reproduction for two related papers built on the same inspectable recurrent-control framework:

- the original AAAI 2026 submission in `docs/recips_aaai2026/`
- the ALIFE 2026 social extension in `docs/recips_social_alife2026/`

The repository contains the core agent implementation, assay definitions, experiment generation and analysis scripts, paper sources, and the static paper website. The structure of the original paper was inspired by [Wolfram's Computational Essay](https://writings.stephenwolfram.com/2017/11/what-is-a-computational-essay/).

## Contents

- [Setup](#setup)
- [Paper-Specific Reproduction](#paper-specific-reproduction)
- [Run the full experiment suite](#run-the-full-experiment-suite)
- [Run tests](#run-tests)
- [Repository layout](#repository-layout)
- [Build the paper PDFs](#build-the-paper-pdfs-optional)
- [Build the paper website](#build-the-paper-website-optional)
- [Abstract](#abstract)
- [BibTeX](#bibtex)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Paper-Specific Reproduction

This repository contains two manuscript tracks with different reproduction entry points.

### AAAI 2026: ReCoN-Ipsundrum

The original paper source lives in `docs/recips_aaai2026/paper-3-v9.tex`. The main reproduction path is still the repository-wide runner:

```bash
./run_experiments.sh
```

That pipeline produces the baseline artifacts used by the AAAI paper under `results/`, including:

- `results/goal-directed/`
- `results/qualiaphilia/`
- `results/exploratory-play/`
- `results/familiarity/`
- `results/pain-tail/`
- `results/lesion/`
- `results/paper/`

If you only want a faster smoke run, use:

```bash
PROFILE=quick ./run_experiments.sh
```

### ALIFE 2026: Social Homeostatic Coupling Extension

The social-extension manuscript source lives in `docs/recips_social_alife2026/main.tex`. Its headline claim is narrower and should be reproduced from the dedicated social runners rather than inferred from the non-social AAAI pipeline.

The ALIFE paper evaluates four matched conditions:

- `social_none`
- `social_cognitive_direct`
- `social_affective_direct`
- `social_full_direct`

across two tasks:

- `FoodShareToy`
- `SocialCorridorWorld`

plus lesions (`sham`, `coupling_off`, `shuffle_partner`) and a coupling sweep over `lambda_affective` and metabolic load.

Important: the module entry points `python3 -m experiments.social_foodshare`, `python3 -m experiments.social_corridor`, and `python3 -m experiments.social_lesion_assay` default to the repository's `quick` profile in their `__main__` blocks. For paper-grade reproduction of the ALIFE manuscript, call the experiment functions explicitly with `profile="paper"`:

```bash
python3 -c 'from experiments.social_foodshare import run_foodshare_experiment; run_foodshare_experiment(profile="paper", outdir="results/social-foodshare-paper")'
python3 -c 'from experiments.social_corridor import run_corridor_experiment; run_corridor_experiment(profile="paper", outdir="results/social-corridor-paper", metabolic_load="low")'
python3 -c 'from experiments.social_lesion_assay import run_social_lesion_assay; run_social_lesion_assay(profile="paper", outdir="results/social-lesions-paper")'
python3 -c 'from analysis.social_claims import build_social_artifacts; build_social_artifacts(foodshare_summary_csv="results/social-foodshare-paper/summary.csv", corridor_summary_csv="results/social-corridor-paper/summary.csv", lesion_summary_csv="results/social-lesions-paper/summary.csv", coupling_sweep_csv="results/social-lesions-paper/coupling_sweep.csv", outdir="results/social-paper-paper")'
```

Those commands generate the manuscript-level social artifacts:

- `results/social-foodshare-paper/episodes.csv`, `summary.csv`, `exact_threshold.csv`
- `results/social-corridor-paper/episodes.csv`, `summary.csv`
- `results/social-lesions-paper/summary.csv`, `coupling_sweep.csv`
- `results/social-paper-paper/headline_summary.csv`
- `results/social-paper-paper/lesion_summary.csv`
- `results/social-paper-paper/coupling_sweep.csv`
- `results/social-paper-paper/claims.json`

The corresponding manuscript figures can then be regenerated with:

```bash
python3 -m experiments.viz_utils.social_paper_figures --social-paper-dir results/social-paper-paper --out-pdf docs/recips_social_alife2026/figures/fig_summary.pdf --out-png docs/recips_social_alife2026/figures/fig_summary.png
python3 -m experiments.viz_utils.social_visuals_figure --out-pdf docs/recips_social_alife2026/figures/fig_visuals.pdf --out-png docs/recips_social_alife2026/figures/fig_visuals.png
```

For ALIFE-specific regression checks, run:

```bash
python3 -m pytest -q tests/test_social_homeostat.py tests/test_social_forward.py tests/test_social_foodshare.py tests/test_social_lesions.py
```

## Run the full experiment suite

`run_experiments.sh` executes the main repository pipeline and writes artifacts to `results/` and logs to `logs/`. It remains the primary runner for the AAAI 2026 paper and now also includes the social experiment modules as part of the repository-wide sweep.

```bash
# Full (paper) profile (more seeds; default)
./run_experiments.sh

# Faster smoke run
PROFILE=quick ./run_experiments.sh
```

Note: the script deletes and recreates `results/` and `logs/` at startup.

For the ALIFE paper specifically, use the dedicated commands in [Paper-Specific Reproduction](#paper-specific-reproduction), because the social module `__main__` entry points default to `quick` unless you call their functions explicitly with `profile="paper"`.

## Run tests

```bash
./run_pytest.sh
```

## Repository layout

- `core/`: ReCoN primitives and Ipsundrum(+affect) dynamics.
- `experiments/`: Assays and figure generation (used by `run_experiments.sh`).
- `analysis/`: “Claims-as-code” exports used by the paper and tables.
- `docs/recips_aaai2026/`: AAAI 2026 manuscript source and compiled paper.
- `docs/recips_social_alife2026/`: ALIFE 2026 social-extension manuscript, figures, and supplementary GIFs.

## Build the paper PDFs (optional)

Requires a LaTeX toolchain (e.g. `latexmk`).

For the AAAI 2026 paper:

```bash
cd docs/recips_aaai2026
latexmk -pdf paper-3-v9.tex
```

For the ALIFE 2026 social paper:

```bash
cd docs/recips_social_alife2026
latexmk -pdf main.tex
```

## Build the paper website (optional)

The static paper website lives in `paper-site/` and is generated from the current `results/` artifacts plus a few exported GIFs.

```bash
python3 -m analysis.build_paper_site
```

That command writes:

- `paper-site/static/data/site-data.json`
- `paper-site/static/media/*`

To preview locally:

```bash
python3 -m http.server 8000 --directory paper-site
```

A GitHub Pages workflow is included at `.github/workflows/deploy-paper-site.yml` and rebuilds/deploys the page from `main`.

## Abstract

Indicator-based approaches to machine consciousness recommend mechanism-linked evidence triangulated
across tasks, supported by architectural inspection and causal intervention. Inspired by Humphrey's
ipsundrum hypothesis, we implement ReCoN-Ipsundrum, an inspectable agent that extends a ReCoN state
machine with a recurrent persistence loop over sensory salience Ns and an optional affect proxy
reporting valence/arousal. Across fixed-parameter ablations (ReCoN, Ipsundrum, Ipsundrum+affect), we
operationalize Humphrey's qualiaphilia (preference for sensory experience for its own sake) as a
familiarity-controlled scenic-over-dull route choice. We find a novelty dissociation: non-affect
variants are novelty-sensitive (Delta scenic-entry = 0.07). Affect coupling is stable (Delta
scenic-entry = 0.01) even when scenic is less novel (median Delta novelty approx. -0.43). In
reward-free exploratory play, the affect variant shows structured local investigation (scan events
31.4 vs. 0.9; cycle score 7.6). In a pain-tail probe, only the affect variant sustains prolonged
planned caution (tail duration 90 vs. 5). Lesioning feedback+integration selectively reduces
post-stimulus persistence in ipsundrum variants (AUC drop 27.62, 27.9%) while leaving ReCoN
unchanged. These dissociations link recurrence to persistence and affect-coupled control to
preference stability, scanning, and lingering caution, illustrating how indicator-like signatures can
be engineered and why mechanistic and causal evidence should accompany behavioral markers.

## BibTeX

```bibtex
@misc{sanyal2026reconipsundrum,
  title         = {ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Control and Mechanism-Linked Consciousness Indicator Assays},
  author        = {Aishik Sanyal},
  year          = {2026},
  eprint        = {2602.23232},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2602.23232},
  url           = {https://arxiv.org/abs/2602.23232},
  note          = {Accepted at AAAI 2026 Spring Symposium - Machine Consciousness: Integrating Theory, Technology, and Philosophy}
}
```
