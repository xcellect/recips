<h1 align="center">ReCoN-Ipsundrum</h1>

<p align="center">
  An inspectable recurrent persistence loop agent with affect-coupled control and mechanism-linked consciousness indicator assays.
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

Welcome to the ReCoN-Ipsundrum codebase. This repository contains the core agent implementation, assay definitions, experiment generation and analysis scripts, as well as the paper source and the static paper website. The structure of this paper is inspired by [Wolfram's Computational Essay](https://writings.stephenwolfram.com/2017/11/what-is-a-computational-essay/).

## Contents

- [Setup](#setup)
- [Run the full experiment suite](#run-the-full-experiment-suite)
- [Run tests](#run-tests)
- [Repository layout](#repository-layout)
- [Build the paper PDF](#build-the-paper-pdf-optional)
- [Build the paper website](#build-the-paper-website-optional)
- [Abstract](#abstract)
- [BibTeX](#bibtex)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the full experiment suite

`run_experiments.sh` executes the complete pipeline and writes artifacts to `results/` and logs to `logs/`.

```bash
# Full (paper) profile (more seeds; default)
./run_experiments.sh

# Faster smoke run
PROFILE=quick ./run_experiments.sh
```

Note: the script deletes and recreates `results/` and `logs/` at startup.

## Run tests

```bash
./run_pytest.sh
```

## Repository layout

- `core/`: ReCoN primitives and Ipsundrum(+affect) dynamics.
- `experiments/`: Assays and figure generation (used by `run_experiments.sh`).
- `analysis/`: “Claims-as-code” exports used by the paper and tables.
- `docs/paper-3-v9.tex`: Paper source (optionally includes `results/paper/claims.tex` if present).

## Build the paper PDF (optional)

Requires a LaTeX toolchain (e.g. `latexmk`). Running the experiments will automatically populate paper results.

```bash
cd docs
latexmk -pdf paper-3-v9.tex
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

```
Indicator-based approaches to machine consciousness recommend mechanism-linked evidence triangulated across tasks, supported by architectural inspection and causal intervention. Inspired by Humphrey's ipsundrum hypothesis, we implement ReCoN-Ipsundrum, an inspectable agent that extends a ReCoN state machine with a recurrent persistence loop over sensory salience Ns and an optional affect proxy reporting valence/arousal. Across fixed-parameter ablations (ReCoN, Ipsundrum, Ipsundrum+affect), we operationalize Humphrey's qualiaphilia (preference for sensory experience for its own sake) as a familiarity-controlled scenic-over-dull route choice. We find a novelty dissociation: non-affect variants are novelty-sensitive (Delta scenic-entry = 0.07). Affect coupling is stable (Delta scenic-entry = 0.01) even when scenic is less novel (median Delta novelty approx. -0.43). In reward-free exploratory play, the affect variant shows structured local investigation (scan events 31.4 vs. 0.9; cycle score 7.6). In a pain-tail probe, only the affect variant sustains prolonged planned caution (tail duration 90 vs. 5). Lesioning feedback+integration selectively reduces post-stimulus persistence in ipsundrum variants (AUC drop 27.62, 27.9%) while leaving ReCoN unchanged. These dissociations link recurrence to persistence and affect-coupled control to preference stability, scanning, and lingering caution, illustrating how indicator-like signatures can be engineered and why mechanistic and causal evidence should accompany behavioral markers.
```

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