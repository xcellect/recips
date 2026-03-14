# Twitter Thread Asset Pack

This folder contains a six-asset social pack built directly from the current `recips` repo and paper results.

Order:
1. `01_hero_same-task-different-internals.gif`
2. `02_architecture_same-substrate-three-interventions.png`
3. `03_novelty-seeking-is-not-preference.png`
4. `04_same-arena-very-different-play-style.png`
5. `05_persistence-alone-isnt-enough.png`
6. `06_cut-the-loop-and-the-signature-collapses.gif`

Key paper-linked numbers used on the cards:
- Familiarity control: ReCoN delta scenic-entry = 0.07
- Familiarity control: Ipsundrum delta scenic-entry = 0.07
- Familiarity control: Ipsundrum+affect delta scenic-entry = 0.01
- Affect novelty note: median delta novelty = -0.43
- Exploratory play: affect scan events = 31.4
- Exploratory play: ReCoN scan events = 0.9
- Exploratory play: affect cycle score = 7.6
- Pain-tail duration: affect = 90
- Pain-tail duration: ReCoN = 5
- Lesion AUC drop: affect = 27.9%

Primary sources in repo:
- `docs/paper-3-v9.tex`
- `results/paper/claims.json`
- `results/familiarity/episodes_improved.csv`
- `results/exploratory-play/exploratory_play_clarified_trace_paper.csv`
- `results/pain-tail/summary.csv`
- `results/lesion/ns_traces.npz`

Generation:
- Re-run with `python3 analysis/build_social_assets.py`
