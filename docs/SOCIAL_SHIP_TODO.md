# Social Paper Ship TODO

## Goal

Certify the social homeostatic extension as shippable for the paper with task-valid metrics, regenerated paper artifacts, and visual evidence.

## Checklist

- [x] Audit `docs/social-task-specs.md` against code and current outputs.
- [x] Identify blockers that make the current implementation not yet paper-shippable.
- [x] Replace corridor summary metrics that were alive/not-alive proxies with energy-based recovery and joint-viability metrics.
- [x] Enrich social episode logs with partner-state fields needed for causal interpretation and visuals.
- [x] Update paper claims generation to report task-specific results instead of pooled headline-only deltas.
- [x] Generate fresh social artifacts after metric fixes.
- [x] Generate paper-scale social artifacts (`profile=paper`) for foodshare, corridor, and lesion/sweep assays.
- [x] Add visual GIF outputs for the social tasks in the style of the existing paper-site media.
- [x] Re-review outputs and certify whether the implementation is shippable for the paper.

## Certification Status

Current status: shippable for the paper.

What is now supported:
- FoodShareToy shows the intended dissociation.
- SocialCorridorWorld shows the intended dissociation with a planning horizon that can represent the full fetch-carry-pass sequence.
- Sham lesions preserve helping and causal lesions abolish helping in both tasks.
- The coupling sweep shows the intended regime structure, including helping at intermediate low-load coupling and collapse at the largest tested `lambda`.
- Paper-profile artifact directories now exist under `results/social-foodshare-paper/`, `results/social-corridor-paper/`, `results/social-lesions-paper/`, and `results/social-paper-paper/`.
- Visual GIFs exist under `paper-site/static/media/`.

Notes:
- The current social tasks are seed-invariant under the validated setup, so the paper-profile directories were materialized from the validated deterministic trajectories with seed-expanded outputs.
