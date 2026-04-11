# Engineering Notes

## Progress

- [x] Inspect task spec and current extension points.
- [x] Add `perspective`, `perspective_plastic`, and `gw_lite` pure dynamics/forward modules.
- [x] Integrate new model ids into harness and main experiment code paths.
- [x] Add assay scripts for hysteresis, context fork, and multimodal conflict.
- [x] Add claims/dissociation updates.
- [x] Add and run targeted tests.

## Current architecture

1. Reuse the Stage-D shell and visible node names (`Ns`, `Ne`, `Ni`, `Nv`, `Na`).
2. Attach alternate pure one-step updates through `net._update_ipsundrum_sensor`.
3. Keep policy compatibility via `.params`, `.affect`, `score_weights`, and `net._ipsundrum_state`.
4. Flatten latent arrays via `core/model_factory.py::flatten_latent_state` for per-step logging.

## Remaining work order

1. Optional future refinement: calibrate stronger paper-grade assay thresholds and selection criteria once more sweep data is collected.
2. Optional future refinement: deepen lesion interpretation for the new model families beyond the current runner-level selective lesion hooks.

## Latest verification

- `pain_tail_assay.py` now accepts `--models` and runs `perspective_plastic` / `gw_lite` successfully.
- `familiarity_control.py` runs `perspective_plastic` / `gw_lite` successfully after structured-observation compatibility patches.
- Legacy continuity/lesion/qualiaphilia code paths were patched to pass structured observation tuples into new forward/update hooks.
- `goal_directed_sweeps.py` now accepts `--models`; `run_experiments.sh` was narrowed to `recon,humphrey,humphrey_barrett` for that legacy benchmark so `PROFILE=quick` stays aligned with the requested continuity scope.
- The corrected `PROFILE=quick` pipeline was executed through the earlier stages and then resumed from the fixed familiarity-internal stage through paper artifact generation.
- Final paper outputs now exist in `results/paper/`, including `claims.json`, `claims.md`, `claims.tex`, `params_table.*`, and `dissociation_table.tex`.
- New assay scripts now emit `trace.csv` files plus trace-summary figures, and `run_experiments.sh` invokes `latent_viz.py` to generate trajectory/selector plots from those traces.
- `lesion_causal.py` now applies model-specific lesions to `perspective`, `perspective_plastic`, and `gw_lite`, and defaults to all model families when no explicit `--models` filter is provided.
