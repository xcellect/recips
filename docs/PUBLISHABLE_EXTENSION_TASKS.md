# Publishable Extension Tasks

## Goal

Make the new-model extension internally consistent as a paper-facing artifact: assays should match the task spec closely enough to support bounded claims, and claim/dissociation outputs should not mark unsupported results as passing.

## Tasks

- [x] Fix paper claim pass/fail rendering so directional comparison claims only pass when their directional criterion is met.
- [x] Fix dissociation-table construction so it respects claim pass status rather than only checking whether a numeric estimate is positive.
- [x] Repair the context-fork assay.
  Current state:
  - `R` uses full latent vectors at the matched fork observation.
  - branch scoring uses a context-anchor latent retention margin instead of `score_internal`.
  - long-delay context-memory comparisons now separate the tuned perspective family from `gw_lite` on the intended representation metric.
- [x] Tune the perspective-family dynamics to restore the intended slow-latent signatures.
  Implemented calibration:
  - increase `alpha_p` from `0.05` to `0.08`
  - strengthen `W_zp`, `W_pp`, and `W_pz` scales in `make_perspective_params`
- [x] Add regression tests for the repaired context-fork metric/claim plumbing.
- [x] Regenerate targeted paper artifacts affected by the fixes.

## Findings

- The implementation satisfies the additive architecture requirements: new model ids, pure kernels, comparator parity, latent logging, assays, and targeted tests are present.
- The main initial blockers were assay validity and paper-output consistency, not missing core model code.
- After the claim/assay fixes and perspective-model calibration, the key paper-facing signatures now line up with the intended extension story.

## Current Paper Status

- `claim_hysteresis_perspective_gt_scalar`: PASS
- `claim_perspective_lesion_selective`: PASS
- `claim_context_delay_perspective_plastic_gt_gw`: PASS
- `claim_plasticity_residue_gt_no_plastic`: PASS
- `claim_selector_lesion_selective`: PASS
- `claim_continuity_no_regression_pain_tail`: PASS
- `claim_continuity_no_regression_familiarity_choice`: PASS

## Residual Caution

- Some directional claims remain narrow or have wide intervals despite passing, so this is better framed as a bounded mechanistic/computational-experiment result than a broad empirical victory claim.
- `gw_lite` still dominates the explicit conflict-robustness story, which is desirable for the comparator-side dissociation.

## Progress Log

- 2026-04-11: Created tracker from implementation review.
- 2026-04-11: Repaired claim pass/fail logic so directional comparison claims respect `meta.pass`.
- 2026-04-11: Repaired dissociation-table logic so unsupported comparison claims no longer appear as positive signatures.
- 2026-04-11: Rewrote the context-fork assay to compute `R` in full latent space and use context-anchor branch margins.
- 2026-04-11: Added regression tests for context-fork helpers and claim pass logic.
- 2026-04-11: Tuned the perspective-family slow-latent dynamics to increase hysteresis, lesion selectivity, and long-delay context separation.
- 2026-04-11: Regenerated `results/hysteresis-probe/*`, `results/context-fork/*`, `results/paper/claims.*`, and `results/paper/dissociation_table.tex`.
