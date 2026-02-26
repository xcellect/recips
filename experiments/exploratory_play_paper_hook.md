# Exploratory Play assay: paper integration

Include these figures in the paper:
- `results/publication-figures/fig1_play.png` (PLAY summary: entropy + coverage + scan + control)
- `results/exploratory-play/fig_exploratory_play_trajectories.png` (time-ordered trajectories, seed=0)
- Optional: `results/exploratory-play/fig_exploratory_play_dwell.png` (dwell distribution)

Update the Results paragraph to mention:
- Neutral texture entropy (or viewpoint entropy if neutral is disabled), plus unique_viewpoints.
- Scan structure (scan_events / scan_depth) alongside true_stall_rate and dwell_p90.
- Loopiness control (cycle_score or tortuosity) and boundary_hugging_fraction.
- Hazard contacts to show entropy changes are not just hazard avoidance.

Assay note:
- The neutral texture channel is assay-only (not part of I_total), used to de-confound hazard avoidance.
