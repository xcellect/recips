# ALife Extension Engineering Notes

## Scope

Mechanism-decomposition extension for the ALife paper: split affect coupling into policy readout versus recurrent-loop modulation and compare the resulting behavior across familiarity control, exploratory play, lesion, and pain-tail assays.

## Code Changes

- Added `--models` CLI support to `experiments/pain_tail_assay.py` so the ablation can be run directly without a wrapper.
- Extended `run_experiments.sh` so the ablation batch now covers familiarity, exploratory play, lesion, and pain-tail.
- Added `analysis/alife_mechanism_decomposition.py` to compile paper-facing summary tables and markdown notes from the raw assay outputs.

## Assay Sources

- Familiarity summary source: `results/extensions/alife_mech_decomp/familiarity/summary.csv`
- Exploratory play summary source: `results/extensions/alife_mech_decomp/exploratory-play/summary_alife_extension_quick.csv`
- Lesion episodes source: `results/extensions/alife_mech_decomp/lesion/episodes_extended.csv`
- Pain-tail summary source: `results/extensions/alife_mech_decomp/pain-tail/summary.csv`

## Commands Run

```bash
MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python -m experiments.familiarity_control --seeds 10 --post_repeats 5 --models humphrey_barrett,humphrey_barrett_readout_only,humphrey_barrett_modulation_only --outdir results/extensions/alife_mech_decomp/familiarity
MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python -m experiments.exploratory_play --profile quick --seeds 10 --steps 200 --config_set ablation --tag alife_extension_quick --outdir results/extensions/alife_mech_decomp/exploratory-play
MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python -m experiments.lesion_causal --seeds 10 --post_window 150 --models humphrey_barrett,humphrey_barrett_readout_only,humphrey_barrett_modulation_only --outdir results/extensions/alife_mech_decomp/lesion
MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python -m experiments.pain_tail_assay --seeds 10 --post_steps 200 --models humphrey_barrett,humphrey_barrett_readout_only,humphrey_barrett_modulation_only --outdir results/extensions/alife_mech_decomp/pain-tail
MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python -m analysis.alife_mechanism_decomposition --root_dir results/extensions/alife_mech_decomp --outdir results/extensions/alife_mech_decomp/summary
```

## High-Signal Readouts

### Familiarity

| model | familiarize_side | scenic_rate_entry | scenic_time_share_barrier | split_delta_novelty | mean_valence_scenic | mean_valence_dull | mean_arousal_scenic | mean_arousal_dull | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ipsundrum | both | nan | nan | nan | nan | nan | nan | nan | 0 |
| Ipsundrum | dull | nan | nan | nan | nan | nan | nan | nan | 0 |
| Ipsundrum | scenic | nan | nan | nan | nan | nan | nan | nan | 0 |
| Ipsundrum+affect | both | 0.9111111111111111 | 0.9111111111111111 | -0.00978590063361124 | 0.7477444272199257 | 0.5802739148649216 | 0.10257698618103991 | 0.13443334755019054 | 44 |
| Ipsundrum+affect | dull | 0.9111111111111111 | 0.9111111111111111 | 0.18556396249846224 | 0.7473434641469704 | 0.5802739148649216 | 0.10296114879571774 | 0.13443334755019054 | 45 |
| Ipsundrum+affect | scenic | 0.9111111111111111 | 0.9111111111111111 | -0.40506606315040256 | 0.7477444272199257 | 0.5753380797457979 | 0.10257698618103991 | 0.135709812403629 | 44 |
| Ipsundrum+affect_modulation_only | both | 0.86 | 0.86 | -0.00664844170787844 | 0.738554562535976 | 0.5749804471744757 | 0.10414584483084255 | 0.17704129591108972 | 50 |
| Ipsundrum+affect_modulation_only | dull | 0.8777777777777779 | 0.8777777777777779 | 0.20969110323263135 | 0.7322366236170523 | 0.5712968227304829 | 0.10749010301307702 | 0.1837057660416722 | 49 |
| Ipsundrum+affect_modulation_only | scenic | 0.8577777777777778 | 0.8715277777777779 | -0.37286719145366365 | 0.7442448477693498 | 0.567383702558316 | 0.10275051676694391 | 0.16907634089315862 | 49 |
| Ipsundrum+affect_readout_only | both | 0.96 | 0.96 | -0.01027128163940104 | 0.8193560339328239 | 0.6318419796143 | 0.08366602130590609 | 0.13352677041261685 | 47 |
| Ipsundrum+affect_readout_only | dull | 0.96 | 0.96 | 0.18365247600337747 | 0.8194591146563347 | 0.6318419796143 | 0.0833001953036156 | 0.13352677041261685 | 47 |
| Ipsundrum+affect_readout_only | scenic | 0.96 | 0.96 | -0.4067855566207882 | 0.8191603624503742 | 0.6256535545968103 | 0.08390375187128898 | 0.1348474820147792 | 47 |
| Recon | both | nan | nan | nan | nan | nan | nan | nan | 0 |
| Recon | dull | nan | nan | nan | nan | nan | nan | nan | 0 |
| Recon | scenic | nan | nan | nan | nan | nan | nan | nan | 0 |

### Exploratory Play

| model | unique_viewpoints | scan_events | turn_in_place_rate | revisit_rate | cycle_score | occupancy_entropy | dwell_p90 | mean_valence | mean_arousal | mean_Ns | mean_precision_eff | mean_alpha_eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ipsundrum_Affect | 136.3 | 33.9 | 0.363 | 0.3185 | 7.5 | 5.477821433923735 | 2.9 | 0.5295343885964974 | 0.1766523461329888 | 0.6813024067361565 | 1.088007637232138 | 0.8163010985340808 |
| Ipsundrum_Affect_readout_only | 131.9 | 34.2 | 0.331 | 0.3405 | 9.9 | 5.3498623059892525 | 2.9 | 0.5592290703618478 | 0.1651420775526813 | 0.6368857066346556 | 1.0 | 0.737133828866843 |
| Ipsundrum_Affect_modulation_only | 155.3 | 34.8 | 0.3995 | 0.2234999999999999 | 0.2 | 5.835892831308745 | 2.9 | 0.4729564332931382 | 0.2064286268220878 | 0.7159282820556634 | 1.1028018295971354 | 0.8199307077597696 |

### Lesion

| model | post_lesion_auc_sham | post_lesion_auc_lesion | post_lesion_halflife_sham | post_lesion_halflife_lesion | immediate_slope_sham | immediate_slope_lesion | ns_at_lesion_sham | ns_at_lesion_lesion | n_sham | n_lesion | auc_drop_sham_minus_lesion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ipsundrum+affect | 99.10963315458073 | 71.47702565726796 | 150.0 | 150.0 | -0.00488670367082728 | -0.3838357615718214 | 0.8577903608764623 | 0.8577903608764623 | 10 | 10 | 27.632607497312776 |
| Ipsundrum+affect_modulation_only | 99.10963315458073 | 71.47702565726796 | 150.0 | 150.0 | -0.00488670367082728 | -0.3838357615718214 | 0.8577903608764623 | 0.8577903608764623 | 10 | 10 | 27.632607497312776 |
| Ipsundrum+affect_readout_only | 90.98775079057285 | 71.33907801134009 | 150.0 | 150.0 | 0.021571971129014232 | -0.245888115643958 | 0.719842714948599 | 0.719842714948599 | 10 | 10 | 19.648672779232754 |

### Pain-Tail

| model | mean_tail_duration | std_tail_duration | mean_ns_auc_above_baseline | std_ns_auc_above_baseline | mean_arousal_integral | mean_turn_rate_tail | mean_forward_rate_tail | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ipsundrum+affect | 128.2 | 92.78745125883732 | 0.1580376260766383 | 0.0581731344654106 | 28.41182600402115 | 0.89 | 0.11 | 10 |
| Ipsundrum+affect_modulation_only | 120.6 | 88.4058821572411 | 0.1580376260766383 | 0.0581731344654106 | 28.41182600402115 | 0.94 | 0.06 | 10 |
| Ipsundrum+affect_readout_only | 125.8 | 95.94188055507584 | 0.2338961607830413 | 0.0109349115160629 | 25.857437709618047 | 0.78 | 0.22 | 10 |

## Fast Interpretation Notes

- Familiarity is the cleanest route-choice assay for readout sensitivity because it directly tracks scenic commitment after structured exposure.
- Lesion AUC drop isolates persistence in the recurrent machinery; if the lesion effect remains strong in modulation-only, persistence is not purely a policy-readout effect.
- Exploratory-play metrics are mixed by design: scanning, revisit structure, and dwell depend on several policy paths rather than one scalar affect score.
- Pain-tail should be treated cautiously here because the current summary mixes large duration variance with near-zero Ns half-life under the baseline-corrected metric.

## Output Inventory

- `results/extensions/alife_mech_decomp/exploratory-play/exploratory_play_clarified_alife_extension_quick.csv`
- `results/extensions/alife_mech_decomp/exploratory-play/exploratory_play_clarified_trace_alife_extension_quick.csv`
- `results/extensions/alife_mech_decomp/exploratory-play/fig_exploratory_play_clarified_alife_extension_quick.png`
- `results/extensions/alife_mech_decomp/exploratory-play/fig_exploratory_play_clarified_heatmaps_alife_extension_quick.png`
- `results/extensions/alife_mech_decomp/exploratory-play/fig_exploratory_play_trajectories_alife_extension_quick.png`
- `results/extensions/alife_mech_decomp/exploratory-play/final_viewpoints.csv`
- `results/extensions/alife_mech_decomp/exploratory-play/summary_alife_extension_quick.csv`
- `results/extensions/alife_mech_decomp/familiarity/episodes_improved.csv`
- `results/extensions/alife_mech_decomp/familiarity/side_bias.csv`
- `results/extensions/alife_mech_decomp/familiarity/summary.csv`
- `results/extensions/alife_mech_decomp/lesion/episodes_extended.csv`
- `results/extensions/alife_mech_decomp/lesion/ns_traces.npz`
- `results/extensions/alife_mech_decomp/pain-tail/episodes.csv`
- `results/extensions/alife_mech_decomp/pain-tail/summary.csv`
- `results/extensions/alife_mech_decomp/summary/exploratory_play_mechanism_summary.csv`
- `results/extensions/alife_mech_decomp/summary/familiarity_mechanism_summary.csv`
- `results/extensions/alife_mech_decomp/summary/lesion_mechanism_summary.csv`
- `results/extensions/alife_mech_decomp/summary/pain_tail_mechanism_summary.csv`
