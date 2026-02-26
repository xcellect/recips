import experiments.pain_tail_assay as pt


def test_baseline_corrected_half_life_returns_zero_when_trace_at_baseline() -> None:
    # Regression test for the old "< 0.5*peak" artifact when Ns returns exactly
    # to baseline (common for non-affect variants with Ns ~= 0.5 at rest).
    ns_trace = [0.5] * 50
    assert pt.baseline_corrected_half_life(
        ns_trace, ns_baseline=0.5, ns_peak=1.0, max_steps=50
    ) == 0


def test_baseline_corrected_half_life_crossing_index() -> None:
    # baseline=0.5, peak=1.0 => target=0.75; first sample <= 0.75 is index 3.
    ns_trace = [1.0, 0.9, 0.8, 0.74, 0.73]
    assert pt.baseline_corrected_half_life(
        ns_trace, ns_baseline=0.5, ns_peak=1.0, max_steps=len(ns_trace)
    ) == 3


def test_baseline_corrected_half_life_zero_when_peak_not_above_baseline() -> None:
    ns_trace = [0.2, 0.2, 0.2]
    assert pt.baseline_corrected_half_life(
        ns_trace, ns_baseline=0.5, ns_peak=0.4, max_steps=len(ns_trace)
    ) == 0

