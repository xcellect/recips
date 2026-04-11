import numpy as np

from experiments.context_fork_assay import _branch_margin, _fork_vector_distance, run_assay


def test_fork_vector_distance_uses_full_latent_vectors():
    phi_a = np.array([3.0, 4.0])
    phi_b = np.array([0.0, 0.0])
    assert np.isclose(_fork_vector_distance(phi_a, phi_b), 5.0)


def test_branch_margin_prefers_branch_closer_to_context_anchor():
    cue_phi = np.array([1.0, 0.0])
    correct_phi = np.array([0.8, 0.1])
    incorrect_phi = np.array([-0.6, 0.0])
    assert _branch_margin(cue_phi, correct_phi, incorrect_phi) > 0.0


def test_context_fork_assay_produces_nonzero_representation_distance():
    result, trace = run_assay("perspective_plastic", arch_seed=0, env_seed=0, delay=6)
    assert result["R"] > 0.0
    assert len(trace) > 0
