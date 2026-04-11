import numpy as np

from core.social_homeostat import HomeostatParams, SocialCouplingParams, SocialObservation, initial_homeostat, step_homeostat


def test_lambda_zero_reproduces_self_only_homeostasis():
    params = HomeostatParams()
    state = initial_homeostat(0.5, params)
    out = step_homeostat(
        state,
        params,
        SocialCouplingParams(lambda_affective=0.0, observe_partner_internal=True),
        partner_observation=SocialObservation(other_energy_est=0.1),
    ).state
    assert np.isclose(out.distress_coupled, out.distress_self)


def test_partner_distress_monotonic_increases_coupled_distress():
    params = HomeostatParams()
    state = initial_homeostat(0.5, params)
    social = SocialCouplingParams(lambda_affective=0.75, observe_partner_internal=True)
    low = step_homeostat(state, params, social, partner_observation=SocialObservation(other_energy_est=0.65)).state
    high = step_homeostat(state, params, social, partner_observation=SocialObservation(other_energy_est=0.05)).state
    assert high.distress_coupled > low.distress_coupled
