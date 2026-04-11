from analysis.paper_claims import Claim, _claim_passes


def test_directional_comparison_claim_uses_meta_pass_flag():
    claim = Claim(
        claim_id="demo",
        value=-0.1,
        ci=(-0.2, -0.05),
        n=16,
        claim_type="mean_ci",
        digits=3,
        meta={"pass": False},
    )
    assert _claim_passes(claim) is False
