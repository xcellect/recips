from experiments.social_foodshare import run_foodshare_experiment


def test_coupling_off_reduces_helping_relative_to_sham(tmp_path):
    _, sham = run_foodshare_experiment(
        conditions=("social_affective_direct",),
        lambda_affective=1.0,
        profile="quick",
        outdir=str(tmp_path / "sham"),
        lesion_mode="sham",
    )
    _, lesion = run_foodshare_experiment(
        conditions=("social_affective_direct",),
        lambda_affective=1.0,
        profile="quick",
        outdir=str(tmp_path / "lesion"),
        lesion_mode="coupling_off",
    )
    assert lesion["help_rate_when_partner_distressed"].mean() < sham["help_rate_when_partner_distressed"].mean()
