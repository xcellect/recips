from core.ipsundrum_model import Builder, LoopParams
from core.evaluation import run_episode, phenomenal_duration, signature


def pulse3(t: int) -> float:
    return 1.0 if t < 3 else 0.0


def pulse6(t: int) -> float:
    return 1.0 if t < 6 else 0.0


def test_stage_A_reflex_is_brief():
    b = Builder()
    net, _ = b.stage_A()
    net.start_root(True)

    tr = run_episode(net, stimulus=pulse3, steps=30)
    # allow message latency; still should be brief
    assert phenomenal_duration(tr, threshold=0.5) <= 6


def test_stage_B_efference_spikes_and_P_confirms():
    b = Builder()
    net, _ = b.stage_B()
    net.start_root(True)

    tr = run_episode(net, stimulus=pulse6, steps=60)
    assert "CONFIRMED" in tr.P_state
    assert max(tr.Ne) > 0.5


def test_stage_C_duration_increases_with_gain():
    low = Builder(params=LoopParams(g=0.4, h=1.0, nonlinearity="linear"))
    netL, _ = low.stage_C(cycles_required=8)
    netL.start_root(True)
    trL = run_episode(netL, stimulus=pulse3, steps=200)
    durL = phenomenal_duration(trL)

    high = Builder(params=LoopParams(g=0.8, h=1.0, nonlinearity="linear"))
    netH, _ = high.stage_C(cycles_required=8)
    netH.start_root(True)
    trH = run_episode(netH, stimulus=pulse3, steps=200)
    durH = phenomenal_duration(trH)

    assert durH > durL


def test_stage_D_persists_and_recovers():
    b = Builder(params=LoopParams(g=1.2, h=1.0, nonlinearity="linear", fatigue=0.0))
    net, _ = b.stage_D(efference_threshold=0.05)
    net.start_root(True)

    tr = run_episode(net, stimulus=pulse3, steps=160)
    assert phenomenal_duration(tr) > 20

    # perturbation: wipe reafferent once, then verify continued activity
    net._ipsundrum_state["reafferent"] = 0.0  # type: ignore[attr-defined]
    tr2 = run_episode(net, stimulus=lambda t: 0.0, steps=80)
    assert phenomenal_duration(tr2) > 10


def test_modality_signature_differs():
    b1 = Builder(params=LoopParams(g=1.1, h=1.0, nonlinearity="linear", fatigue=0.01))
    n1, _ = b1.stage_D(efference_threshold=0.05)
    n1.start_root(True)
    tr1 = run_episode(n1, stimulus=pulse3, steps=256, seed=1)

    b2 = Builder(params=LoopParams(g=1.3, h=1.0, nonlinearity="linear", fatigue=0.05))
    n2, _ = b2.stage_D(efference_threshold=0.05)
    n2.start_root(True)
    tr2 = run_episode(n2, stimulus=pulse3, steps=256, seed=1)

    s1 = signature(tr1)
    s2 = signature(tr2)
    dist2 = float(((s1 - s2) ** 2).sum())
    assert dist2 > 1e-6
