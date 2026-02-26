from core.ipsundrum_model import Builder, AffectParams, LoopParams


def test_affect_nodes_present_only_when_enabled():
    b_no_aff = Builder(params=LoopParams(), affect=AffectParams(enabled=False))
    net_no_aff, _ = b_no_aff.stage_D()
    assert "Ni" not in net_no_aff.nodes
    assert "Nv" not in net_no_aff.nodes
    assert "Na" not in net_no_aff.nodes

    b_aff = Builder(params=LoopParams(), affect=AffectParams(enabled=True))
    net_aff, _ = b_aff.stage_D()
    assert "Ni" in net_aff.nodes
    assert "Nv" in net_aff.nodes
    assert "Na" in net_aff.nodes
