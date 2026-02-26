import pytest

from core.ipsundrum_model import Builder
from core.recon_core import ReCoNConfig


def test_recon_strict_env_enforces_strict_config(monkeypatch) -> None:
    monkeypatch.setenv("RECON_STRICT", "1")
    with pytest.raises(ValueError):
        Builder(recon_config=ReCoNConfig())


def test_recon_mode_flag_set_from_env(monkeypatch) -> None:
    monkeypatch.setenv("RECON_STRICT", "1")
    b = Builder()
    net, _ = b.stage_B()
    assert net.recon_mode == "strict"
    assert net.get("Root").config.strict_fsm
