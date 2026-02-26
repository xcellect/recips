"""Model naming utilities (internal ids vs. standardized display names).

This repo historically used mixed identifiers like:
- recon / ReCoN
- humphrey / ipsundrum
- humphrey_barrett / barrett / ipsundrum+affect

For results and paper outputs, we standardize display names to:
- Recon
- Ipsundrum
- Ipsundrum+affect
"""

from __future__ import annotations

import re
from typing import Iterable, List

# Internal ids used in code paths / builders
MODEL_ID_RECON = "recon"
MODEL_ID_IPSUNDRUM = "humphrey"
MODEL_ID_IPSUNDRUM_AFFECT = "humphrey_barrett"

MODEL_ID_ORDER: List[str] = [
    MODEL_ID_RECON,
    MODEL_ID_IPSUNDRUM,
    MODEL_ID_IPSUNDRUM_AFFECT,
]

# Standardized display names for all saved results (csv/plots/tables)
MODEL_NAME_RECON = "Recon"
MODEL_NAME_IPSUNDRUM = "Ipsundrum"
MODEL_NAME_IPSUNDRUM_AFFECT = "Ipsundrum+affect"

MODEL_DISPLAY_ORDER: List[str] = [
    MODEL_NAME_RECON,
    MODEL_NAME_IPSUNDRUM,
    MODEL_NAME_IPSUNDRUM_AFFECT,
]

_DISPLAY_BY_ID = {
    MODEL_ID_RECON: MODEL_NAME_RECON,
    MODEL_ID_IPSUNDRUM: MODEL_NAME_IPSUNDRUM,
    MODEL_ID_IPSUNDRUM_AFFECT: MODEL_NAME_IPSUNDRUM_AFFECT,
}


def _alias_key(name: str) -> str:
    key = str(name).strip().lower()
    key = key.replace(" ", "")
    key = key.replace("-", "_")
    key = key.replace("+", "_")
    key = key.replace("/", "_")
    key = re.sub(r"_+", "_", key)
    return key.strip("_")


def canonical_model_id(name: str) -> str:
    """Map a user-facing/legacy name to an internal model id.

    Unknown names are normalized (lowercase + '_' separators) and returned.
    """
    key = _alias_key(name)

    if key in ("recon", "recon_baseline", "reconstageb", "recon_stageb", "recon_stage_b", "reconb"):
        return MODEL_ID_RECON

    if key in ("humphrey", "ipsundrum"):
        return MODEL_ID_IPSUNDRUM

    if key in (
        "humphrey_barrett",
        "humphreybarrett",
        "hb",
        "full",
        "barrett",
        "ipsundrum_affect",
        "ipsundrumaffect",
        "ipsundrum_plus_affect",
    ):
        return MODEL_ID_IPSUNDRUM_AFFECT

    # Support ablation variants expressed with the standardized prefix.
    if key.startswith("ipsundrum_affect_"):
        suffix = key[len("ipsundrum_affect_") :]
        return f"{MODEL_ID_IPSUNDRUM_AFFECT}_{suffix}"

    # Support non-affect variants expressed with the standardized prefix.
    if key.startswith("ipsundrum_"):
        suffix = key[len("ipsundrum_") :]
        return f"{MODEL_ID_IPSUNDRUM}_{suffix}"

    if key.startswith("humphrey_barrett_"):
        return key

    return key


def canonical_model_display(name: str) -> str:
    """Map a model id or alias to the standardized display name.

    For ablation variants, the standardized base name is preserved and the suffix
    is appended (e.g. humphrey_barrett_readout_only -> Ipsundrum+affect_readout_only).
    """
    model_id = canonical_model_id(name)

    if model_id in _DISPLAY_BY_ID:
        return _DISPLAY_BY_ID[model_id]

    if model_id.startswith(f"{MODEL_ID_IPSUNDRUM_AFFECT}_"):
        return f"{MODEL_NAME_IPSUNDRUM_AFFECT}{model_id[len(MODEL_ID_IPSUNDRUM_AFFECT):]}"

    if model_id.startswith(f"{MODEL_ID_IPSUNDRUM}_"):
        return f"{MODEL_NAME_IPSUNDRUM}{model_id[len(MODEL_ID_IPSUNDRUM):]}"

    if model_id.startswith(f"{MODEL_ID_RECON}_"):
        return f"{MODEL_NAME_RECON}{model_id[len(MODEL_ID_RECON):]}"

    return str(name)


def canonicalize_model_list(models: Iterable[str]) -> List[str]:
    return [canonical_model_id(m) for m in models]
