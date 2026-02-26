"""Matplotlib styling helpers (paper-friendly defaults)."""

from __future__ import annotations

from typing import Sequence


_APPLIED = False


def apply_times_style(*, serif_fallbacks: Sequence[str] | None = None) -> None:
    """Apply a Times-like serif style for matplotlib figures.

    Notes:
    - We prefer Times if installed; otherwise matplotlib will fall back to the
      first available serif in the list.
    - We avoid requiring LaTeX (text.usetex) so plots render in minimal envs.
    """
    global _APPLIED
    if _APPLIED:
        return

    import matplotlib as mpl

    fallbacks = list(serif_fallbacks or [])
    # Provide robust defaults across common Linux/macOS setups.
    fallbacks.extend(
        [
            "Times New Roman",
            "Times",
            "Nimbus Roman No9 L",
            "TeX Gyre Termes",
            "STIXGeneral",
            "DejaVu Serif",
        ]
    )

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": fallbacks,
            # Times-like math without requiring TeX.
            "mathtext.fontset": "stix",
            # Better PDF/SVG text embedding when exporting vector formats.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    _APPLIED = True

