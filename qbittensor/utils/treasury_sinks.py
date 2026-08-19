# The MIT License (MIT)
# Copyright © 2026 qBitTensor Labs
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Treasury emission sinks used to rotate validator weight targets each tempo.

These are hotkeys associated with the EVM treasury contract.
"""

from __future__ import annotations

import os
from typing import Optional

TREASURY_SINK_HOTKEYS: tuple[str, ...] = (
    "5DCLafsAKaLeZwm9hjMHvrQNjtucSwBhKyTLYnYmMvhxF2Uc",
    "5CMQvyveSZGbm61TigSjt5fsepwBTHiyykCzPQwH2BivE58u",
    "5Fq2wrGHppr81GGtEdSknBQi7uivug54xx2KgReX5aCyRRho",
    "5CDfAovpyrQoLqobEi3KLWXCUHauu5NMtyfvFfNU7FxArKkt",
    "5Hg2vYcw4vHxceXRqnA9nqoWwPuQDs2zoBGM7c1j1nanpG2H",
)
TREASURY_SINK_SET: frozenset[str] = frozenset(TREASURY_SINK_HOTKEYS)
DEFAULT_TREASURY_SINK_HOTKEY: str = TREASURY_SINK_HOTKEYS[0]

TREASURY_SINK_HOTKEYS_ENV: str = "TREASURY_SINK_HOTKEYS"


def sink_hotkeys() -> tuple[str, ...]:
    raw = os.environ.get(TREASURY_SINK_HOTKEYS_ENV, "") or ""
    keys = tuple(part.strip() for part in raw.split(",") if part.strip())
    return keys if keys else TREASURY_SINK_HOTKEYS


def default_sink_hotkey() -> str:
    return sink_hotkeys()[0]


def sink_set() -> frozenset[str]:
    return frozenset(sink_hotkeys())


def resolve_sink_hotkey(
    candidate: Optional[str],
    *,
    fallback: Optional[str] = None,
) -> str:
    """Accept a platform sink only if it is in the configured list.

    Unknown/missing candidates use ``fallback`` when provided, otherwise the
    first hotkey in the configured sink list.
    """
    keys = sink_hotkeys()
    if isinstance(candidate, str) and candidate in keys:
        return candidate
    if isinstance(fallback, str) and fallback:
        return fallback
    return keys[0]


def sink_jwt_needs_refresh(
    *,
    sink_hotkey: Optional[str],
    jwt_tempo_id: Optional[int],
    current_tempo_id: Optional[int],
) -> bool:
    """True when the cached JWT has no usable sink or is from a prior tempo."""
    if not (isinstance(sink_hotkey, str) and sink_hotkey in sink_set()):
        return True
    if isinstance(jwt_tempo_id, int) and isinstance(current_tempo_id, int):
        return jwt_tempo_id != current_tempo_id
    return False
