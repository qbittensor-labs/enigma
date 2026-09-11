# The MIT License (MIT)
# Copyright © 2026 qBitTensor Labs
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Owner-wallet burn sinks used to rotate validator burn weights each tempo.

Keep this list in sync with tensorauth.burn_hotkeys on the platform. Tensorauth
HMAC-picks one of these per tempo and puts it on the trusted-validator JWT as
``burn_hotkey``. The validator only allowlists that claim — it does not pick
the key itself and does not look up SubnetOwnerHotkey on-chain.
"""

from __future__ import annotations

import os
from typing import Optional

BURN_SINK_HOTKEYS: tuple[str, ...] = (
    "5GmpedVP2r9haksUBD743wek8t89jMz96RTkV2zkVdRR4e1B",
)
DEFAULT_BURN_SINK_HOTKEY: str = BURN_SINK_HOTKEYS[0]

BURN_SINK_HOTKEYS_ENV: str = "BURN_SINK_HOTKEYS"


def burn_sink_hotkeys() -> tuple[str, ...]:
    raw = os.environ.get(BURN_SINK_HOTKEYS_ENV, "") or ""
    keys = tuple(part.strip() for part in raw.split(",") if part.strip())
    return keys if keys else BURN_SINK_HOTKEYS


def default_burn_sink_hotkey() -> str:
    return burn_sink_hotkeys()[0]


def burn_sink_set() -> frozenset[str]:
    return frozenset(burn_sink_hotkeys())


def resolve_burn_sink_hotkey(
    candidate: Optional[str],
    *,
    fallback: Optional[str] = None,
) -> str:
    """Accept a platform burn sink only if it is in the configured list.

    Unknown/missing candidates use ``fallback`` when provided, otherwise the
    first hotkey in the configured burn sink list.
    """
    keys = burn_sink_hotkeys()
    if isinstance(candidate, str) and candidate in keys:
        return candidate
    if isinstance(fallback, str) and fallback:
        return fallback
    return keys[0]


def burn_sink_jwt_needs_refresh(
    *,
    burn_sink_hotkey: Optional[str],
    jwt_tempo_id: Optional[int],
    current_tempo_id: Optional[int],
) -> bool:
    """True when the cached JWT has no usable burn sink or is from a prior tempo."""
    if not (isinstance(burn_sink_hotkey, str) and burn_sink_hotkey in burn_sink_set()):
        return True
    if isinstance(jwt_tempo_id, int) and isinstance(current_tempo_id, int):
        return jwt_tempo_id != current_tempo_id
    return False
