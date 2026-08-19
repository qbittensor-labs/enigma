# The MIT License (MIT)
# Copyright © 2026 qBitTensor Labs

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Simple allowlist for deciding which validators the miner should offer submissions to.

The on-chain TreasuryController stores the "trusted validators" as EVM addresses
in a private set with no public getter. Therefore we maintain a hardcoded list
of the corresponding SS58 hotkeys here.

Usage:
- Update DEFAULT_TRUSTED_HOTKEYS when the admin whitelists or removes validators.
- Operators can override at runtime with --treasury.trusted_hotkeys.
- Completely disable the gate with --treasury.disable_whitelist_check.

Non-trusted (or disabled) validators can still report submission statuses,
but will not be given new solution candidates.
"""

from typing import Optional, Set

import bittensor as bt


# ---------------------------------------------------------------------------
# Hardcoded trusted validator hotkeys (SS58 addresses).
#
# This is the list the miner will offer submissions to by default.
# Keep it in sync with on-chain whitelist changes.
#
# Current list (update as needed):
# ---------------------------------------------------------------------------
DEFAULT_TRUSTED_HOTKEYS: list[str] = [
    "5GzjAcUcD3pFk5ybJ1qP4tMfnyk2Kh3SX8R2kMQwPU2dTs63",
    "5EZ52JMq4S7PYqzmLAggYahyDirMx3p1f1uBtLQgx6fk7kR8",
    "5D7etdUeLm4JsSZGkBrHoiyjezPHAJUiafEd5XU4wNnWSL63",
]


class TrustedValidatorChecker:
    """
    Lightweight checker used by the miner.

    Public API kept small and stable:
      - is_trusted(hotkey: str) -> bool
      - refresh()                 (no-op, for compatibility)
      - trusted_hotkeys           (property)
    """

    def __init__(
        self,
        trusted_hotkeys: Optional[str] = None,  # comma-separated SS58 override
        disable: bool = False,
    ):
        self._disable = bool(disable)
        self._trusted_hotkeys: Set[str] = set()

        if self._disable:
            bt.logging.warning("Trusted validator whitelist gating is DISABLED")
            return

        # Start from the built-in list
        allow = set(DEFAULT_TRUSTED_HOTKEYS)

        # CLI override completely replaces the list when provided
        if trusted_hotkeys:
            provided = {h.strip() for h in trusted_hotkeys.split(",") if h.strip()}
            if provided:
                allow = provided
                bt.logging.info(f"Overriding trusted hotkeys from CLI ({len(allow)} entries)")

        self._trusted_hotkeys = allow

        if self._trusted_hotkeys:
            bt.logging.info(
                f"🔐 Miner will only offer submissions to {len(self._trusted_hotkeys)} whitelisted validator(s)"
            )
        else:
            bt.logging.info("🔐 No trusted hotkeys configured — will offer to all validators")

    def refresh(self, force: bool = False) -> None:
        """Compatibility no-op."""
        pass

    def is_trusted(self, hotkey: str) -> bool:
        """
        Should we offer a submission to this validator hotkey?

        Rules:
        - If disabled -> always True
        - If allowlist is empty -> True (conservative, avoid locking everyone out)
        - Otherwise -> hotkey must be in the allowlist
        """
        if self._disable:
            return True
        if not self._trusted_hotkeys:
            return True
        return hotkey in self._trusted_hotkeys

    @property
    def trusted_hotkeys(self) -> Set[str]:
        return set(self._trusted_hotkeys)
