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

"""Per-challenge prize-pool cap — decide whether this tempo should fund or burn.

The prize pool lives in a single treasury wallet. The cap is
``alpha_per_challenge × active_challenges``. This module only answers
"is the pool at/above that ceiling?". Recipient hotkeys (JWT sink vs JWT burn)
are chosen by the validator, not here.

Active-challenge count is derived from the platform: list challenges, fetch
each, count those with at least one live milestone. Any failure returns
``None`` and the caller funds (fail-safe). A genuine ``0`` zeroes the cap.

Configuration:

  TREASURY_CAP_ENABLED              "1"/"true" to enable (default: off)
  TREASURY_CAP_ALPHA_PER_CHALLENGE  cap per active challenge, in alpha (default: 200000)

Treasury alpha is read for the vault coldkey that owns the current treasury
sink hotkey on the metagraph (not a separate env var).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

ENABLED_ENV: str = "TREASURY_CAP_ENABLED"
ALPHA_PER_CHALLENGE_ENV: str = "TREASURY_CAP_ALPHA_PER_CHALLENGE"

DEFAULT_ALPHA_PER_CHALLENGE: float = 200_000.0

MILESTONE_STATUS_COMPLETE: str = "Complete"


@dataclass(frozen=True)
class TreasuryCapConfig:
    enabled: bool
    alpha_per_challenge: float

    @property
    def is_operational(self) -> bool:
        return bool(self.enabled and self.alpha_per_challenge > 0)


@dataclass(frozen=True)
class CapDecision:
    """Whether this tempo's treasury share should burn, plus diagnostics."""

    burn: bool
    reason: str
    active_challenges: Optional[int] = None
    treasury_alpha: Optional[float] = None
    effective_cap: Optional[float] = None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def load_config() -> TreasuryCapConfig:
    try:
        alpha_per_challenge = float(
            os.environ.get(ALPHA_PER_CHALLENGE_ENV, "") or DEFAULT_ALPHA_PER_CHALLENGE
        )
    except (TypeError, ValueError):
        alpha_per_challenge = DEFAULT_ALPHA_PER_CHALLENGE

    return TreasuryCapConfig(
        enabled=_env_bool(ENABLED_ENV, default=False),
        alpha_per_challenge=alpha_per_challenge,
    )


def _parse_iso(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_milestone_live(milestone: dict, now: datetime) -> bool:
    """Live if started, not ended, and not complete.

    Null/absent ``start_date`` means open from the past (live). Unparseable
    start also leans live so a garbled date cannot undercount and burn.
    """
    status = str(milestone.get("status") or "").strip().lower()
    if status == MILESTONE_STATUS_COMPLETE.lower():
        return False
    raw_start = milestone.get("start_date")
    if raw_start is not None:
        start = _parse_iso(raw_start)
        if start is not None and start > now:
            return False
    end = _parse_iso(milestone.get("end_date"))
    if end is not None and end <= now:
        return False
    return True


def count_active_challenges(client, now: datetime) -> Optional[int]:
    """Count challenges with at least one live milestone. ``None`` on any doubt."""
    try:
        listing = client.list_challenges()
    except Exception:
        return None

    if isinstance(listing, dict):
        if "challenges" not in listing:
            return None
        challenges = listing.get("challenges")
    else:
        challenges = listing
    if not isinstance(challenges, list):
        return None

    count = 0
    for entry in challenges:
        if not isinstance(entry, dict) or not entry.get("id"):
            return None
        try:
            detail = client.get_challenge(entry["id"])
        except Exception:
            return None
        if not isinstance(detail, dict):
            return None
        milestones = detail.get("milestones")
        if not isinstance(milestones, list):
            return None
        if any(is_milestone_live(m, now) for m in milestones if isinstance(m, dict)):
            count += 1
    return count


def effective_cap_alpha(cfg: TreasuryCapConfig, active_challenges: int) -> float:
    return cfg.alpha_per_challenge * max(0, active_challenges)


def read_treasury_alpha(subtensor, coldkey: str, netuid: int) -> Optional[float]:
    """Treasury coldkey's total alpha on ``netuid``. ``None`` on any failure."""
    try:
        positions = subtensor.staking.positions(coldkey_ss58=coldkey)
    except Exception:
        return None

    if not positions:
        return 0.0

    total = 0.0
    try:
        for info in positions:
            try:
                if int(getattr(info, "netuid", -1)) != int(netuid):
                    continue
            except (TypeError, ValueError):
                continue
            stake = getattr(info, "stake", 0.0)
            if hasattr(stake, "amount"):
                total += float(stake.amount)
            elif hasattr(stake, "tao"):
                total += float(stake.tao)
            else:
                try:
                    total += float(stake)
                except (TypeError, ValueError):
                    continue
    except Exception:
        return None
    return total


def should_burn_treasury_share(
    treasury_alpha: Optional[float],
    cfg: TreasuryCapConfig,
    active_challenges: Optional[int],
) -> bool:
    """True only when the cap is operational and alpha is known to be at/above it."""
    if not cfg.is_operational:
        return False
    if active_challenges is None:
        return False
    if treasury_alpha is None:
        return False
    return treasury_alpha >= effective_cap_alpha(cfg, active_challenges)


def decide_treasury_share(
    cfg: TreasuryCapConfig,
    *,
    client,
    subtensor,
    netuid: int,
    now: datetime,
    treasury_coldkey: Optional[str],
) -> CapDecision:
    """Single entry for the validator: burn or fund, with a reason."""
    if not cfg.is_operational:
        return CapDecision(burn=False, reason="disabled")

    active = count_active_challenges(client, now)
    if active is None:
        return CapDecision(burn=False, reason="unknown_count")

    cap = effective_cap_alpha(cfg, active)
    if not treasury_coldkey:
        return CapDecision(
            burn=False,
            reason="unreadable_alpha",
            active_challenges=active,
            effective_cap=cap,
        )
    alpha = read_treasury_alpha(subtensor, treasury_coldkey, netuid)
    if alpha is None:
        return CapDecision(
            burn=False,
            reason="unreadable_alpha",
            active_challenges=active,
            effective_cap=cap,
        )
    if should_burn_treasury_share(alpha, cfg, active):
        return CapDecision(
            burn=True,
            reason="at_cap",
            active_challenges=active,
            treasury_alpha=alpha,
            effective_cap=cap,
        )
    return CapDecision(
        burn=False,
        reason="below_cap",
        active_challenges=active,
        treasury_alpha=alpha,
        effective_cap=cap,
    )
