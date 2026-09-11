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

import types
from datetime import datetime, timezone

import pytest

from qbittensor.utils.treasury_cap import (
    DEFAULT_ALPHA_PER_CHALLENGE,
    TreasuryCapConfig,
    count_active_challenges,
    decide_treasury_share,
    effective_cap_alpha,
    is_milestone_live,
    load_config,
    read_treasury_alpha,
    should_burn_treasury_share,
)

NOW = datetime(2026, 9, 7, tzinfo=timezone.utc)
PAST = "2026-06-01T00:00:00Z"
PAST2 = "2026-05-01T00:00:00Z"
FUTURE = "2027-01-01T00:00:00Z"

CAP_ENVS = [
    "TREASURY_CAP_ENABLED",
    "TREASURY_CAP_ALPHA_PER_CHALLENGE",
]


@pytest.fixture(autouse=True)
def _clear_cap_env(monkeypatch):
    for name in CAP_ENVS:
        monkeypatch.delenv(name, raising=False)


def _cfg(**kw):
    base = dict(
        enabled=True,
        alpha_per_challenge=250_000.0,
    )
    base.update(kw)
    return TreasuryCapConfig(**base)


def _ms(status="Incomplete", start=PAST, end=None):
    return {"status": status, "start_date": start, "end_date": end}


class _Client:
    def __init__(self, details, list_raises=False):
        self._details = details
        self._list_raises = list_raises

    def list_challenges(self):
        if self._list_raises:
            raise RuntimeError("list down")
        return {"challenges": [{"id": cid, "name": cid} for cid in self._details]}

    def get_challenge(self, cid):
        val = self._details[cid]
        if val == "raise":
            raise RuntimeError("detail down")
        if val == "malformed":
            return {"id": cid}
        return {"id": cid, "milestones": val}


class _Stake:
    def __init__(self, amount):
        self.amount = amount


class _Position:
    def __init__(self, netuid, amount):
        self.netuid = netuid
        self.stake = _Stake(amount)


class _Staking:
    def __init__(self, positions=None, raises=False):
        self._positions = positions or []
        self._raises = raises

    def positions(self, coldkey_ss58=None):
        if self._raises:
            raise RuntimeError("rpc down")
        return self._positions


def _subtensor(positions=None, raises=False):
    return types.SimpleNamespace(staking=_Staking(positions, raises))


class TestConfig:
    def test_defaults(self):
        cfg = load_config()
        assert cfg.enabled is False
        assert cfg.alpha_per_challenge == DEFAULT_ALPHA_PER_CHALLENGE == 200_000.0
        assert cfg.is_operational is False

    def test_no_active_challenge_count_field(self):
        assert not hasattr(load_config(), "active_challenges")

    def test_operational_when_enabled(self, monkeypatch):
        monkeypatch.setenv("TREASURY_CAP_ENABLED", "true")
        assert load_config().is_operational is True


class TestIsMilestoneLive:
    def test_live_when_started_not_ended_not_complete(self):
        assert is_milestone_live(_ms(start=PAST, end=None), NOW) is True
        assert is_milestone_live(_ms(start=PAST, end=FUTURE), NOW) is True

    def test_null_start_date_means_open_and_live(self):
        assert is_milestone_live(_ms(start=None), NOW) is True
        assert is_milestone_live(_ms(start="not-a-date"), NOW) is True

    def test_status_compared_case_insensitively(self):
        assert is_milestone_live(_ms(status="Complete", start=PAST), NOW) is False
        assert is_milestone_live(_ms(status="complete", start=PAST), NOW) is False
        assert is_milestone_live(_ms(status="COMPLETE", start=PAST), NOW) is False

    def test_not_live_when_unstarted_or_ended(self):
        assert is_milestone_live(_ms(start=FUTURE), NOW) is False
        assert is_milestone_live(_ms(start=PAST2, end=PAST), NOW) is False
        assert is_milestone_live(_ms(start=None, end=PAST), NOW) is False


class TestCountActiveChallenges:
    def test_counts_only_challenges_with_a_live_milestone(self):
        client = _Client({
            "rsa": [_ms(start=PAST), _ms(status="Complete", start=PAST2)],
            "hqp": [_ms(start=PAST)],
            "treasury": [_ms(start=PAST2, end=PAST)],
            "mock": [_ms(start=None)],
            "btc": [_ms(start=FUTURE)],
        })
        assert count_active_challenges(client, NOW) == 3

    def test_genuine_zero(self):
        assert count_active_challenges(_Client({"btc": [_ms(start=FUTURE)]}), NOW) == 0

    def test_empty_milestone_list_is_honored_not_failed(self):
        client = _Client({"draft": [], "hqp": [_ms(start=PAST)]})
        assert count_active_challenges(client, NOW) == 1

    def test_none_on_list_or_detail_failure(self):
        assert count_active_challenges(_Client({}, list_raises=True), NOW) is None
        assert count_active_challenges(_Client({"rsa": "raise"}), NOW) is None

    def test_none_on_malformed_detail(self):
        assert count_active_challenges(_Client({"rsa": "malformed"}), NOW) is None

    def test_none_on_missing_challenges_key(self):
        class _BadEnvelope:
            def list_challenges(self):
                return {"data": [{"id": "rsa"}]}

            def get_challenge(self, cid):  # pragma: no cover - never reached
                raise AssertionError

        assert count_active_challenges(_BadEnvelope(), NOW) is None

    def test_none_on_malformed_listing_entry(self):
        class _BadEntries:
            def list_challenges(self):
                return {"challenges": [{"challenge_id": "rsa"}]}

            def get_challenge(self, cid):  # pragma: no cover - never reached
                raise AssertionError

        assert count_active_challenges(_BadEntries(), NOW) is None


class TestEffectiveCapAndBurn:
    def test_effective_cap_scales(self):
        assert effective_cap_alpha(_cfg(), 2) == 500_000.0
        assert effective_cap_alpha(_cfg(), 3) == 750_000.0
        assert effective_cap_alpha(_cfg(), 0) == 0.0

    def test_burns_at_or_above_cap(self):
        assert should_burn_treasury_share(500_000.0, _cfg(), 2) is True
        assert should_burn_treasury_share(499_999.0, _cfg(), 2) is False

    def test_zero_active_burns_everything(self):
        assert should_burn_treasury_share(0.0, _cfg(), 0) is True
        assert should_burn_treasury_share(123.0, _cfg(), 0) is True

    def test_unknown_count_never_burns(self):
        assert should_burn_treasury_share(9_000_000.0, _cfg(), None) is False

    def test_unreadable_balance_never_burns(self):
        assert should_burn_treasury_share(None, _cfg(), 2) is False

    def test_not_operational_never_burns(self):
        assert should_burn_treasury_share(9_000_000.0, _cfg(enabled=False), 2) is False


class TestReadTreasuryAlpha:
    def test_sums_all_hotkeys_on_netuid(self):
        sub = _subtensor([_Position(63, 400_000), _Position(1, 999), _Position(63, 250_000)])
        assert read_treasury_alpha(sub, "5Vault", 63) == pytest.approx(650_000.0)

    def test_rpc_failure_returns_none(self):
        assert read_treasury_alpha(_subtensor(raises=True), "5Vault", 63) is None


class TestDecideTreasuryShare:
    def test_disabled_does_not_touch_client(self):
        class _Boom:
            def list_challenges(self):
                raise AssertionError

        decision = decide_treasury_share(
            _cfg(enabled=False),
            client=_Boom(),
            subtensor=_subtensor(),
            netuid=63,
            now=NOW,
            treasury_coldkey="5Vault",
        )
        assert decision.burn is False
        assert decision.reason == "disabled"

    def test_at_cap(self):
        client = _Client({"rsa": [_ms(start=PAST)]})
        sub = _subtensor([_Position(1, 400_000)])
        decision = decide_treasury_share(
            _cfg(alpha_per_challenge=200_000.0),
            client=client,
            subtensor=sub,
            netuid=1,
            now=NOW,
            treasury_coldkey="5Vault",
        )
        assert decision.burn is True
        assert decision.reason == "at_cap"
        assert decision.active_challenges == 1

    def test_missing_coldkey_funds(self):
        client = _Client({"rsa": [_ms(start=PAST)]})
        decision = decide_treasury_share(
            _cfg(),
            client=client,
            subtensor=_subtensor([_Position(1, 400_000)]),
            netuid=1,
            now=NOW,
            treasury_coldkey=None,
        )
        assert decision.burn is False
        assert decision.reason == "unreadable_alpha"
