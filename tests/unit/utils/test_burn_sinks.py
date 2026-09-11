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

from qbittensor.utils.burn_sinks import (
    BURN_SINK_HOTKEYS,
    DEFAULT_BURN_SINK_HOTKEY,
    burn_sink_jwt_needs_refresh,
    resolve_burn_sink_hotkey,
)


class TestResolveBurnSinkHotkey:
    def test_missing_defaults_to_first(self):
        assert resolve_burn_sink_hotkey(None) == DEFAULT_BURN_SINK_HOTKEY

    def test_unknown_defaults_to_first(self):
        assert resolve_burn_sink_hotkey("5NotInTheList") == DEFAULT_BURN_SINK_HOTKEY

    def test_unknown_uses_explicit_fallback(self):
        assert resolve_burn_sink_hotkey("5NotInTheList", fallback="local-burn") == "local-burn"

    def test_accepts_listed_burn_sink(self):
        listed = BURN_SINK_HOTKEYS[0]
        assert resolve_burn_sink_hotkey(listed) == listed


class TestBurnSinkJwtNeedsRefresh:
    def test_missing_needs_refresh(self):
        assert burn_sink_jwt_needs_refresh(
            burn_sink_hotkey=None, jwt_tempo_id=1, current_tempo_id=1
        )

    def test_unknown_needs_refresh(self):
        assert burn_sink_jwt_needs_refresh(
            burn_sink_hotkey="5Nope", jwt_tempo_id=1, current_tempo_id=1
        )

    def test_stale_tempo_needs_refresh(self):
        assert burn_sink_jwt_needs_refresh(
            burn_sink_hotkey=BURN_SINK_HOTKEYS[0],
            jwt_tempo_id=10,
            current_tempo_id=11,
        )

    def test_current_valid_sink_does_not_refresh(self):
        assert not burn_sink_jwt_needs_refresh(
            burn_sink_hotkey=BURN_SINK_HOTKEYS[0],
            jwt_tempo_id=10,
            current_tempo_id=10,
        )

    def test_valid_sink_without_tempo_does_not_refresh(self):
        assert not burn_sink_jwt_needs_refresh(
            burn_sink_hotkey=BURN_SINK_HOTKEYS[0],
            jwt_tempo_id=None,
            current_tempo_id=10,
        )


class TestEnvOverride:
    def test_env_list_replaces_builtin(self, monkeypatch):
        from qbittensor.utils import burn_sinks as burns

        monkeypatch.setenv("BURN_SINK_HOTKEYS", "5FirstBurnSink,5SecondBurnSink")
        assert burns.burn_sink_hotkeys() == ("5FirstBurnSink", "5SecondBurnSink")
        assert burns.default_burn_sink_hotkey() == "5FirstBurnSink"
        assert burns.resolve_burn_sink_hotkey("5SecondBurnSink") == "5SecondBurnSink"
        assert burns.resolve_burn_sink_hotkey("5NotInEnvList") == "5FirstBurnSink"
