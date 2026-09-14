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

from qbittensor.utils.treasury_sinks import (
    DEFAULT_TREASURY_SINK_HOTKEY,
    TREASURY_SINK_HOTKEYS,
    resolve_sink_hotkey,
    treasury_wallet_coldkey,
)


class TestResolveSinkHotkey:
    def test_missing_defaults_to_first(self):
        assert resolve_sink_hotkey(None) == DEFAULT_TREASURY_SINK_HOTKEY

    def test_unknown_defaults_to_first(self):
        assert resolve_sink_hotkey("5NotInTheList") == DEFAULT_TREASURY_SINK_HOTKEY

    def test_unknown_uses_explicit_fallback(self):
        assert resolve_sink_hotkey("5NotInTheList", fallback="local-vault") == "local-vault"

    def test_accepts_listed_sink(self):
        listed = TREASURY_SINK_HOTKEYS[3]
        assert resolve_sink_hotkey(listed) == listed


class TestTreasuryWalletColdkey:
    def test_reads_coldkey_for_sink_uid(self):
        sink = TREASURY_SINK_HOTKEYS[0]
        mg = type("MG", (), {})()
        mg.hotkeys = [sink, "miner"]
        mg.coldkeys = ["5VaultColdkey", "5MinerColdkey"]
        assert treasury_wallet_coldkey(mg, sink) == "5VaultColdkey"

    def test_missing_sink_returns_none(self):
        mg = type("MG", (), {})()
        mg.hotkeys = ["miner"]
        mg.coldkeys = ["5MinerColdkey"]
        assert treasury_wallet_coldkey(mg, TREASURY_SINK_HOTKEYS[0]) is None


class TestEnvOverride:
    def test_env_list_replaces_builtin(self, monkeypatch):
        from qbittensor.utils import treasury_sinks as sinks

        monkeypatch.setenv(
            "TREASURY_SINK_HOTKEYS",
            "5FirstSinkHotkey,5SecondSinkHotkey",
        )
        assert sinks.sink_hotkeys() == ("5FirstSinkHotkey", "5SecondSinkHotkey")
        assert sinks.default_sink_hotkey() == "5FirstSinkHotkey"
        assert sinks.resolve_sink_hotkey("5SecondSinkHotkey") == "5SecondSinkHotkey"
        assert sinks.resolve_sink_hotkey("5NotInEnvList") == "5FirstSinkHotkey"
