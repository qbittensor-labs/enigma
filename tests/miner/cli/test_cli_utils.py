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

import pytest
from unittest.mock import MagicMock, patch

from qbittensor.cli.miner.utils.color import COLORS, STATUS_BLUE, STATUS_GRAY, c, validator_status_color
from qbittensor.cli.miner.utils.constants import MINER_DB_TABLE_PREFIX
from qbittensor.cli.miner.mine_enigma import (
    _quiet_cli_logging,
    _short_ss58,
    _submission_table,
    _show_submission_details,
)


class TestColor:
    def test_c_wraps_index(self):
        assert c(0) == f"#{COLORS[0]}"
        assert c(len(COLORS)) == f"#{COLORS[0]}"

    def test_c_negative_index_wraps(self):
        assert c(-1) == f"#{COLORS[-1]}"

    def test_validator_status_color(self):
        assert validator_status_color("NotRun") == f"#{STATUS_BLUE}"
        assert validator_status_color("Submitted") == f"#{STATUS_BLUE}"
        assert validator_status_color("Cancelled") == f"#{STATUS_GRAY}"
        assert validator_status_color("Success") == c(2)
        assert validator_status_color("Pending") == c(3)
        assert validator_status_color("Running") == c(3)
        assert validator_status_color("Failure") == c(1)
        assert validator_status_color("BuildFailure") == c(1)


class TestConstants:
    def test_miner_db_table_prefix(self):
        assert MINER_DB_TABLE_PREFIX == "miner_submissions"


class TestQuietCliLogging:
    def test_raises_bt_logging_to_warning(self):
        import logging

        import bittensor as bt

        bt.logging._logger.setLevel(logging.INFO)
        _quiet_cli_logging()
        assert bt.logging._logger.level == logging.WARNING
        assert logging.getLogger("bittensor").level == logging.WARNING


class TestSubmissionStatusDisplay:
    """Cover the miner CLI submission list + detail rendering, especially has_failure suppression.

    When ANY validator reports a non-transient failure (e.g. BuildFailure, WallTimeFailure,
    generic Failure, UploadFailure), transient states (Submitted / Pending / Running) for
    OTHER validators are shown as dim '—' instead of the actual word.
    """

    def _mk_sub(self, tx="0xtxabc", ms="ms-xyz", submitted_at=None, validator_statuses=None):
        statuses = validator_statuses or {}
        runs = [
            {
                "validation_id": f"csv-{i}",
                "validator_hotkey": hk,
                "status": info.get("status"),
                "updated_at": info.get("updated_at"),
            }
            for i, (hk, info) in enumerate(statuses.items())
        ]
        return {
            "tx_hash": tx,
            "challenge_milestone_id": ms,
            "submitted_at": submitted_at,
            "required_validation_runs": 3,
            "runs": runs,
        }

    def test_table_no_failures_shows_real_statuses(self):
        subs = [
            self._mk_sub(validator_statuses={
                "5V1": {"status": "Submitted", "updated_at": None},
                "5V2": {"status": "Pending", "updated_at": None},
                "5V3": {"status": "Success", "updated_at": None},
            })
        ]
        table = _submission_table(subs, 0)
        # Just ensure it builds without error and has rows
        assert table is not None
        assert len(table.rows) == 1

    def test_table_has_failure_suppresses_transients_to_dash(self):
        # One failure present -> Submitted/Pending/Running on others become —
        subs = [
            self._mk_sub(validator_statuses={
                "5Good": {"status": "Success", "updated_at": None},
                "5Bad": {"status": "BuildFailure", "updated_at": None},
                "5Late": {"status": "Submitted", "updated_at": None},
                "5Busy": {"status": "Running", "updated_at": None},
                "5Wait": {"status": "Pending", "updated_at": None},
            })
        ]
        table = _submission_table(subs, 0)
        assert table is not None
        # The rendering code for the transients under has_failure path is executed.

    def test_table_uses_granular_failure_values(self):
        subs = [self._mk_sub(validator_statuses={
            "5V": {"status": "WallTimeFailure", "updated_at": None},
        })]
        table = _submission_table(subs, 0)
        assert table is not None

    def test_detail_view_suppresses_on_failure(self):
        console = MagicMock()
        sub = self._mk_sub(validator_statuses={
            "5V1": {"status": "Failure", "updated_at": None, "reported_at": None},
            "5V2": {"status": "Submitted", "updated_at": None, "reported_at": None},
            "5V3": {"status": "Running", "updated_at": None, "reported_at": None},
        })
        with patch("qbittensor.cli.miner.mine_enigma._wait_for_quit_key"):
            _show_submission_details(console, sub)
        # exercises has_failure + transient dash branch in details

    def test_detail_view_shows_pending_running_when_no_failure(self):
        console = MagicMock()
        sub = self._mk_sub(validator_statuses={
            "5V1": {"status": "Pending", "updated_at": None, "reported_at": None},
            "5V2": {"status": "Running", "updated_at": None, "reported_at": None},
            "5V3": {"status": "Success", "updated_at": None, "reported_at": None},
        })
        with patch("qbittensor.cli.miner.mine_enigma._wait_for_quit_key"):
            _show_submission_details(console, sub)
        assert console.print.called  # at least rendered something

    def test_detail_view_empty_runs_does_not_leak_markup(self):
        console = MagicMock()
        sub = self._mk_sub(validator_statuses={})
        with patch("qbittensor.cli.miner.mine_enigma._wait_for_quit_key"):
            _show_submission_details(console, sub)
        panel = console.print.call_args_list[0][0][0]
        plain = panel.renderable.plain
        assert "No validator status reports received yet." in plain
        assert "[dim]" not in plain
        assert "[/dim]" not in plain

    def test_empty_submissions_table(self):
        table = _submission_table([], 0)
        assert table is not None

    def test_short_ss58_first_six_last_four(self):
        hotkey = "5EZ52JMq4S7PYqzmLAggYahyDirMx3p1f1uBtLQgx6fk7kR8"
        assert _short_ss58(hotkey) == "5EZ52J...7kR8"
        assert _short_ss58(None) == "—"
        assert _short_ss58("short") == "short"

    def test_detail_view_shortens_validator_hotkey(self):
        console = MagicMock()
        hotkey = "5EZ52JMq4S7PYqzmLAggYahyDirMx3p1f1uBtLQgx6fk7kR8"
        sub = self._mk_sub(validator_statuses={
            hotkey: {
                "status": "IncorrectFailure",
                "updated_at": None,
                "reported_at": None,
            },
        })
        with patch("qbittensor.cli.miner.mine_enigma._wait_for_quit_key"):
            _show_submission_details(console, sub)
        panel = console.print.call_args_list[0][0][0]
        plain = panel.renderable.plain
        assert "5EZ52J...7kR8" in plain
        assert hotkey not in plain

    def test_table_shortens_tx_hash(self):
        tx = "0x" + ("ab" * 32)
        table = _submission_table([self._mk_sub(tx=tx)], 0)
        cells = table.columns[1]._cells
        assert cells[0] == _short_ss58(tx)
        assert tx not in cells[0]

    def test_detail_view_uses_q_footer_not_enter_prompt(self):
        from rich.align import Align
        from rich.text import Text

        console = MagicMock()
        sub = self._mk_sub()
        with patch("qbittensor.cli.miner.mine_enigma._wait_for_quit_key") as wait:
            _show_submission_details(console, sub)
        wait.assert_called_once()
        printed = [call.args[0] for call in console.print.call_args_list if call.args]
        footers = [p for p in printed if isinstance(p, Align)]
        assert footers
        footer_text = footers[0].renderable
        assert isinstance(footer_text, Text)
        assert "Back to Your Submissions" in footer_text.plain
        assert "Esc" in footer_text.plain
        assert "Back to menu" not in footer_text.plain


class _DummyLive:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, *args, **kwargs):
        pass


class TestPauseAfterSubmissionError:
    def test_format_milestones_pauses_after_submission_error(self):
        import click
        from qbittensor.cli.miner.mine_enigma import (
            CliApiAuth,
            KEY_ENTER,
            KEY_ESC,
            format_milestones,
        )

        detail = {
            "id": "challenge-1",
            "name": "Test",
            "milestones": [
                {"id": "m1", "name": "Milestone 1", "status": "Incomplete"},
            ],
        }
        auth = CliApiAuth(
            wallet_name="default",
            wallet_hotkey="default",
            network="finney",
            netuid=63,
        )
        keys = iter([KEY_ENTER, KEY_ESC])
        pauses: list[str] = []

        with (
            patch("qbittensor.cli.miner.mine_enigma._HAVE_TERMIOS", True),
            patch("qbittensor.cli.miner.mine_enigma.sys.stdin.isatty", return_value=True),
            patch("qbittensor.cli.miner.mine_enigma._read_key_unix", side_effect=lambda: next(keys)),
            patch("qbittensor.cli.miner.mine_enigma.Live", _DummyLive),
            patch("qbittensor.cli.miner.mine_enigma._clear_screen"),
            patch(
                "qbittensor.cli.miner.mine_enigma.run_milestone_solution_upload",
                side_effect=click.ClickException("wallet missing"),
            ),
            patch(
                "qbittensor.cli.miner.mine_enigma._pause_for_ack",
                side_effect=lambda console: pauses.append("paused"),
            ),
        ):
            format_milestones(MagicMock(), detail, api_auth=auth)

        assert pauses == ["paused"]

    def test_main_menu_pauses_after_submission_flow_error(self):
        from qbittensor.cli.miner.mine_enigma import (
            CliApiAuth,
            run_miner_main_menu,
        )

        auth = CliApiAuth(
            wallet_name="default",
            wallet_hotkey="default",
            network="finney",
            netuid=63,
        )
        pauses: list[str] = []
        prompts = iter(["1", "esc"])

        with (
            patch("qbittensor.cli.miner.mine_enigma._clear_screen"),
            patch("qbittensor.cli.miner.mine_enigma._HAVE_TERMIOS", False),
            patch("qbittensor.cli.miner.mine_enigma.sys.stdin.isatty", return_value=False),
            patch(
                "qbittensor.cli.miner.mine_enigma.query_and_format_challenges",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "qbittensor.cli.miner.mine_enigma.Prompt.ask",
                side_effect=lambda *a, **k: next(prompts),
            ),
            patch(
                "qbittensor.cli.miner.mine_enigma._pause_for_ack",
                side_effect=lambda console: pauses.append("paused"),
            ),
        ):
            run_miner_main_menu(MagicMock(), base_url=None, api_auth=auth)

        assert pauses == ["paused"]

    def test_pause_for_ack_uses_q_go_back_not_enter_prompt(self):
        from rich.align import Align
        from rich.text import Text
        from qbittensor.cli.miner.mine_enigma import _pause_for_ack

        console = MagicMock()
        with patch("qbittensor.cli.miner.mine_enigma._wait_for_quit_key") as wait:
            _pause_for_ack(console)
        wait.assert_called_once()
        printed = [call.args[0] for call in console.print.call_args_list if call.args]
        footers = [p for p in printed if isinstance(p, Align)]
        assert footers
        footer_text = footers[0].renderable
        assert isinstance(footer_text, Text)
        assert "Go back" in footer_text.plain
        assert "Esc" in footer_text.plain
        assert "Press Enter" not in footer_text.plain

    def test_read_key_unix_distinguishes_esc_from_arrows(self, monkeypatch):
        import qbittensor.cli.miner.mine_enigma as mod

        class Buf:
            def __init__(self, data: bytes):
                self.data = data

            def read(self, fd: int, n: int) -> bytes:
                out, self.data = self.data[:n], self.data[n:]
                return out

            def ready(self):
                return [0] if self.data else []

        monkeypatch.setattr(mod.termios, "tcgetattr", lambda fd: [])
        monkeypatch.setattr(mod.termios, "tcsetattr", lambda *a, **k: None)
        monkeypatch.setattr(mod.tty, "setcbreak", lambda fd: None)
        monkeypatch.setattr(mod.sys.stdin, "fileno", lambda: 0)

        buf = Buf(b"\x1b")
        monkeypatch.setattr(mod.os, "read", buf.read)
        monkeypatch.setattr(
            mod.select, "select", lambda *a, **k: (buf.ready(), [], [])
        )
        assert mod._read_key_unix() == mod.KEY_ESC
        assert mod._is_back_key(mod.KEY_ESC)

        buf = Buf(b"\x1b[A")
        monkeypatch.setattr(mod.os, "read", buf.read)
        monkeypatch.setattr(
            mod.select, "select", lambda *a, **k: (buf.ready(), [], [])
        )
        assert mod._read_key_unix() == mod.KEY_UP
        assert not mod._is_back_key(mod.KEY_UP)
        assert not mod._is_back_key(mod.KEY_DOWN)
        assert not mod._is_back_key("q")
        assert not mod._is_back_key("\x1b[A")
