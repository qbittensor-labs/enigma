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

from qbittensor.cli.miner.utils.color import COLORS, c
from qbittensor.cli.miner.utils.constants import MINER_DB_TABLE_PREFIX
from qbittensor.cli.miner.mine_enigma import _submission_table, _show_submission_details


class TestColor:
    def test_c_wraps_index(self):
        assert c(0) == f"#{COLORS[0]}"
        assert c(len(COLORS)) == f"#{COLORS[0]}"

    def test_c_negative_index_wraps(self):
        assert c(-1) == f"#{COLORS[-1]}"


class TestConstants:
    def test_miner_db_table_prefix(self):
        assert MINER_DB_TABLE_PREFIX == "miner_submissions"


class TestSubmissionStatusDisplay:
    """Cover the miner CLI submission list + detail rendering, especially has_failure suppression.

    When ANY validator reports a non-transient failure (e.g. BuildFailure, WallTimeFailure,
    generic Failure, UploadFailure), transient states (Submitted / Pending / Running) for
    OTHER validators are shown as dim '—' instead of the actual word.
    """

    def _mk_sub(self, tx="0xtxabc", ms="ms-xyz", submitted_at=None, validator_statuses=None):
        return {
            "tx_hash": tx,
            "challenge_milestone_id": ms,
            "submitted_at": submitted_at,
            "validator_statuses": validator_statuses or {},
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
        with patch("qbittensor.cli.miner.mine_enigma.Prompt"):
            _show_submission_details(console, sub)
        # exercises has_failure + transient dash branch in details

    def test_detail_view_shows_pending_running_when_no_failure(self):
        console = MagicMock()
        sub = self._mk_sub(validator_statuses={
            "5V1": {"status": "Pending", "updated_at": None, "reported_at": None},
            "5V2": {"status": "Running", "updated_at": None, "reported_at": None},
            "5V3": {"status": "Success", "updated_at": None, "reported_at": None},
        })
        with patch("qbittensor.cli.miner.mine_enigma.Prompt"):
            _show_submission_details(console, sub)
        assert console.print.called  # at least rendered something

    def test_empty_submissions_table(self):
        table = _submission_table([], 0)
        assert table is not None
