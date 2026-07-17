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

from datetime import datetime, timezone

import pytest

from qbittensor.database.db_connection import DBConnection


@pytest.fixture
def miner_query(miner_db):
    return miner_db.db_query_miner


class TestDBQueryMiner:
    def _insert(self, query, tx_hash, miner_db, *, submitted_at=None, updated_at=None, challenge_milestone_id="milestone-1"):
        ok = query.insert_miner_submission(
            upload_id="up-1",
            challenge_milestone_id=challenge_milestone_id,
            miner_hotkey="5MinerHotkey",
            tx_hash=tx_hash,
            challenge_id="ch-1",
            transfer_block_hash="block",
            transfer_from_ss58="5From",
            transfer_to_ss58="5To",
            transfer_amount_rao="1000",
        )
        assert ok is True
        if submitted_at is not None or updated_at is not None:
            session = miner_db.get_db_session()
            try:
                from qbittensor.database.miner.db_models import MinerSubmission
                row = session.query(MinerSubmission).filter_by(tx_hash=tx_hash).one()
                if submitted_at is not None:
                    row.submitted_at = submitted_at
                if updated_at is not None:
                    row.updated_at = updated_at
                session.commit()
            finally:
                session.close()

    def test_insert_and_update_by_tx_hash(self, miner_query, miner_db):
        self._insert(miner_query, "0xfirst", miner_db)
        ok = miner_query.insert_miner_submission(
            upload_id="up-2",
            challenge_milestone_id="milestone-2",
            miner_hotkey="5MinerHotkey",
            tx_hash="0xfirst",
            challenge_id="ch-1",
            transfer_block_hash="block2",
            transfer_from_ss58="5From",
            transfer_to_ss58="5To",
            transfer_amount_rao="2000",
        )
        assert ok is True
        nxt = miner_query.get_next_miner_submission()
        assert nxt.tx_hash == "0xfirst"
        assert nxt.upload_id == "up-2"

    def test_get_next_prefers_unsubmitted(self, miner_query, miner_db):
        self._insert(miner_query, "0xold", miner_db, submitted_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
        self._insert(miner_query, "0xnew", miner_db)
        nxt = miner_query.get_next_miner_submission()
        assert nxt.tx_hash == "0xnew"

    def test_record_solution_served_to_validator(self, miner_query, miner_db):
        self._insert(miner_query, "0xserve", miner_db)
        assert miner_query.record_solution_served_to_validator("0xserve") is True
        nxt = miner_query.get_next_miner_submission()
        assert nxt.submitted_at is not None

    def test_record_solution_served_missing_tx_returns_false(self, miner_query):
        assert miner_query.record_solution_served_to_validator("0xmissing") is False

    def test_insert_miner_submission_status_insert_and_update(self, miner_query, miner_db):
        self._insert(miner_query, "0xstatus", miner_db)
        assert miner_query.insert_miner_submission_status(
            "m1", "RUNNING", "5Validator", "0xstatus"
        ) is True
        assert miner_query.insert_miner_submission_status(
            "m1", "Success", "5Validator", "0xstatus"
        ) is True

    def test_insert_miner_submission_status_unknown_tx_returns_false_no_error(self, miner_query):
        """Unknown tx_hash (e.g. stale status from validator history or cross-check) must be ignored gracefully.

        This prevents FOREIGN KEY constraint errors when validators send status updates
        (including Failure) for submissions the miner no longer has in its local DB.
        """
        # Must return False without raising IntegrityError on the FK to miner_submissions.tx_hash
        assert miner_query.insert_miner_submission_status(
            challenge_milestone_id="milestone-x",
            solution_status="Failure",
            validator_hotkey="5Validator",
            tx_hash="0xnonexistent-tx-that-was-never-inserted",
        ) is False

    def test_record_submitted_only_on_first_offer(self, miner_query, miner_db):
        """record_submission_submitted_to_validator inserts 'Submitted' only on first call (DO NOTHING)."""
        self._insert(miner_query, "0xfirst-offer", miner_db, challenge_milestone_id="ms-1")
        vhot = "5Val1"

        assert miner_query.record_submission_submitted_to_validator("0xfirst-offer", vhot, "ms-1") is True
        # Second call for identical (validator,tx,milestone) does nothing but succeeds
        assert miner_query.record_submission_submitted_to_validator("0xfirst-offer", vhot, "ms-1") is True

        listed = miner_query.list_my_submissions_with_status(limit=10)
        sub = next((s for s in listed if s["tx_hash"] == "0xfirst-offer"), None)
        assert sub is not None
        assert sub["validator_statuses"]["5Val1"]["status"] == "Submitted"

    def test_get_next_for_validator_reoffers_on_submitted(self, miner_query, miner_db):
        """'Submitted' marker does not count as seen; re-offer until real status arrives."""
        self._insert(miner_query, "0xreoffer", miner_db, challenge_milestone_id="ms-re")
        vhot = "5ValReoffer"

        miner_query.record_submission_submitted_to_validator("0xreoffer", vhot, "ms-re")
        nxt = miner_query.get_next_miner_submission_for_validator(vhot)
        assert nxt is not None and nxt.tx_hash == "0xreoffer"

    def test_get_next_for_validator_stops_after_real_status(self, miner_query, miner_db):
        """Once validator reports real status (Pending/Success/Failure/granular), stop re-offering to it."""
        self._insert(miner_query, "0xclaimed", miner_db, challenge_milestone_id="ms-claim")
        vhot = "5ValClaim"
        miner_query.record_submission_submitted_to_validator("0xclaimed", vhot, "ms-claim")
        miner_query.insert_miner_submission_status("ms-claim", "Pending", vhot, "0xclaimed")
        assert miner_query.get_next_miner_submission_for_validator(vhot) is None

        # Granular failure reason value as status also blocks
        self._insert(miner_query, "0xfailed", miner_db, challenge_milestone_id="ms-fail")
        miner_query.record_submission_submitted_to_validator("0xfailed", vhot, "ms-fail")
        miner_query.insert_miner_submission_status("ms-fail", "BuildFailure", vhot, "0xfailed")
        assert miner_query.get_next_miner_submission_for_validator(vhot) is None

    def test_get_next_for_validator_independent_per_validator(self, miner_query, miner_db):
        """Each validator has its own seen state; one validator's status does not affect another."""
        self._insert(miner_query, "0xmulti", miner_db, challenge_milestone_id="ms-multi")
        v1, v2 = "5ValOne", "5ValTwo"
        miner_query.record_submission_submitted_to_validator("0xmulti", v1, "ms-multi")
        miner_query.insert_miner_submission_status("ms-multi", "Success", v1, "0xmulti")

        # v1 has real status -> no more for v1
        assert miner_query.get_next_miner_submission_for_validator(v1) is None
        # v2 has only nothing/Submitted path -> still offered
        assert miner_query.get_next_miner_submission_for_validator(v2) is not None

    def test_list_my_submissions_with_status_aggregates_per_validator(self, miner_query, miner_db):
        """list_my... returns per-validator latest status (including granular reasons)."""
        self._insert(miner_query, "0xlist", miner_db, challenge_milestone_id="ms-list")
        miner_query.record_submission_submitted_to_validator("0xlist", "5V-A", "ms-list")
        miner_query.insert_miner_submission_status("ms-list", "Running", "5V-A", "0xlist")
        miner_query.insert_miner_submission_status("ms-list", "WallTimeFailure", "5V-B", "0xlist")

        listed = miner_query.list_my_submissions_with_status(limit=5)
        sub = next((s for s in listed if s["tx_hash"] == "0xlist"), None)
        assert sub is not None
        vs = sub["validator_statuses"]
        assert vs["5V-A"]["status"] == "Running"
        assert vs["5V-B"]["status"] == "WallTimeFailure"
