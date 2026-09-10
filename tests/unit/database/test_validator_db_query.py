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
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from qbittensor.database.migrations.runner import run_migrations_for_db
from qbittensor.database.validator.db_query import DBQuery
from qbittensor.utils.solution_status import SolutionStatus


@pytest.fixture
def validator_query(validator_db):
    return validator_db.db_query


def _insert_solution(query, **overrides):
    defaults = dict(
        challenge_validation_solution_id="cv-1",
        container_id="cid-abc123",
        container_name="ctr_name",
        image_id="val_label_sol_image",
        challenge_id="ch-1",
        challenge_milestone_id="milestone-1",
        absolute_path_to_solution="/tmp/solution/path",
        submission_id="sub-1",
        solution_status=SolutionStatus.RUNNING.value,
        tx_hash="0xabc",
        miner_hotkey="5Miner",
        cleaned=False,
        validation_id="csv-1",
    )
    defaults.update(overrides)
    # insert now returns the solution id (str) on success (preferred stable key)
    # or None/empty on failure. Bare assert checks for truthy id.
    returned_id = query.insert_challenge_solution(**defaults)
    assert returned_id
    defaults["id"] = returned_id
    return defaults


class TestDBQuery:
    def test_insert_and_get_by_stable_key(self, validator_query):
        data = _insert_solution(validator_query)
        # Prefer lookup by the stable PK id (returned from insert and now in data)
        row = validator_query.get_challenge_solution_by_id(data["id"])
        assert row is not None
        assert row.absolute_path_to_solution == data["absolute_path_to_solution"]

        # Also retrievable by the stable primary key id (preferred once you have it
        # in a passed-around object like SolutionPostProcessInfo)
        row_by_id = validator_query.get_challenge_solution_by_id(data["id"])
        assert row_by_id is not None
        assert row_by_id.submission_id == data["submission_id"]

    def test_update_solution_status_by_stable_keys(self, validator_query):
        data = _insert_solution(validator_query)
        # Use the id returned by insert (now present in data dict)
        original_id = data["id"]
        row = validator_query.get_challenge_solution_by_id(original_id)
        assert row is not None

        # By the internal id (preferred, carried in SolutionPostProcessInfo / .execution)
        ok = validator_query.update_solution_status_by_id(
            original_id, SolutionStatus.SUCCESS.value
        )
        assert ok is True
        row = validator_query.get_challenge_solution_by_id(original_id)
        assert row.solution_status == SolutionStatus.SUCCESS.value

        # Also test by id again for failed
        ok2 = validator_query.update_solution_status_by_id(
            original_id, SolutionStatus.FAILURE.value
        )
        assert ok2 is True
        row = validator_query.get_challenge_solution_by_id(original_id)
        assert row.solution_status == SolutionStatus.FAILURE.value

    def test_insert_early_and_update_challenge_solution(self, validator_query):
        created_id = validator_query.create_challenge_solution(
            challenge_validation_solution_id="cv-early",
            challenge_milestone_id="milestone-1",
            submission_id="sub-early",
            solution_status=SolutionStatus.PENDING.value,
            tx_hash="0xearly",
            miner_hotkey="5Miner",
            challenge_id="ch-1",
            validation_id="csv-early",
        )
        assert created_id
        row = validator_query.get_challenge_solution_by_id(created_id)
        assert row is not None
        assert row.solution_status == SolutionStatus.PENDING.value
        assert row.id == created_id  # demonstrate id returned from create

        # Update using stable id (preferred; no tx_hash lookup for the update)
        assert validator_query.update_challenge_solution_by_id(
            solution_id=created_id,
            container_id="cid-final",
            container_name="ctr_final",
            image_id="img_final",
            absolute_path_to_solution="/tmp/final/path",
            solution_status=SolutionStatus.RUNNING.value,
        )
        row = validator_query.get_challenge_solution_by_id(created_id)
        assert row.container_id == "cid-final"
        assert row.absolute_path_to_solution == "/tmp/final/path"
        assert row.solution_status == SolutionStatus.RUNNING.value

    def test_create_challenge_solution_without_challenge_id(self, validator_query):
        created_id = validator_query.create_challenge_solution(
            challenge_validation_solution_id="cv-cross",
            challenge_milestone_id="milestone-1",
            submission_id="sub-cross",
            solution_status=SolutionStatus.PENDING.value,
            tx_hash="0xcross",
            miner_hotkey="5Miner",
            challenge_id=None,
            validation_id="csv-cross",
        )
        assert created_id
        row = validator_query.get_challenge_solution_by_id(created_id)
        assert row is not None
        assert row.challenge_id is None

    def test_get_image_id_lookups(self, validator_query):
        data = _insert_solution(validator_query)
        # Verify data via stable key (by id) then read field
        row = validator_query.get_challenge_solution_by_id(data["id"])
        assert row is not None
        assert row.image_id == data["image_id"]

    def test_get_submission_and_milestone_via_stable_keys(self, validator_query):
        data = _insert_solution(validator_query)
        # Use stable key (by id) + read from row
        row = validator_query.get_challenge_solution_by_id(data["id"])
        assert row is not None
        assert row.submission_id == data["submission_id"]
        assert row.challenge_milestone_id == data["challenge_milestone_id"]

        # Also via get_solution_by_submission_id (still supported for external IDs)
        row2 = validator_query.get_solution_by_submission_id(data["submission_id"])
        assert row2 is not None
        assert row2.challenge_milestone_id == data["challenge_milestone_id"]

    def test_create_rejects_tx_reuse_on_identifier_mismatch(self, validator_query):
        """Cross-check / re-run guard: same tx is allowed for identical file-upload/ch/mil (or same sub),
        but rejected if any of file-upload (challenge_validation_solution_id), challenge_id,
        milestone, or (for different work) submission_id differ. This disallows tx reuse.
        """
        tx = "0xreusetest"

        # First legitimate creation (as in a normal submission)
        first_id = validator_query.create_challenge_solution(
            challenge_validation_solution_id="upload-abc",  # the "file upload"
            challenge_milestone_id="m1",
            submission_id="sub-original",
            solution_status=SolutionStatus.PENDING.value,
            tx_hash=tx,
            miner_hotkey="5Miner",
            challenge_id="ch-1",
            validation_id="csv-reuse",
        )
        assert first_id

        row = validator_query.get_challenge_solution_by_id(first_id)
        assert row is not None
        assert row.challenge_validation_solution_id == "upload-abc"

        # Re-run with *exact same* identifiers (including submission) → allowed (re-use/refresh)
        same_id = validator_query.create_challenge_solution(
            challenge_validation_solution_id="upload-abc",
            challenge_milestone_id="m1",
            submission_id="sub-original",
            solution_status=SolutionStatus.PENDING.value,
            tx_hash=tx,
            miner_hotkey="5Miner",
            challenge_id="ch-1",
            validation_id="csv-reuse",
        )
        assert same_id
        assert same_id == first_id  # same row (upsert)

        # Cross-check style re-run: different submission_id (cross-check task id), but
        # *same* file upload / challenge / milestone → allowed (re-run the same work)
        cross_id = validator_query.create_challenge_solution(
            challenge_validation_solution_id="upload-abc",  # same file upload
            challenge_milestone_id="m1",
            submission_id="cross-check-task-xyz",  # different claim/sub id
            solution_status=SolutionStatus.PENDING.value,
            tx_hash=tx,
            miner_hotkey="5Miner",
            challenge_id="ch-1",
            validation_id="csv-reuse",
        )
        assert cross_id
        assert cross_id == first_id

        # Now a bad re-use: same tx, different file upload → rejected
        bad_file = validator_query.create_challenge_solution(
            challenge_validation_solution_id="upload-different-file",
            challenge_milestone_id="m1",
            submission_id="sub-original",
            solution_status=SolutionStatus.PENDING.value,
            tx_hash=tx,
            miner_hotkey="5Miner",
            challenge_id="ch-1",
            validation_id="csv-reuse",
        )
        assert bad_file is None

        # Different milestone for same tx → rejected
        bad_mil = validator_query.create_challenge_solution(
            challenge_validation_solution_id="upload-abc",
            challenge_milestone_id="m2-different",
            submission_id="sub-original",
            solution_status=SolutionStatus.PENDING.value,
            tx_hash=tx,
            miner_hotkey="5Miner",
            challenge_id="ch-1",
            validation_id="csv-reuse",
        )
        assert bad_mil is None

        # Different challenge for same validation_id → rejected
        bad_ch = validator_query.create_challenge_solution(
            challenge_validation_solution_id="upload-abc",
            challenge_milestone_id="m1",
            submission_id="sub-original",
            solution_status=SolutionStatus.PENDING.value,
            tx_hash=tx,
            miner_hotkey="5Miner",
            challenge_id="ch-999",
            validation_id="csv-reuse",
        )
        assert bad_ch is None

        # Different submission_id *and* different work for same validation_id → rejected
        bad_sub = validator_query.create_challenge_solution(
            challenge_validation_solution_id="upload-other",
            challenge_milestone_id="m-other",
            submission_id="sub-different-work",
            solution_status=SolutionStatus.PENDING.value,
            tx_hash=tx,
            miner_hotkey="5Miner",
            challenge_id="ch-other",
            validation_id="csv-reuse",
        )
        assert bad_sub is None

        # The original row is untouched
        row = validator_query.get_challenge_solution_by_id(first_id)
        assert row.challenge_validation_solution_id == "upload-abc"
        assert row.challenge_milestone_id == "m1"

    def test_remove_by_id(self, validator_query):
        data = _insert_solution(validator_query)
        # Use id returned from _insert_solution (now in data)
        assert validator_query.remove_solution_by_id(data["id"]) is True
        assert validator_query.get_challenge_solution_by_id(data["id"]) is None

    def test_insert_for_maintenance_incentive_and_get_active(self, validator_query):
        assert validator_query.insert_for_maintenance_incentive("5Miner", "m1", "0xincentive") is True
        # duplicate tx_hash is ignored
        assert validator_query.insert_for_maintenance_incentive("5Miner", "m1", "0xincentive") is True
        miners = validator_query.get_active_miners()
        assert "5Miner" in miners

    def test_get_miner_submission_statuses(self, validator_query):
        data = _insert_solution(validator_query, solution_status=SolutionStatus.SUCCESS.value)
        statuses = validator_query.get_miner_submission_statuses(data["miner_hotkey"])
        assert len(statuses) == 1
        assert statuses[0].tx_hash == data["tx_hash"]

    def test_missing_rows_return_none_or_empty(self, validator_query):
        # get_solution_by_tx_hash removed; test other missing cases
        assert validator_query.get_miner_submission_statuses("5Nobody") == []
        # New stable-key updaters should gracefully return False for missing
        assert validator_query.update_solution_status_by_id("no-such-id", "foo") is False

    def test_create_allows_same_tx_hash_with_distinct_validation_ids(self, validator_query):
        """Multiple platform validations may share a fee tx (extra runs / cross-checks)."""
        tx = "0xshared-fee"
        first = validator_query.create_challenge_solution(
            challenge_validation_solution_id="upload-abc",
            challenge_milestone_id="m1",
            submission_id="sub-1",
            solution_status=SolutionStatus.PENDING.value,
            tx_hash=tx,
            miner_hotkey="5Miner",
            challenge_id="ch-1",
            validation_id="csv-run-1",
        )
        second = validator_query.create_challenge_solution(
            challenge_validation_solution_id="upload-abc",
            challenge_milestone_id="m1",
            submission_id="sub-1",
            solution_status=SolutionStatus.PENDING.value,
            tx_hash=tx,
            miner_hotkey="5Miner",
            challenge_id="ch-1",
            validation_id="csv-run-2",
        )
        assert first and second
        assert first != second


def _query_on_pre_0007_migrated_db(tmp_path):
    """DBQuery against the schema migration 0007 produces on an existing table.

    Production validators that existed before validation_id uniqueness get a
    PARTIAL unique index (`WHERE validation_id IS NOT NULL`), not a UNIQUE
    constraint. Tests that go through create_all do not cover that path.
    """
    db_path = tmp_path / "migrated_0007.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE challenge_solutions (
                    id VARCHAR(50) NOT NULL PRIMARY KEY,
                    challenge_validation_solution_id VARCHAR(50) NOT NULL,
                    container_id VARCHAR(100) NOT NULL,
                    container_name VARCHAR(100) NOT NULL,
                    image_id VARCHAR(100) NOT NULL,
                    challenge_id VARCHAR(100),
                    challenge_milestone_id VARCHAR(100) NOT NULL,
                    max_solution_runtime_seconds INTEGER,
                    absolute_path_to_solution VARCHAR(100) NOT NULL,
                    submission_id VARCHAR(100) NOT NULL,
                    solution_status VARCHAR(100) NOT NULL,
                    tx_hash VARCHAR(100) NOT NULL UNIQUE,
                    miner_hotkey VARCHAR(100) NOT NULL,
                    cleaned BOOLEAN NOT NULL DEFAULT 0,
                    created_at DATETIME,
                    updated_at DATETIME
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE schema_migrations (
                    version INTEGER PRIMARY KEY,
                    description VARCHAR(255) NOT NULL,
                    applied_at DATETIME
                )
                """
            )
        )
        for version, description in (
            (1, "baseline"),
            (2, "cleaned"),
            (3, "runtime"),
            (4, "verified"),
            (5, "clear"),
            (6, "force"),
        ):
            conn.execute(
                text(
                    "INSERT INTO schema_migrations (version, description, applied_at) "
                    "VALUES (:v, :d, datetime('now'))"
                ),
                {"v": version, "d": description},
            )
        conn.commit()

    run_migrations_for_db(engine, "challenge_solutions")
    return DBQuery(sessionmaker(bind=engine)), engine


def _assert_0007_partial_unique_index(engine):
    with engine.connect() as conn:
        indexes = list(conn.execute(text("PRAGMA index_list(challenge_solutions)")))
        validation_unique = None
        tx_hash_unique = False
        for idx in indexes:
            if not idx[2]:
                continue
            cols = [c[2] for c in conn.execute(text(f"PRAGMA index_info({idx[1]})"))]
            if cols == ["validation_id"]:
                validation_unique = idx
            if cols == ["tx_hash"]:
                tx_hash_unique = True
        assert validation_unique is not None, indexes
        # origin 'c' + partial=1 is the 0007 CREATE UNIQUE INDEX ... WHERE form
        assert validation_unique[3] == "c"
        assert validation_unique[4] == 1
        assert tx_hash_unique is False


class TestDBQueryMigrated0007Schema:
    def test_create_challenge_solution_on_partial_unique_index(self, tmp_path):
        """Regression: ON CONFLICT (validation_id) fails on 0007's partial unique index."""
        query, engine = _query_on_pre_0007_migrated_db(tmp_path)
        _assert_0007_partial_unique_index(engine)

        created = query.create_challenge_solution(
            challenge_validation_solution_id="cv-1",
            challenge_milestone_id="m1",
            submission_id="sub-1",
            solution_status=SolutionStatus.PENDING.value,
            tx_hash="0xabc",
            miner_hotkey="5EZ52JMq4S7PYqzmLAggYahyDirMx3p1f1uBtLQgx6fk7kR8",
            challenge_id="fac1d276-e99c-4ee4-8307-e8709f94d471",
            validation_id="csv-1",
        )
        assert created
        row = query.get_challenge_solution_by_id(created)
        assert row is not None
        assert row.validation_id == "csv-1"
        assert row.miner_hotkey == "5EZ52JMq4S7PYqzmLAggYahyDirMx3p1f1uBtLQgx6fk7kR8"

        reused = query.create_challenge_solution(
            challenge_validation_solution_id="cv-1",
            challenge_milestone_id="m1",
            submission_id="sub-1",
            solution_status=SolutionStatus.PENDING.value,
            tx_hash="0xabc",
            miner_hotkey="5EZ52JMq4S7PYqzmLAggYahyDirMx3p1f1uBtLQgx6fk7kR8",
            challenge_id="fac1d276-e99c-4ee4-8307-e8709f94d471",
            validation_id="csv-1",
        )
        assert reused == created

        second = query.create_challenge_solution(
            challenge_validation_solution_id="cv-1",
            challenge_milestone_id="m1",
            submission_id="sub-1",
            solution_status=SolutionStatus.PENDING.value,
            tx_hash="0xabc",
            miner_hotkey="5EZ52JMq4S7PYqzmLAggYahyDirMx3p1f1uBtLQgx6fk7kR8",
            challenge_id="fac1d276-e99c-4ee4-8307-e8709f94d471",
            validation_id="csv-2",
        )
        assert second
        assert second != created
