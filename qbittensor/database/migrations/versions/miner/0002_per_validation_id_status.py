# The MIT License (MIT)
# Copyright © 2026 qBitTensor Labs

"""Add validation_id to miner_submission_statuses and required_validation_runs to miner_submissions."""

from sqlalchemy import text

VERSION = 2
DESCRIPTION = "Per-validation_id miner status rows; required_validation_runs on submissions"


def upgrade(engine, telemetry_service=None):
    with engine.connect() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
        }
        if "miner_submissions" in tables:
            cols = [row[1] for row in conn.execute(text("PRAGMA table_info(miner_submissions)"))]
            if "required_validation_runs" not in cols:
                conn.execute(
                    text(
                        "ALTER TABLE miner_submissions "
                        "ADD COLUMN required_validation_runs INTEGER NOT NULL DEFAULT 3"
                    )
                )

        if "miner_submission_statuses" in tables:
            cols = [
                row[1]
                for row in conn.execute(text("PRAGMA table_info(miner_submission_statuses)"))
            ]
            if "validation_id" not in cols:
                conn.execute(
                    text(
                        "ALTER TABLE miner_submission_statuses "
                        "ADD COLUMN validation_id VARCHAR(50)"
                    )
                )
            # Rebuild to drop the old unique(hotkey, tx, milestone) so extra
            # runs from the same validator can coexist.
            conn.execute(text("PRAGMA foreign_keys=OFF"))
            conn.execute(
                text(
                    """
                    CREATE TABLE miner_submission_statuses_new (
                        id VARCHAR(50) NOT NULL PRIMARY KEY,
                        challenge_milestone_id VARCHAR(100) NOT NULL,
                        solution_status VARCHAR(100) NOT NULL,
                        validator_hotkey VARCHAR(100) NOT NULL,
                        tx_hash VARCHAR(100) NOT NULL,
                        validation_id VARCHAR(50),
                        created_at DATETIME,
                        updated_at DATETIME,
                        FOREIGN KEY(tx_hash) REFERENCES miner_submissions(tx_hash)
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    INSERT INTO miner_submission_statuses_new
                    SELECT id, challenge_milestone_id, solution_status, validator_hotkey,
                           tx_hash, validation_id, created_at, updated_at
                    FROM miner_submission_statuses
                    """
                )
            )
            conn.execute(text("DROP TABLE miner_submission_statuses"))
            conn.execute(
                text(
                    "ALTER TABLE miner_submission_statuses_new "
                    "RENAME TO miner_submission_statuses"
                )
            )
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS "
                    "uq_miner_submission_status_validation_id "
                    "ON miner_submission_statuses(validation_id) "
                    "WHERE validation_id IS NOT NULL"
                )
            )
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS "
                    "uq_miner_submission_status_offer_marker "
                    "ON miner_submission_statuses(validator_hotkey, tx_hash, challenge_milestone_id) "
                    "WHERE validation_id IS NULL"
                )
            )
            conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.commit()


def downgrade(engine, telemetry_service=None):
    pass
