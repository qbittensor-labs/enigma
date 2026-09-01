# The MIT License (MIT)
# Copyright © 2026 qBitTensor Labs

"""
Migration 0007: Add validation_id to challenge_solutions and drop uniqueness on tx_hash.

Each platform challenge_submission_validation row is its own local execution.
The same tx_hash may therefore appear more than once (padded extra runs /
cross-checks). Uniqueness moves to validation_id.
"""

from sqlalchemy import text

VERSION = 7
DESCRIPTION = "Add validation_id unique column; drop unique constraint on tx_hash"


def upgrade(engine, telemetry_service=None):
    with engine.connect() as conn:
        table_check = conn.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='challenge_solutions'"
            )
        )
        if not table_check.fetchone():
            return

        columns = [row[1] for row in conn.execute(text("PRAGMA table_info(challenge_solutions)"))]
        if "validation_id" not in columns:
            conn.execute(
                text("ALTER TABLE challenge_solutions ADD COLUMN validation_id VARCHAR(50)")
            )

        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "uq_challenge_solutions_validation_id "
                "ON challenge_solutions(validation_id) "
                "WHERE validation_id IS NOT NULL"
            )
        )

        # Recreate table without UNIQUE(tx_hash). SQLite cannot drop that
        # constraint in place.
        indexes = list(conn.execute(text("PRAGMA index_list(challenge_solutions)")))
        has_tx_unique = False
        for idx in indexes:
            # PRAGMA index_list: seq, name, unique, origin, partial
            if idx[2]:
                idx_info = list(conn.execute(text(f"PRAGMA index_info({idx[1]})")))
                cols = [c[2] for c in idx_info]
                if cols == ["tx_hash"]:
                    has_tx_unique = True
                    break
        if has_tx_unique:
            conn.execute(text("PRAGMA foreign_keys=OFF"))
            conn.execute(
                text(
                    """
                    CREATE TABLE challenge_solutions_new (
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
                        tx_hash VARCHAR(100) NOT NULL,
                        validation_id VARCHAR(50),
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
                    INSERT INTO challenge_solutions_new (
                        id, challenge_validation_solution_id, container_id, container_name,
                        image_id, challenge_id, challenge_milestone_id,
                        max_solution_runtime_seconds, absolute_path_to_solution,
                        submission_id, solution_status, tx_hash, validation_id,
                        miner_hotkey, cleaned, created_at, updated_at
                    )
                    SELECT
                        id, challenge_validation_solution_id, container_id, container_name,
                        image_id, challenge_id, challenge_milestone_id,
                        max_solution_runtime_seconds, absolute_path_to_solution,
                        submission_id, solution_status, tx_hash, validation_id,
                        miner_hotkey, cleaned, created_at, updated_at
                    FROM challenge_solutions
                    """
                )
            )
            conn.execute(text("DROP TABLE challenge_solutions"))
            conn.execute(text("ALTER TABLE challenge_solutions_new RENAME TO challenge_solutions"))
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS "
                    "uq_challenge_solutions_validation_id "
                    "ON challenge_solutions(validation_id) "
                    "WHERE validation_id IS NOT NULL"
                )
            )
            conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.commit()


def downgrade(engine, telemetry_service=None):
    pass
