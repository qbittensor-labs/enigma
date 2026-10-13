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

"""
Migration 0008: Track whether a terminal verify update reached the platform.

solution_status stays the real result. platform_reported is null while the
run is in progress, false when the verify PATCH has not landed, and true
once the platform accepts it. The retry columns schedule resends of
pending_verify_report until report_retry_until.
"""

from sqlalchemy import text

VERSION = 8
DESCRIPTION = "Add platform_reported and verify-retry columns to challenge_solutions"


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
        statements = []
        if "platform_reported" not in columns:
            statements.append(
                "ALTER TABLE challenge_solutions ADD COLUMN platform_reported BOOLEAN"
            )
        if "report_retry_until" not in columns:
            statements.append(
                "ALTER TABLE challenge_solutions ADD COLUMN report_retry_until DATETIME"
            )
        if "report_next_retry_at" not in columns:
            statements.append(
                "ALTER TABLE challenge_solutions ADD COLUMN report_next_retry_at DATETIME"
            )
        if "pending_verify_report" not in columns:
            statements.append(
                "ALTER TABLE challenge_solutions ADD COLUMN pending_verify_report TEXT"
            )
        for statement in statements:
            conn.execute(text(statement))
        if statements:
            conn.commit()


def downgrade(engine):
    pass
