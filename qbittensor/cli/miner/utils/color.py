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

"""Palette for Enigma CLI. Values are hex RGB without ``#``; use ``c()`` in Rich styles."""

# Order matches semantic roles: pink, violet, cyan, muted, accent (table borders / lines).
COLORS = ["5b8e7d", "bc4b51", "8cb369", "f4e285", "f4a259"]

# Validator-report status colors (aligned with portal SubmissionCard).
STATUS_BLUE = "60a5fa"  # NotRun / Submitted
STATUS_GRAY = "64748b"  # Cancelled


def c(index: int) -> str:
    """Rich hex color ``#rrggbb`` from ``COLORS`` (index wraps)."""
    return f"#{COLORS[index % len(COLORS)]}"


def validator_status_color(status: str) -> str:
    """Rich hex color for a validator-report status."""
    lower = str(status).lower()
    if lower in ("notrun", "submitted"):
        return f"#{STATUS_BLUE}"
    if lower == "cancelled":
        return f"#{STATUS_GRAY}"
    if lower == "success":
        return c(2)
    if lower in ("pending", "running"):
        return c(3)
    return c(1)
