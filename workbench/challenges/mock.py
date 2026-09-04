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

import os

from qbittensor.challenges.mock_challenge import (
    MockChallenge, Problem, Solution, Verif,
    sign_mock_payload, generate_keypair,
)


# Default public key for local workbench tests. Matches the baked-in Ed25519
# seed in example_solution/mock_solution.py so `enigma-workbench test mock`
# verifies without extra flags. Override with ENIGMA_MOCK_PUBLIC_KEY or
# --public-key if you sign with a different key.
DEFAULT_PUBLIC_KEY = "7b4e2f00b6d6ff2e338b42b2bf599201d285f1fddd4f7b3b99a824fdd36cc8dc"


def _get_public_key(override: str | None = None) -> str:
    """Resolve the public key: explicit arg > env var > default."""
    return override or os.environ.get("ENIGMA_MOCK_PUBLIC_KEY") or DEFAULT_PUBLIC_KEY


def generate_mock(difficulty: int = 1, public_key_hex: str | None = None):
    """Generate a mock challenge. Returns (problem, verif)."""
    pub = _get_public_key(public_key_hex)
    challenge = MockChallenge(difficulty=difficulty, public_key_hex=pub)
    problem, verif = challenge.generate(seed=0)
    return problem, verif


def solve_mock(private_key_hex: str | None = None) -> Solution:
    """Sign a mock challenge payload. Returns a Solution."""
    key = private_key_hex or os.environ.get("ENIGMA_MOCK_PRIVATE_KEY")
    if not key:
        raise ValueError(
            "Private key required. Set ENIGMA_MOCK_PRIVATE_KEY env var "
            "or pass --private-key."
        )
    return sign_mock_payload(key)
