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

from pathlib import Path

import pytest

from workbench.cli import _ensure_challenges
from workbench.runner.docker_runner import docker_build_command, ensure_enigma_challenges

ROOT = Path(__file__).resolve().parents[2]

EXAMPLE_DOCKERFILES = (
    ROOT / "workbench/challenges/breaking_rsa/example_solution/Dockerfile",
    ROOT / "workbench/challenges/hardening_quantum_proof/example_solution/Dockerfile",
    ROOT / "workbench/challenges/mock_challenge/example_solution/Dockerfile",
)


def test_example_dockerfiles_are_zip_shaped():
    for dockerfile in EXAMPLE_DOCKERFILES:
        text = dockerfile.read_text()
        assert "COPY --chown=miner:miner enigma_challenges" in text
        assert "qbittensor/challenges" not in text
        cmd = docker_build_command(str(dockerfile.parent), "test-image")
        assert "-f" not in cmd
        assert cmd[-1] == str(dockerfile.parent.resolve())


def test_ensure_copies_when_missing(tmp_path):
    dest, copied = ensure_enigma_challenges(str(tmp_path))
    assert copied
    assert dest.is_dir()
    assert (dest / "solution_output.py").is_file()
    dest2, copied2 = ensure_enigma_challenges(str(tmp_path))
    assert not copied2
    assert dest2 == dest


def test_ensure_rejects_symlink(tmp_path):
    (tmp_path / "enigma_challenges").symlink_to(ROOT / "qbittensor" / "challenges")
    with pytest.raises(ValueError, match="symlink"):
        ensure_enigma_challenges(str(tmp_path))


def test_cli_ensure_writes_and_is_idempotent(tmp_path):
    _ensure_challenges(str(tmp_path))
    assert (tmp_path / "enigma_challenges" / "solution_output.py").is_file()
    _ensure_challenges(str(tmp_path))
