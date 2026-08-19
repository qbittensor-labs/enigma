# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
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

"""Small unit tests for gpu_access that mock DockerOps (no real docker)."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from qbittensor.validator.utils.gpu_verification import gpu_access


def test_gpu_container_requires_gpus(monkeypatch):
    # Pass gpus="" explicitly: env alone falls back to DEFAULT "all" and would
    # attempt a real docker build. The function still short-circuits on empty gpus.
    monkeypatch.setenv("VALIDATOR_DOCKER_GPUS", "")
    assert gpu_access.test_gpu_container(gpus="") is False


def test_gpu_container_success_path(monkeypatch):
    """Happy path: build and run succeed, marker present."""
    monkeypatch.setenv("VALIDATOR_DOCKER_GPUS", "all")

    mock_ops = MagicMock()
    # build succeeds
    mock_ops.build_image.return_value = MagicMock(returncode=0)
    # run succeeds with marker in output
    run_res = MagicMock(stdout="some output\nSUCCESS: GPU is accessible from this container.\n", stderr="")
    mock_ops.run_container.return_value = run_res

    with patch("qbittensor.validator.utils.gpu_verification.gpu_access.DockerOps", return_value=mock_ops):
        result = gpu_access.test_gpu_container()
        assert result is True

    mock_ops.build_image.assert_called_once()
    mock_ops.run_container.assert_called_once()


def test_gpu_container_build_fails(monkeypatch):
    monkeypatch.setenv("VALIDATOR_DOCKER_GPUS", "all")

    mock_ops = MagicMock()
    err = subprocess.CalledProcessError(1, ["build"])
    err.stderr = b"build error"
    mock_ops.build_image.side_effect = err

    with patch("qbittensor.validator.utils.gpu_verification.gpu_access.DockerOps", return_value=mock_ops):
        result = gpu_access.test_gpu_container()
        assert result is False


def test_remove_image_uses_rmi():
    mock_ops = MagicMock()
    with patch("qbittensor.validator.utils.gpu_verification.gpu_access.DockerOps", return_value=mock_ops):
        gpu_access._remove_gpu_verification_image("test-img")
        mock_ops.rmi.assert_called_once_with("test-img", force=True)
