# The MIT License (MIT)
# Copyright © 2026 qBitTensor Labs
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from qbittensor.challenges.solution_output import (
    SOLUTION_LOG_FILENAME,
    extract_artifacts,
    split_on_separator,
)
from .docker_runner import RunResult


def find_entry_point(
    solution_dir: str,
    challenge_type: str,
    entry_point: str | None = None,
    names: list[str] | None = None,
) -> str | None:
    """Find the solver script in the solution directory."""
    if entry_point:
        path = Path(solution_dir) / entry_point
        return str(path) if path.exists() else None

    defaults = {
        "breaking_rsa": ["breaking_rsa.py"],
        "hardening_quantum_proof": ["hardening_quantum_proof.py"],
        "mock": ["mock_solution.py"],
    }
    candidates = names or defaults.get(challenge_type, [f"{challenge_type}.py"])

    for name in candidates:
        path = Path(solution_dir) / name
        if path.exists():
            return str(path)

    return None


def run_direct(
    script_path: str,
    challenge_id: str,
    problem_json: str,
    output_dir: str,
    timeout: int = 300,
) -> RunResult:
    """Run a solver script directly as a subprocess."""
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["OUTPUT_DIR"] = output_dir

    # Solvers import `enigma_challenges`. `test`/`build` copy that package
    # into the solution dir when missing; this is a fallback for direct mode.
    solution_dir = Path(script_path).resolve().parent
    local_pkg = solution_dir / "enigma_challenges"
    symlink_dir = None
    if local_pkg.is_dir():
        env["PYTHONPATH"] = str(solution_dir) + os.pathsep + env.get("PYTHONPATH", "")
    else:
        repo_pkg = Path(__file__).resolve().parent.parent.parent / "qbittensor" / "challenges"
        if repo_pkg.is_dir():
            symlink_dir = tempfile.mkdtemp(prefix="workbench-pypath-")
            Path(symlink_dir, "enigma_challenges").symlink_to(repo_pkg)
            env["PYTHONPATH"] = symlink_dir + os.pathsep + env.get("PYTHONPATH", "")

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path, challenge_id, problem_json],
            capture_output=True, timeout=timeout, env=env,
        )
        duration = time.time() - start

        combined = (result.stdout or b"") + (result.stderr or b"")
        extract_artifacts(combined, output_dir)
        logs_bytes, _, _ = split_on_separator(combined)
        log_text = logs_bytes.decode("utf-8", errors="replace")

        log_path = os.path.join(output_dir, SOLUTION_LOG_FILENAME)
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write(log_text)

        return RunResult(
            success=result.returncode == 0,
            exit_code=result.returncode,
            log=log_text,
            duration=duration,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            success=False, exit_code=-1,
            log=f"Solver timed out after {timeout}s",
            duration=time.time() - start,
        )
    except FileNotFoundError:
        return RunResult(
            success=False, exit_code=-1,
            log=f"Python interpreter or script not found: {script_path}",
            duration=time.time() - start,
        )
    finally:
        if symlink_dir:
            shutil.rmtree(symlink_dir, ignore_errors=True)
