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

import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import click

from workbench.challenges import breaking_rsa as breaking_rsa_challenge
from workbench.challenges import hardening_quantum_proof as hqp_challenge
from workbench.challenges import mock as mock_challenge
from workbench.runner.docker_runner import (
    check_docker, build_image, run_container, DEFAULT_WALL_TIME,
)

# Best-effort host NVIDIA/CUDA version for local developer visibility
# (validators always report this via telemetry; workbench is the local analog).
try:
    from qbittensor.utils.services.telemetry import _get_nvidia_driver_info
except Exception:
    _get_nvidia_driver_info = None  # type: ignore
from workbench.runner.direct_runner import find_entry_point, run_direct
from workbench.validator import validate_output, validate_dockerfile_security
from workbench.verifier import verify_breaking_rsa, verify_hardening_quantum_proof, verify_mock
from workbench.report import print_report

from qbittensor.challenges.solution_output import RESULT_JSON_FILENAME


def _prepare_challenge_input_dir(workspace: str) -> str:
    """Create a fresh host dir for the /challenge_input mount (validator parity)."""
    mount_dir = os.path.join(workspace, "challenge_input_mount")
    os.makedirs(mount_dir, mode=0o755, exist_ok=True)
    return mount_dir


def _write_breaking_rsa_challenge_input(mount_dir: str, problem) -> None:
    """Write challenge_input.json matching validator breaking_rsa_setup."""
    path = os.path.join(mount_dir, "challenge_input.json")
    with open(path, "w") as f:
        json.dump(problem.to_dict(), f)


def _write_hqp_challenge_input(
    mount_dir: str,
    challenge_id: str,
    difficulty: int,
    host_qasm_path: str,
) -> None:
    """Write circuit.qasm + challenge_input.json matching validator hqp_setup."""
    qasm_dest = os.path.join(mount_dir, "circuit.qasm")
    shutil.copy2(host_qasm_path, qasm_dest)

    challenge_input = {
        "challenge_id": challenge_id,
        "difficulty": difficulty,
        "qasm_file": "/challenge_input/circuit.qasm",
    }
    path = os.path.join(mount_dir, "challenge_input.json")
    with open(path, "w") as f:
        json.dump(challenge_input, f)


def _write_mock_challenge_input(mount_dir: str) -> None:
    """Write challenge_input.txt matching validator mock_solution_setup."""
    path = os.path.join(mount_dir, "challenge_input.txt")
    with open(path, "w") as f:
        f.write(
            "This is a simple solution setup file. "
            "The output should include the word 'Hello'. Hello!"
        )


def _warn_non_default(wall_time, allow_network):
    """Warn when settings diverge from validator defaults."""
    warnings = []
    if wall_time != DEFAULT_WALL_TIME:
        warnings.append(
            f"  Wall time: {wall_time}s (validator enforces {DEFAULT_WALL_TIME}s / {DEFAULT_WALL_TIME // 3600}h)"
        )
    if allow_network:
        warnings.append(
            "  Network: enabled (validator runs containers with --network none)"
        )
    if warnings:
        click.echo("\nWARNING: Non-default settings detected:")
        for w in warnings:
            click.echo(w)
        click.echo("Test with default settings before submitting to the validator.\n")


def _preflight_dockerfile(solution: str) -> None:
    """Check Dockerfile presence (case-insensitive) and platform security policy.

    Fails fast with a clear message before attempting docker build, so the
    developer experience matches what the validator will enforce.
    """
    sec = validate_dockerfile_security(solution)
    if not sec.passed:
        if "No Dockerfile" in (sec.message or ""):
            click.echo(f"Error: No Dockerfile found in {solution}. Docker mode requires a Dockerfile.")
        else:
            click.echo(f"Error: Dockerfile rejected by platform security policy: {sec.message}")
            click.echo("This solution will be rejected by the validator. Fix the Dockerfile and retest.")
        sys.exit(1)


def _print_nvidia_info_if_available() -> None:
    """Print host NVIDIA driver + CUDA version (if detectable) for local debugging.

    Useful when testing solutions that use --gpus / CUDA inside their containers.
    The same data is reported by live validators in telemetry (system_* keys).
    """
    if _get_nvidia_driver_info is None:
        return
    try:
        drv, cuda = _get_nvidia_driver_info()
        if drv != "none" or cuda != "none":
            click.echo(f"[NVIDIA] driver={drv}  cuda={cuda}")
        # else: silent on non-GPU machines (normal for most dev)
    except Exception:
        pass


@click.group()
def cli():
    """Enigma Developer Workbench -- local testing tool for challenge solutions."""
    pass


@cli.group()
def test():
    """Test a solution against a challenge."""
    pass


@test.command("breaking-rsa")
@click.option("--difficulty", default=300, help="Bit-width of the semiprime to factor (default: 300)")
@click.option("--solution", required=True, type=click.Path(exists=True), help="Path to solution directory")
@click.option("--mode", type=click.Choice(["docker", "direct"]), default="docker", help="Execution mode")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility")
@click.option("--wall-time", default=DEFAULT_WALL_TIME, help=f"Wall time in seconds (default: {DEFAULT_WALL_TIME} = {DEFAULT_WALL_TIME // 3600}h, matches validator)")
@click.option("--allow-network", is_flag=True, help="Allow network access in container (validator disables network)")
@click.option("--keep-output", is_flag=True, help="Keep output directory after test")
def test_breaking_rsa(difficulty, solution, mode, seed, wall_time, allow_network, keep_output):
    """Test a Breaking RSA solution."""
    total_start = time.time()

    # Check prerequisites
    if mode == "docker":
        if not check_docker():
            click.echo("Error: Docker is not available. Install Docker or use --mode direct.")
            sys.exit(1)
        _preflight_dockerfile(solution)
        _print_nvidia_info_if_available()
        _warn_non_default(wall_time, allow_network)

    # Generate challenge — difficulty is the bit-width
    try:
        problem, verif, seed_used = breaking_rsa_challenge.generate_breaking_rsa(difficulty, difficulty, seed)
    except Exception as e:
        click.echo(f"Error generating challenge: {e}")
        sys.exit(1)

    challenge_id = str(uuid.uuid4())
    problem_json = problem.to_json()

    problem_summary = {
        "Difficulty: ": difficulty,
        "Problem:    ": problem_json,
    }

    # Workspace holds output + challenge_input_mount (Docker parity with validator)
    workspace = tempfile.mkdtemp(prefix="workbench-")
    output_dir = os.path.join(workspace, "output")
    os.makedirs(output_dir, exist_ok=True)

    build_result = None
    run_result = None

    try:
        if mode == "docker":
            build_result = build_image(solution, "breaking_rsa")
            if not build_result.success:
                print_report(
                    "breaking_rsa", mode, seed_used, problem_summary,
                    build_result=build_result,
                    total_time=time.time() - total_start,
                )
                sys.exit(1)

            mount_dir = _prepare_challenge_input_dir(workspace)
            _write_breaking_rsa_challenge_input(mount_dir, problem)
            run_result = run_container(
                "breaking_rsa", mount_dir, output_dir,
                timeout=wall_time, network=allow_network,
            )
        else:
            entry = find_entry_point(solution, "breaking_rsa")
            if not entry:
                click.echo(f"Error: No solver script found in {solution}. Expected breaking_rsa.py.")
                sys.exit(1)
            run_result = run_direct(entry, challenge_id, problem_json, output_dir, wall_time)

        # Validate
        validation_results = validate_output(
            output_dir, "breaking_rsa",
            check_dockerfile=(mode == "docker"),
            solution_dir=solution,
        )

        # Verify if validation passed
        verify_result = None
        schema_ok = all(c.passed for c in validation_results)
        if schema_ok:
            from qbittensor.challenges.breaking_rsa import Solution
            result_path = Path(os.path.join(output_dir, RESULT_JSON_FILENAME))
            sol = Solution.from_json_file(result_path)
            verify_result = verify_breaking_rsa(problem, sol, verif)

        total_time = time.time() - total_start
        success = print_report(
            "breaking_rsa", mode, seed_used, problem_summary,
            build_result=build_result, run_result=run_result,
            validation_results=validation_results,
            verify_result=verify_result,
            total_time=total_time,
        )

    finally:
        if keep_output:
            click.echo(f"Output kept at: {output_dir}")
            click.echo(f"Workspace kept at: {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)

    sys.exit(0 if success else 1)


@test.command("hardening-quantum-proof")
@click.option("--difficulty", default=1, help="Difficulty label (default: 1)")
@click.option("--solution", required=True, type=click.Path(exists=True), help="Path to solution directory")
@click.option("--mode", type=click.Choice(["docker", "direct"]), default="docker", help="Execution mode")
@click.option("--circuit", default=None, help="Sample circuit ID (default: random)")
@click.option("--wall-time", default=DEFAULT_WALL_TIME, help=f"Wall time in seconds (default: {DEFAULT_WALL_TIME} = {DEFAULT_WALL_TIME // 3600}h, matches validator)")
@click.option("--allow-network", is_flag=True, help="Allow network access in container (validator disables network)")
@click.option("--keep-output", is_flag=True, help="Keep output directory after test")
def test_hardening_quantum_proof(difficulty, solution, mode, circuit, wall_time, allow_network, keep_output):
    """Test a Hardening Quantum Proof solution."""
    total_start = time.time()

    if mode == "docker":
        if not check_docker():
            click.echo("Error: Docker is not available. Install Docker or use --mode direct.")
            sys.exit(1)
        _preflight_dockerfile(solution)
        _print_nvidia_info_if_available()
        _warn_non_default(wall_time, allow_network)

    try:
        problem, verif, circuit_id = hqp_challenge.load_sample_circuit(circuit, difficulty)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}")
        sys.exit(1)

    challenge_id = str(uuid.uuid4())
    problem_json = problem.to_json()

    problem_summary = {
        "Circuit:    ": circuit_id,
        "Difficulty: ": difficulty,
        "Qubits:     ": len(verif.peaked_state),
    }

    workspace = tempfile.mkdtemp(prefix="workbench-")
    output_dir = os.path.join(workspace, "output")
    os.makedirs(output_dir, exist_ok=True)

    build_result = None
    run_result = None

    try:
        if mode == "docker":
            build_result = build_image(solution, "hardening_quantum_proof")
            if not build_result.success:
                print_report(
                    "hardening_quantum_proof", mode, 0, problem_summary,
                    build_result=build_result,
                    total_time=time.time() - total_start,
                )
                sys.exit(1)

            mount_dir = _prepare_challenge_input_dir(workspace)
            _write_hqp_challenge_input(
                mount_dir, challenge_id, problem.difficulty, problem.qasm_file,
            )
            run_result = run_container(
                "hardening_quantum_proof", mount_dir, output_dir,
                timeout=wall_time, network=allow_network,
            )
        else:
            entry = find_entry_point(solution, "hardening_quantum_proof")
            if not entry:
                click.echo(f"Error: No solver script found in {solution}. Expected hardening_quantum_proof.py.")
                sys.exit(1)
            run_result = run_direct(entry, challenge_id, problem_json, output_dir, wall_time)

        validation_results = validate_output(
            output_dir, "hardening_quantum_proof",
            check_dockerfile=(mode == "docker"),
            solution_dir=solution,
        )

        verify_result = None
        schema_ok = all(c.passed for c in validation_results)
        if schema_ok:
            from qbittensor.challenges.hardening_quantum_proof import Solution
            result_path = Path(os.path.join(output_dir, RESULT_JSON_FILENAME))
            sol = Solution.from_json_file(result_path)
            verify_result = verify_hardening_quantum_proof(problem, sol, verif)

        total_time = time.time() - total_start
        success = print_report(
            "hardening_quantum_proof", mode, 0, problem_summary,
            build_result=build_result, run_result=run_result,
            validation_results=validation_results,
            verify_result=verify_result,
            total_time=total_time,
        )

    finally:
        if keep_output:
            click.echo(f"Output kept at: {output_dir}")
            click.echo(f"Workspace kept at: {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)

    sys.exit(0 if success else 1)


@test.command("mock")
@click.option("--difficulty", default=1, help="Difficulty label (default: 1)")
@click.option("--solution", required=True, type=click.Path(exists=True), help="Path to solution directory")
@click.option("--mode", type=click.Choice(["docker", "direct"]), default="docker", help="Execution mode")
@click.option("--private-key", default=None, help="Ed25519 private key (hex). Defaults to ENIGMA_MOCK_PRIVATE_KEY env var.")
@click.option("--public-key", default=None, help="Ed25519 public key (hex). Defaults to built-in key.")
@click.option("--wall-time", default=DEFAULT_WALL_TIME, help=f"Wall time in seconds (default: {DEFAULT_WALL_TIME} = {DEFAULT_WALL_TIME // 3600}h, matches validator)")
@click.option("--allow-network", is_flag=True, help="Allow network access in container (validator disables network)")
@click.option("--keep-output", is_flag=True, help="Keep output directory after test")
def test_mock(difficulty, solution, mode, private_key, public_key, wall_time, allow_network, keep_output):
    """Test the mock (plumbing test) challenge."""
    total_start = time.time()

    if mode == "docker":
        if not check_docker():
            click.echo("Error: Docker is not available. Install Docker or use --mode direct.")
            sys.exit(1)
        _preflight_dockerfile(solution)
        _print_nvidia_info_if_available()
        _warn_non_default(wall_time, allow_network)

    # Resolve private key
    priv_key = private_key or os.environ.get("ENIGMA_MOCK_PRIVATE_KEY")
    if not priv_key:
        click.echo(
            "Error: Private key required. Set ENIGMA_MOCK_PRIVATE_KEY env var "
            "or pass --private-key."
        )
        sys.exit(1)

    # Generate challenge (needs public key for verification)
    try:
        problem, verif = mock_challenge.generate_mock(difficulty, public_key_hex=public_key)
    except ValueError as e:
        click.echo(f"Error: {e}")
        sys.exit(1)

    challenge_id = str(uuid.uuid4())
    problem_json = problem.to_json()

    problem_summary = {
        "Difficulty: ": difficulty,
        "Public key: ": verif.public_key_hex[:16] + "...",
    }

    workspace = tempfile.mkdtemp(prefix="workbench-")
    output_dir = os.path.join(workspace, "output")
    os.makedirs(output_dir, exist_ok=True)

    build_result = None
    run_result = None

    try:
        if mode == "docker":
            build_result = build_image(solution, "mock")
            if not build_result.success:
                print_report(
                    "mock", mode, 0, problem_summary,
                    build_result=build_result,
                    total_time=time.time() - total_start,
                )
                sys.exit(1)

            mount_dir = _prepare_challenge_input_dir(workspace)
            _write_mock_challenge_input(mount_dir)
            run_result = run_container(
                "mock", mount_dir, output_dir,
                timeout=wall_time,
                env_vars={"ENIGMA_MOCK_PRIVATE_KEY": priv_key},
                network=allow_network,
            )
        else:
            entry = find_entry_point(solution, "mock")
            if not entry:
                click.echo(f"Error: No solver script found in {solution}. Expected mock_solver.py.")
                sys.exit(1)
            run_result = run_direct(entry, challenge_id, problem_json, output_dir, wall_time)

        validation_results = validate_output(
            output_dir, "mock",
            check_dockerfile=(mode == "docker"),
            solution_dir=solution,
        )

        verify_result = None
        schema_ok = all(c.passed for c in validation_results)
        if schema_ok:
            from qbittensor.challenges.mock_challenge import Solution
            result_path = Path(os.path.join(output_dir, RESULT_JSON_FILENAME))
            sol = Solution.from_json_file(result_path)
            verify_result = verify_mock(problem, sol, verif)

        total_time = time.time() - total_start
        success = print_report(
            "mock", mode, 0, problem_summary,
            build_result=build_result, run_result=run_result,
            validation_results=validation_results,
            verify_result=verify_result,
            total_time=total_time,
        )

    finally:
        if keep_output:
            click.echo(f"Output kept at: {output_dir}")
            click.echo(f"Workspace kept at: {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)

    sys.exit(0 if success else 1)


@cli.command()
def keygen():
    """Generate a new Ed25519 keypair for the mock challenge."""
    from qbittensor.challenges.mock_challenge import generate_keypair
    private_hex, public_hex = generate_keypair()
    click.echo(f"\nEd25519 Keypair Generated")
    click.echo(f"========================")
    click.echo(f"Private key: {private_hex}")
    click.echo(f"Public key:  {public_hex}")
    click.echo(f"\nOn the miner's machine:")
    click.echo(f"  export ENIGMA_MOCK_PRIVATE_KEY={private_hex}")
    click.echo(f"\nOn the validator's machine:")
    click.echo(f"  export ENIGMA_MOCK_PUBLIC_KEY={public_hex}")
    click.echo()


@cli.command()
def milestones():
    """Show challenge parameters and defaults."""
    click.echo(f"""
Enigma Challenge Parameters
============================

All challenges run in Docker with:
  --network none (no network access)
  --wall-time {DEFAULT_WALL_TIME}s ({DEFAULT_WALL_TIME // 3600}h max runtime, matches validator)

Use --allow-network and --wall-time to override for development.

breaking_rsa
  --difficulty   Bit-width of the semiprime to factor (default: 300)

hardening_quantum_proof
  --difficulty   Difficulty label (default: 1)
  --circuit      Sample circuit ID (default: random)

mock
  --difficulty   Difficulty label (default: 1)
  --private-key  Ed25519 private key hex (or ENIGMA_MOCK_PRIVATE_KEY env var)
  --public-key   Ed25519 public key hex (or built-in default)

Example solutions: workbench/challenges/*/example_solution/
""")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--challenge", required=True, type=click.Choice(["breaking_rsa", "hardening_quantum_proof", "mock"]), help="Challenge type")
def validate(path, challenge):
    """Validate output directory structure (no solver run)."""
    results = validate_output(path, challenge)

    click.echo("\n--- Structural Validation ---")
    all_passed = True
    for check in results:
        tag = "PASS" if check.passed else "FAIL"
        click.echo(f"[{tag}] {check.name}")
        if not check.passed and check.message:
            click.echo(f"       {check.message}")
            all_passed = False

    click.echo(f"\nResult: {'ALL CHECKS PASSED' if all_passed else 'CHECKS FAILED'}\n")
    sys.exit(0 if all_passed else 1)


@cli.command()
@click.option("--difficulty", default=None, type=int, help="Filter by difficulty level")
def samples(difficulty):
    """List available sample circuits for Hardening Quantum Proof."""
    sample_list = hqp_challenge.list_samples(difficulty=difficulty)
    if not sample_list:
        click.echo("No sample circuits found." + (f" (difficulty={difficulty})" if difficulty else ""))
        return

    click.echo(f"\n{'ID':<24} {'Diff':<6} {'Qubits':<8} {'Type'}")
    click.echo("-" * 60)
    for s in sample_list:
        click.echo(f"{s['id']:<24} {s['difficulty']:<6} {s['qubit_count']:<8}")
    click.echo()
