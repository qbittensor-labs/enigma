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
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import click

from workbench.challenges import hardening_quantum_proof as hqp_challenge
from workbench.challenges.registry import (
    ChallengeSpec,
    extra_challenge_names,
    get_spec,
    shipped_slugs,
)
from workbench.runner.docker_runner import (
    DEFAULT_WALL_TIME,
    build_image,
    check_docker,
    ensure_enigma_challenges,
    run_container,
)

# Best-effort host NVIDIA/CUDA version for local developer visibility
# (validators always report this via telemetry; workbench is the local analog).
try:
    from qbittensor.utils.services.telemetry import _get_nvidia_driver_info
except Exception:
    _get_nvidia_driver_info = None  # type: ignore
from workbench.runner.direct_runner import find_entry_point, run_direct
from workbench.validator import validate_output, validate_dockerfile_security
from workbench.report import print_report

from qbittensor.challenges.solution_output import RESULT_JSON_FILENAME


def _prepare_challenge_input_dir(workspace: str) -> str:
    """Create a fresh host dir for the /challenge_input mount (validator parity)."""
    mount_dir = os.path.join(workspace, "challenge_input_mount")
    os.makedirs(mount_dir, mode=0o755, exist_ok=True)
    return mount_dir


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


def _ensure_challenges(solution: str) -> None:
    """Copy qbittensor/challenges into solution/enigma_challenges if missing."""
    try:
        dest, copied = ensure_enigma_challenges(solution)
    except ValueError as e:
        click.echo(f"Error: {e}")
        sys.exit(1)
    if copied:
        click.echo(f"Copied qbittensor/challenges -> {dest}")


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


class WorkbenchGroup(click.Group):
    """``run`` is an alias of ``test``."""

    def get_command(self, ctx, cmd_name):
        if cmd_name == "run":
            cmd_name = "test"
        return super().get_command(ctx, cmd_name)

    def list_commands(self, ctx):
        names = super().list_commands(ctx)
        if "test" in names and "run" not in names:
            names = list(names)
            names.insert(names.index("test") + 1, "run")
        return names


@click.group(cls=WorkbenchGroup)
def cli():
    """Enigma Developer Workbench -- local testing tool for challenge solutions."""
    pass


def _resolve_spec(name: str) -> ChallengeSpec:
    spec = get_spec(name)
    if spec is not None:
        return spec
    extras = extra_challenge_names()
    extra_note = ""
    if extras:
        extra_note = (
            f"\nAlso found on disk (no workbench handler yet): {', '.join(extras)}"
        )
    raise click.UsageError(
        f"Unknown challenge {name!r}. "
        f"Shipped: {', '.join(shipped_slugs())}."
        f"{extra_note}\n"
        "See `enigma-workbench challenges`."
    )


def _docker_preflight(solution: str, wall_time, allow_network) -> None:
    if not check_docker():
        click.echo("Error: Docker is not available. Install Docker or use --mode direct.")
        sys.exit(1)
    _preflight_dockerfile(solution)
    _print_nvidia_info_if_available()
    _warn_non_default(wall_time, allow_network)


@cli.command("test")
@click.argument("challenge")
@click.option("--solution", required=True, type=click.Path(exists=True), help="Path to solution directory")
@click.option("--mode", type=click.Choice(["docker", "direct"]), default="docker", help="Execution mode")
@click.option("--difficulty", type=int, default=None, help="Challenge difficulty (meaning varies by challenge)")
@click.option("--seed", type=int, default=None, help="Random seed (Breaking RSA)")
@click.option("--circuit", default=None, help="Sample circuit ID (Hardening Quantum Proof)")
@click.option("--private-key", default=None, help="Ed25519 private key hex (mock)")
@click.option("--public-key", default=None, help="Ed25519 public key hex (mock)")
@click.option("--wall-time", default=DEFAULT_WALL_TIME, help=f"Wall time in seconds (default: {DEFAULT_WALL_TIME} = {DEFAULT_WALL_TIME // 3600}h, matches validator)")
@click.option("--allow-network", is_flag=True, help="Allow network access in container (validator disables network)")
@click.option("--keep-output", is_flag=True, help="Keep output directory after test")
def test_cmd(
    challenge, solution, mode, difficulty, seed, circuit, private_key, public_key,
    wall_time, allow_network, keep_output,
):
    """Generate a challenge, build/run the solver, validate and verify.

    CHALLENGE is a slug from workbench/challenges/ (breaking-rsa,
    hardening-quantum-proof, mock). Underscores and aliases work too.

    Alias: enigma-workbench run CHALLENGE ...
    """
    spec = _resolve_spec(challenge)
    if difficulty is None:
        difficulty = spec.default_difficulty

    _ensure_challenges(solution)
    total_start = time.time()
    if mode == "docker":
        _docker_preflight(solution, wall_time, allow_network)

    try:
        prepared = spec.prepare(
            difficulty=difficulty,
            seed=seed,
            circuit=circuit,
            private_key=private_key,
            public_key=public_key,
        )
    except (ValueError, FileNotFoundError) as e:
        click.echo(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error generating challenge: {e}")
        sys.exit(1)

    challenge_id = str(uuid.uuid4())
    workspace = tempfile.mkdtemp(prefix="workbench-")
    output_dir = os.path.join(workspace, "output")
    os.makedirs(output_dir, exist_ok=True)

    build_result = None
    run_result = None
    success = False

    try:
        if mode == "docker":
            build_result = build_image(solution, prepared.type_id)
            if not build_result.success:
                print_report(
                    prepared.type_id, mode, prepared.seed_used, prepared.problem_summary,
                    build_result=build_result,
                    total_time=time.time() - total_start,
                )
                sys.exit(1)

            mount_dir = _prepare_challenge_input_dir(workspace)
            prepared.write_input(mount_dir)
            run_result = run_container(
                prepared.type_id, mount_dir, output_dir,
                timeout=wall_time,
                env_vars=prepared.env_vars,
                network=allow_network,
            )
        else:
            entry = find_entry_point(
                solution, prepared.type_id, names=prepared.entry_points,
            )
            if not entry:
                expected = " or ".join(prepared.entry_points)
                click.echo(f"Error: No solver script found in {solution}. Expected {expected}.")
                sys.exit(1)
            run_result = run_direct(
                entry, challenge_id, prepared.problem_json, output_dir, wall_time,
            )

        validation_results = validate_output(
            output_dir, prepared.type_id,
            check_dockerfile=(mode == "docker"),
            solution_dir=solution,
        )

        verify_result = None
        schema_ok = all(c.passed for c in validation_results)
        if schema_ok:
            result_path = Path(os.path.join(output_dir, RESULT_JSON_FILENAME))
            sol = prepared.load_solution(result_path)
            verify_result = prepared.verify(prepared.problem, sol, prepared.verif)

        success = print_report(
            prepared.type_id, mode, prepared.seed_used, prepared.problem_summary,
            build_result=build_result, run_result=run_result,
            validation_results=validation_results,
            verify_result=verify_result,
            total_time=time.time() - total_start,
        )
    finally:
        if keep_output:
            click.echo(f"Output kept at: {output_dir}")
            click.echo(f"Workspace kept at: {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)

    sys.exit(0 if success else 1)


@cli.command("build")
@click.option("--solution", required=True, type=click.Path(exists=True), help="Path to solution directory")
@click.option("--challenge", default=None, help="Challenge slug for the image tag (default: directory name)")
def build_cmd(solution, challenge):
    """Docker-build a solution image without generating or running a challenge.

    Use this to iterate on the Dockerfile. Full validator-shaped checking is
    still `enigma-workbench test CHALLENGE --solution ...`.
    """
    _ensure_challenges(solution)
    _docker_preflight(solution, DEFAULT_WALL_TIME, False)
    if challenge:
        type_id = _resolve_spec(challenge).type_id
    else:
        type_id = Path(solution).resolve().name.replace("-", "_")
    result = build_image(solution, type_id)
    click.echo(result.log)
    if result.success:
        click.echo(f"\nBuilt workbench-test-{type_id} in {result.duration:.1f}s")
        sys.exit(0)
    click.echo(f"\nBuild failed (exit {result.exit_code}) after {result.duration:.1f}s")
    sys.exit(1)


@cli.command("challenges")
def challenges_cmd():
    """List challenge slugs `test` / `run` accept."""
    click.echo("Shipped:")
    for spec in (get_spec(s) for s in shipped_slugs()):
        assert spec is not None
        click.echo(f"  {spec.slug:<28} {spec.help}")
        if spec.extra_flags:
            click.echo(f"  {'':<28} {spec.extra_flags}")
    extras = extra_challenge_names()
    if extras:
        click.echo("On disk, no handler:")
        for name in extras:
            click.echo(f"  {name}")
    click.echo()


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
    lines = [
        "Enigma Challenge Parameters",
        "============================",
        "",
        "All challenges run in Docker with:",
        "  --network none (no network access)",
        f"  --wall-time {DEFAULT_WALL_TIME}s ({DEFAULT_WALL_TIME // 3600}h max runtime, matches validator)",
        "",
        "Use --allow-network and --wall-time to override for development.",
        "",
    ]
    for slug in shipped_slugs():
        spec = get_spec(slug)
        assert spec is not None
        lines.append(spec.slug)
        if spec.extra_flags:
            lines.append(f"  {spec.extra_flags}")
        lines.append("")
    lines.append("Example solutions: workbench/challenges/*/example_solution/")
    click.echo("\n" + "\n".join(lines) + "\n")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--challenge", required=True, help="Challenge slug or type id")
def validate(path, challenge):
    """Validate output directory structure (no solver run)."""
    spec = get_spec(challenge)
    type_id = spec.type_id if spec else challenge
    results = validate_output(path, type_id)

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
