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

"""Workbench challenge registry.

CLI names are slugs (``breaking-rsa``). Underscores and directory names such as
``mock_challenge`` also resolve. Extra directories under ``workbench/challenges/``
are listed so ``enigma-workbench test <dir>`` can name them, but only modules
with a registered handler can actually run.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from workbench.challenges import breaking_rsa as breaking_rsa_challenge
from workbench.challenges import hardening_quantum_proof as hqp_challenge
from workbench.challenges import mock as mock_challenge
from workbench.verifier import (
    VerifyResult,
    verify_breaking_rsa,
    verify_hardening_quantum_proof,
    verify_mock,
)


def challenges_dir() -> Path:
    return Path(__file__).resolve().parent


def normalize_slug(name: str) -> str:
    return name.strip().lower().replace("_", "-")


@dataclass
class PreparedChallenge:
    type_id: str
    problem: Any
    verif: Any
    problem_json: str
    problem_summary: dict[str, Any]
    seed_used: int
    write_input: Callable[[str], None]
    load_solution: Callable[[Path], Any]
    verify: Callable[[Any, Any, Any], VerifyResult]
    entry_points: list[str]
    env_vars: dict[str, str] | None = None


@dataclass
class ChallengeSpec:
    slug: str
    type_id: str
    default_difficulty: int
    entry_points: list[str]
    prepare: Callable[..., PreparedChallenge]
    aliases: tuple[str, ...] = ()
    help: str = ""
    extra_flags: str = ""


def _write_breaking_rsa_input(mount_dir: str, problem) -> None:
    path = os.path.join(mount_dir, "challenge_input.json")
    with open(path, "w") as f:
        json.dump(problem.to_dict(), f)


def _write_hqp_input(mount_dir: str, challenge_id: str, difficulty: int, host_qasm_path: str) -> None:
    qasm_dest = os.path.join(mount_dir, "circuit.qasm")
    shutil.copy2(host_qasm_path, qasm_dest)
    path = os.path.join(mount_dir, "challenge_input.json")
    with open(path, "w") as f:
        json.dump(
            {
                "challenge_id": challenge_id,
                "difficulty": difficulty,
                "qasm_file": "/challenge_input/circuit.qasm",
            },
            f,
        )


def _write_mock_input(mount_dir: str) -> None:
    path = os.path.join(mount_dir, "challenge_input.txt")
    with open(path, "w") as f:
        f.write(
            "This is a simple solution setup file. "
            "The output should include the word 'Hello'. Hello!"
        )


def _prepare_breaking_rsa(difficulty: int, seed: int | None, **_kwargs) -> PreparedChallenge:
    from qbittensor.challenges.breaking_rsa import Solution

    problem, verif, seed_used = breaking_rsa_challenge.generate_breaking_rsa(
        difficulty, difficulty, seed
    )
    return PreparedChallenge(
        type_id="breaking_rsa",
        problem=problem,
        verif=verif,
        problem_json=problem.to_json(),
        problem_summary={"Difficulty: ": difficulty, "Problem:    ": problem.to_json()},
        seed_used=seed_used,
        write_input=lambda mount: _write_breaking_rsa_input(mount, problem),
        load_solution=Solution.from_json_file,
        verify=verify_breaking_rsa,
        entry_points=["breaking_rsa.py"],
    )


def _prepare_hqp(difficulty: int, circuit: str | None = None, **_kwargs) -> PreparedChallenge:
    from qbittensor.challenges.hardening_quantum_proof import Solution

    problem, verif, circuit_id = hqp_challenge.load_sample_circuit(circuit, difficulty)
    challenge_id = str(uuid.uuid4())
    return PreparedChallenge(
        type_id="hardening_quantum_proof",
        problem=problem,
        verif=verif,
        problem_json=problem.to_json(),
        problem_summary={
            "Circuit:    ": circuit_id,
            "Difficulty: ": difficulty,
            "Qubits:     ": len(verif.peaked_state),
        },
        seed_used=0,
        write_input=lambda mount: _write_hqp_input(
            mount, challenge_id, problem.difficulty, problem.qasm_file
        ),
        load_solution=Solution.from_json_file,
        verify=verify_hardening_quantum_proof,
        entry_points=["hardening_quantum_proof.py"],
    )


def _prepare_mock(
    difficulty: int,
    private_key: str | None = None,
    public_key: str | None = None,
    **_kwargs,
) -> PreparedChallenge:
    from qbittensor.challenges.mock_challenge import Solution

    priv_key = private_key or os.environ.get("ENIGMA_MOCK_PRIVATE_KEY")
    problem, verif = mock_challenge.generate_mock(difficulty, public_key_hex=public_key)
    env_vars = {"ENIGMA_MOCK_PRIVATE_KEY": priv_key} if priv_key else None
    return PreparedChallenge(
        type_id="mock",
        problem=problem,
        verif=verif,
        problem_json=problem.to_json(),
        problem_summary={
            "Difficulty: ": difficulty,
            "Public key: ": verif.public_key_hex[:16] + "...",
        },
        seed_used=0,
        write_input=_write_mock_input,
        load_solution=Solution.from_json_file,
        verify=verify_mock,
        entry_points=["mock_solution.py"],
        env_vars=env_vars,
    )


SPECS: tuple[ChallengeSpec, ...] = (
    ChallengeSpec(
        slug="breaking-rsa",
        type_id="breaking_rsa",
        default_difficulty=300,
        entry_points=["breaking_rsa.py"],
        prepare=_prepare_breaking_rsa,
        aliases=("breaking_rsa",),
        help="Factor a random semiprime.",
        extra_flags="--difficulty (bit-width, default 300), --seed",
    ),
    ChallengeSpec(
        slug="hardening-quantum-proof",
        type_id="hardening_quantum_proof",
        default_difficulty=1,
        entry_points=["hardening_quantum_proof.py"],
        prepare=_prepare_hqp,
        aliases=("hardening_quantum_proof", "hqp"),
        help="Find the peaked state of a sample circuit.",
        extra_flags="--difficulty (default 1), --circuit",
    ),
    ChallengeSpec(
        slug="mock",
        type_id="mock",
        default_difficulty=1,
        entry_points=["mock_solution.py"],
        prepare=_prepare_mock,
        aliases=("mock_challenge", "mock-challenge"),
        help="Plumbing test (Ed25519 signature).",
        extra_flags="--private-key / --public-key (optional; example uses a baked-in key)",
    ),
)


def _spec_index() -> dict[str, ChallengeSpec]:
    index: dict[str, ChallengeSpec] = {}
    for spec in SPECS:
        for key in (spec.slug, spec.type_id, *spec.aliases):
            index[normalize_slug(key)] = spec
    return index


def get_spec(name: str) -> ChallengeSpec | None:
    return _spec_index().get(normalize_slug(name))


def shipped_slugs() -> list[str]:
    return [spec.slug for spec in SPECS]


def extra_challenge_names() -> list[str]:
    """Directory / module names under workbench/challenges/ that are not shipped."""
    known = {spec.type_id for spec in SPECS}
    known.update(normalize_slug(a).replace("-", "_") for spec in SPECS for a in spec.aliases)
    known.update({"mock_challenge"})
    extra: list[str] = []
    root = challenges_dir()
    for path in sorted(root.iterdir()):
        if path.name.startswith("_") or path.name == "__pycache__":
            continue
        if path.suffix == ".py":
            stem = path.stem
            if stem in {"registry"}:
                continue
            if stem not in known and normalize_slug(stem) not in {s.slug for s in SPECS}:
                extra.append(stem)
        elif path.is_dir() and path.name not in known:
            extra.append(path.name)
    return extra


def available_slugs() -> list[str]:
    names = shipped_slugs()
    names.extend(normalize_slug(n) for n in extra_challenge_names())
    # unique, shipped first
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered
