# Enigma Developer Workbench

Local testing for challenge solutions: generate a problem, run your solver,
validate output, and verify — without connecting to the subnet.

Test here in **Docker mode** before `mine-enigma`. Each upload costs TAO.
Direct mode skips Docker and does not match the validator. Submission
instructions: [Miner Guide](../qbittensor/miner/README.md).

Live challenges and prizes: [Enigma](https://www.qbittensorlabs.com/enigma).

## Setup

Python 3.12+ and Docker. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

That registers `enigma-workbench` (`python -m workbench` is the same CLI).

On Apple Silicon, Docker must build and run `--platform linux/amd64`.

## Usage

```bash
enigma-workbench test breaking-rsa \
    --solution workbench/challenges/breaking_rsa/example_solution/ \
    --difficulty 300

enigma-workbench test hardening-quantum-proof --difficulty 0 \
    --solution workbench/challenges/hardening_quantum_proof/example_solution/

enigma-workbench test mock \
    --solution workbench/challenges/mock_challenge/example_solution/
```

`test CHALLENGE` is the full pipeline (generate, build, run, validate, verify).
`run` is an alias. Slugs: `breaking-rsa`, `hardening-quantum-proof`, `mock`
(underscores work). `enigma-workbench challenges` lists them.

```bash
enigma-workbench build --solution ./my_solver/     # Docker image only
enigma-workbench test breaking-rsa --solution ./my_solver/ --mode direct
enigma-workbench challenges
enigma-workbench milestones
enigma-workbench samples [--difficulty 1]
enigma-workbench validate <output_dir> --challenge breaking-rsa
enigma-workbench keygen
```

Solver dirs you add under `workbench/challenges/<name>/` are gitignored
(except shipped `example_solution/` and `sample_circuits/`).

### Options

| Option | Description |
|--------|-------------|
| `--solution <path>` | Solution directory (`Dockerfile` for Docker mode) |
| `--mode docker\|direct` | `docker` (default) or local subprocess |
| `--difficulty <int>` | Meaning depends on the challenge |
| `--seed <int>` | Breaking RSA |
| `--circuit <id>` | Hardening Quantum Proof sample (default: random) |
| `--private-key` / `--public-key` | Mock keys (example has a baked-in key) |
| `--wall-time <secs>` | Max runtime (default 14400, matches validator) |
| `--allow-network` | Docker network (validator uses `--network none`) |
| `--keep-output` | Keep the output directory |

## How it works

1. **Generate** a challenge instance
2. **Build** (Docker mode) `docker build --platform linux/amd64` with the
   solution directory as context — same as the validator. If
   `enigma_challenges/` is missing, `test`/`build` copy `qbittensor/challenges`
   there so the directory you tested is what you zip.
3. **Run** the solver. Docker: read-only `/challenge_input/`, no CLI args, no
   output volume, read-only rootfs. Emit logs + separator + base64 zip on
   stdout. Direct mode: CLI args `<challenge_id> <problem_json>` and `OUTPUT_DIR`.
4. **Validate** required files (`result.json`, `container.log`, `solve_info.json`)
5. **Verify** against the known answer

Docker mode matches validator **security and I/O** (`--network none`, read-only
rootfs, tmpfs `/tmp`, non-root `miner`, Dockerfile policy, image size limit,
stdout protocol). It does **not** apply `--cpus 24` / `--memory 85g` / `--gpus`
unless you export `VALIDATOR_DOCKER_CPU_LIMIT`, `VALIDATOR_MEMORY_LIMIT`,
`VALIDATOR_DOCKER_GPUS`. A pass is required before submitting; it is not a
guarantee on validator hardware (RTX PRO 6000, 24 vCPU, 85 GB RAM).

## Solution layout and zip

A **submission** directory needs a `Dockerfile` at the zip root (or in a single
top-level folder) that only `COPY`s paths inside the zip. `test`/`build` copy
`qbittensor/challenges` into `enigma_challenges/` if it is missing. Zip that
same directory:

```bash
enigma-workbench test breaking-rsa --solution ./my_solver
( cd ./my_solver && zip -r ../my_solver.zip . -x '*.pyc' -x '*__pycache__*' )
```

Then `mine-enigma` — see the [Miner Guide](../qbittensor/miner/README.md).

Examples: `workbench/challenges/*/example_solution/`.

## Benchmarking

```bash
./workbench/benchmark_breaking_rsa.sh
```

Docker pipeline across bit sizes `300 335 340 … 380` (then +5), stops on first
failure, writes `benchmark_results.csv`. Default solution:
`workbench/challenges/breaking_rsa/example_solution`.

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHON` | auto-detect | Python 3.12+ |
| `SEED` | `42` | Random seed |
| `WALL_TIME` | `14400` | Seconds per run |
| `CSV_FILE` | `benchmark_results.csv` | Output path |
| `SOLUTIONS` | example_solution | Comma-separated solution dirs |

```bash
WALL_TIME=3600 ./workbench/benchmark_breaking_rsa.sh
```
