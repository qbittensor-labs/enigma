# Zip layout (build context = directory that contains this Dockerfile):
#
#   Dockerfile
#   mock_solution.py
#   enigma_challenges/   (vendored by workbench / the submit helper)
#
# ``mock_solution.py`` imports ``enigma_challenges.solution_output``. The
# build context must include that package; this Dockerfile copies it in.
#
# Build:
#   docker build -t enigma-mock-example .
#
# Run:
#   docker run --rm enigma-mock-example > out.bin
#
# Output contract: the miner writes text logs to stdout, then the
# ``SOLUTION_OUTPUT_SEPARATOR`` line, then a base64-encoded zip of the
# solution artifacts. No volumes are mounted by the validator — the
# validator captures stdout via ``docker logs`` after the container
# exits and splits it into logs and the solution-output zip on its own host
# workspace. The validator's read-only-rootfs + ``--tmpfs /tmp`` flags still
# apply, so the miner can use ``/tmp`` for scratch space but writes its
# final output exclusively to stdout.

FROM python:3.12-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir "cryptography>=42.0.0" \
    && useradd -r -s /sbin/nologin -d /app -U miner

COPY --chown=miner:miner enigma_challenges /app/enigma_challenges/
COPY --chown=miner:miner mock_solution.py /app/mock_solution.py

USER miner
ENTRYPOINT ["python3", "/app/mock_solution.py"]
