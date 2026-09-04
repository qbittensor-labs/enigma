FROM python:3.12-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir "cryptography>=42.0.0" \
    && useradd -r -s /sbin/nologin -d /app -U miner

COPY --chown=miner:miner enigma_challenges /app/enigma_challenges/
COPY --chown=miner:miner mock_solution.py /app/mock_solution.py

USER miner
ENTRYPOINT ["python3", "/app/mock_solution.py"]
