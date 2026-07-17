<div align="center">

<img src="./logo.png"/>

# **Enigma** (SN 63) <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/1395424987816661103)](https://discord.gg/xJ9JKPMJQD)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Decentralized Challenge Platform on Bittensor

[Discord](https://discord.gg/xJ9JKPMJQD) • [Network](https://taostats.io/subnets/63) • [Website](https://www.qbittensorlabs.com/enigma) • [Treasury](https://www.qbittensorlabs.com/enigma/treasury) • [GitHub](https://github.com/qbittensor-labs/enigma)

</div>

---

**Enigma** is Subnet 63 on Bittensor — a decentralized platform for pressure-testing foundational technologies through open challenges with real prize pools. It incentivizes the global community (researchers, hackers, engineers, and students) to break cryptographic systems, AI safeguards, post-quantum protocols, and other deep-tech targets.

By leveraging Bittensor's incentive layer, Enigma turns critical security research into a transparent, competitive, and publicly verifiable process. Winners drain prize pools, and successful solutions are open-sourced to advance the ecosystem. None of this works without Bittensor: network emissions fund the prize pools, validators verify solutions and govern payouts, and the permissionless design brings solvers in from inside and outside the ecosystem.

> **Current Focus**: Breaking today to build a better tomorrow. Two challenges — Breaking RSA and Hardening Quantum Proof — are live, each developed and launched in collaboration with a high-profile partner with deep expertise in the field. The subnet pioneers Bittensor's Treasury Wallet feature to fund challenge prize pools.

---

## How Enigma Works

1. **Challenges are posted** with funded prize pools in SN63 Alpha. Each challenge is developed and launched in collaboration with a high-profile collaborator with expertise in the field — Terra Quantum for Breaking RSA, BlueQubit for Hardening Quantum Proof.
2. **Participants compete** by submitting verified solutions (source code, not just answers). Unlimited resubmissions allowed; each submission carries a small TAO fee. See [Miner Setup](#miner-setup) for how to submit.
3. **Validators** score and verify solutions. A dedicated mechanism directs *all* miner emissions into the Treasury Wallet so they can accumulate into prize pools. Validators later vote on any disbursement.
4. **Winner takes all** — the first valid solution drains the prize. Proof is on-chain.
5. **Code is published** as open source after verification, becoming the starting point for the next attempt.
6. **The next milestone is harder**, and the cycle repeats. Prizes start at a baseline and grow the longer a milestone goes unsolved.

## Incentive Mechanism (Important)

Enigma is a **bounty-based challenge subnet**, not a continuous production subnet.

- All miner emissions are directed to the **Treasury Wallet** (a smart contract).
- These emissions accumulate as SN63 Alpha and form the prize pools for challenges.
- Miners do **not** receive ongoing emissions based on weight. Instead, they compete for large, discrete prizes when they solve challenges.
- Payouts from the Treasury require a formal proposal + vote by validators (60% success threshold, 50% quorum, timelock). The team cannot unilaterally withdraw funds.

This is intentional. It allows the subnet to fund meaningful research bounties (e.g. RSA factoring milestones) while keeping the funds under transparent, on-chain governance.

---

## Treasury Wallet

- **Details**

  - Vault Contract Address: `0xB291C87759E2BAf678734C45A44121091d999220`
  - Vault SS58 Cold Key: `5EgP27pkachXDvWpYGfFjatQkWkGXfaoGkP35jdXC4xwPmtZ`
  - Vault SS58 Hot Key: `5DCLafsAKaLeZwm9hjMHvrQNjtucSwBhKyTLYnYmMvhxF2Uc`
  - Governor Contract Address: `0x41a1BE0a7408717877DE25e2c62c2Fb71a04D8A9`
  - Details (verify with `treasury/scripts/list_proposals.py`):
    ```
    ====================================================================================================
    🔍 Contract Configuration:
      Name:                  Enigma-Treasury-v1.1-20260505
      Target NetUID:         63
      Treasury Admin:        0xa5ACB66F2e1e5307cd536F7fd346b0301b7bC0Ca
      TAO Limit:             1,000.0000 TAO
      Alpha Limit:           25,000.0000 Alpha
      ERC20 Limit:           10,000.0000 Tokens
      Limit Reset Period:    172800 seconds (~2 days 0 hrs)
      Success Threshold:     6000 BPS (60.0%)
      Quorum:                5000 BPS (50.0%)
      Proposal Expiration:   14400 blocks (~2 days 0 hrs)
      Voting Delay:          900 blocks (~3 hrs 0 mins)
      Voting Period:         21600 blocks (~3 days 0 hrs)
      Timelock Delay:        86400 seconds (~1 days 0 hrs)
    ====================================================================================================
    ```

The Treasury Wallet is a core component of Subnet 63, implemented as a smart contract on the EVM layer of Bittensor. It serves as the primary funding mechanism for challenges and ecosystem development. Custody is trustless end to end — every rule is enforced on-chain and every action is publicly auditable — making Enigma's prize pools among the most protected funds on Bittensor.

- **Funding**: All miner emissions are directed to the Treasury Wallet, accumulating SN63 Alpha tokens that form the challenge prize pools.
- **Governance**: Managed through a Governor contract with timelock delays, voting periods, and quorum requirements to ensure secure fund management. The Treasury Admin key can only create and cancel proposals, never approve them — no funds can leave the wallet without validator approval.
- **Voting Power**: Votes are weighted by Bittensor's native Voting Power, a chain feature designed in collaboration between the Opentensor Foundation and the Church of Rao and deployed in support of treasury wallets.
- **Battle-tested**: The treasury system itself was Enigma's inaugural challenge target — a funded bounty on breaking it that nobody claimed (see [Breaking Treasury Wallets](#breaking-treasury-wallets--closed) below).
- **Transparency**: Contract configuration, balances, proposals, votes, and the full payout transaction history are collated at [qbittensorlabs.com/enigma/treasury](https://www.qbittensorlabs.com/enigma/treasury).
- **Technical Details**: The full contract design, source, and the exact test and deployment tooling we used are published in [treasury/](treasury/) — so anyone can audit what's running at the addresses above, and any subnet can replicate the setup.

Challenges and prizes will be paid to the winner based upon a proposal and vote by validators.

For current treasury balance and active prize pools, visit the [Enigma page](https://www.qbittensorlabs.com/enigma).

---

## Challenges

### Breaking RSA — Live

Developed and launched in collaboration with [Terra Quantum](https://terraquantum.swiss/news/breaking-rsa-challenge-cryptanalysis/), a Swiss quantum technology company specializing in post-quantum readiness and cryptographic risk assessment.

Factor large semiprimes to break RSA encryption. Solutions run in Docker containers on validator hardware (NVIDIA RTX PRO 6000, 24 vCPU, 85 GB RAM, `--network none`, `linux/amd64`) with a 4-hour wall time.

See the [Breaking RSA challenge README](workbench/challenges/breaking_rsa/README.md) for details, and the [Miner Guide](qbittensor/miner/README.md) for submission instructions.

### Hardening Quantum Proof — Live

Developed and launched in collaboration with [BlueQubit](https://www.bluequbit.io), a quantum software company and one of the leading research groups on peaked circuits.

Given a quantum circuit, find the peaked state — the output with a disproportionately high measurement probability. Peaked circuits can serve as quantum proofs: verifiable tests that a real quantum computer can solve but a classical system cannot. If classical solvers can crack them, the proof doesn't hold.

See the [Hardening Quantum Proof challenge README](workbench/challenges/hardening_quantum_proof/README.md) for details, and the [Miner Guide](qbittensor/miner/README.md) for submission instructions.

### Breaking Treasury Wallets — Closed

Enigma's inaugural challenge: a bounty on breaking the treasury wallet system that now custodies the prize pools. Nobody broke it, and the system was validated into production (closed May 21, 2026).

### Results So Far

Enigma's first challenges opened in early June 2026. One month in:

- Breaking RSA has climbed from 340-bit to 480-bit keys. RSA-340 fell overnight. The next milestone jumped 120 bits to RSA-460, widely called impossible on a single node in four hours — then a miner cracked it in 3.9 hours with a GPU lattice siever built on the General Number Field Sieve, the same family of methods behind the 829-bit factoring record. RSA-480 fell on July 2. RSA-500 is open now.
- Hardening Quantum Proof Level 1 fell to a matrix product state simulator driven by canonical beam search. The Level 2 winner started from that published code and added an "unswap" technique that sheds bond dimension without destroying the peak; the source comments credit the Level 1 solution.
- When a vulnerability turned up in one challenge environment, the community reported it rather than exploiting it, and it was patched before anyone abused it.

RSA-460 was assumed infeasible on one node in four hours — the AI models we asked said no outright. A miner did it in 3.9. Finding those gaps between assumed and actual limits is the point, and the fixed one-node, four-hour environment is what makes results comparable across milestones. The winning methods scale with hardware and time.

For current prize pools, milestones, and additional details, visit the [Enigma page](https://www.qbittensorlabs.com/enigma).

---

## Voting Setup

To participate in Treasury Governance and cast votes on proposals, validators must follow the [Validator Onboarding Guide](VALIDATORS.md) which provides setup instructions for an EVM wallet.

- [Validator Voting Guide](VOTING.md)

---

## Validator Setup

For complete validator operator instructions (environment variables, Docker requirements, local database, high-level operation, etc.) see the dedicated guide:

**→ [qbittensor/validator/README.md](qbittensor/validator/README.md)**

### Quick Launch

#### With PM2 (recommended)

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Clone the repository
git clone https://github.com/qbittensor-labs/enigma.git
cd enigma
pip install -e .

# 3. Run the validator with PM2
pm2 start --interpreter .venv/bin/python --name enigma-validator neurons/validator.py -- --netuid 63 --logging.info --wallet.name <your_wallet_name> --wallet.hotkey <your_hotkey>
```

**Note**: Replace `<your_wallet_name>` and `<your_hotkey>` with your Bittensor wallet details (defaults to 'default' if not specified). For localnet testing, add `--subtensor.network local`.

#### Running Without PM2

```bash
python neurons/validator.py --netuid 63 --logging.info --wallet.name <your_wallet_name> --wallet.hotkey <your_hotkey>
```

### GPU Device

To bind the validator to a specific GPU, use the `--neuron.device` flag (see the validator README for details).

## Miner Setup

See the dedicated miner operator guide:

**→ [qbittensor/miner/README.md](qbittensor/miner/README.md)**

## Minimum Compute Requirements

**Validator** (high requirements):
- GPU: RTX PRO 6000 96 GB VRAM
- CPU: 26 cores at 2.5 GHz+
- RAM: 96 GB

**Miner** (lightweight):
- GPU: Not required
- CPU: 1 core minimum at 2.0 GHz+
- RAM: 8 GB minimum

Full details (including storage, OS, and network recommendations) are in [`min_compute.yml`](min_compute.yml).

## Development

For contributor setup, development workflow, testing, and linting instructions (including the required `pip install -e .` step and `pip install -r requirements-dev.txt`), see [CONTRIBUTING.md](CONTRIBUTING.md).
