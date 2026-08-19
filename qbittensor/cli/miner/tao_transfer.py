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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import bittensor as bt
import click
from bittensor.wallet import Keypair
from rich.console import Console
from rich.text import Text

from qbittensor.utils.transfer_proof import TRANSFER_DEST_SS58

# Version string embedded in the on-chain remark for this proof format.
FEE_BINDING_REMARK_VERSION = "enigma/fee-binding/v1"


@dataclass(frozen=True)
class TransferProofTx:
    """On-chain transfer identifiers after inclusion (required to verify the extrinsic)."""

    extrinsic_hash: str
    block_hash: str


def transfer_proof_tx_from_receipt(response: Any) -> TransferProofTx:
    """Extract extrinsic hash and inclusion block hash from a submit_extrinsic receipt."""
    ex_hash = getattr(response, "extrinsic_hash", None)
    block_hash = getattr(response, "block_hash", None)
    if isinstance(response, dict):
        ex_hash = ex_hash or response.get("extrinsic_hash")
        block_hash = block_hash or response.get("block_hash")
    if not ex_hash or not block_hash:
        raise ValueError(
            "Transfer submitted but extrinsic_hash and block_hash were not both returned."
        )
    return TransferProofTx(extrinsic_hash=str(ex_hash), block_hash=str(block_hash))


def build_fee_binding_remark(
    *,
    miner_hotkey: str,
    milestone_id: str,
    upload_endpoint_id: str,
    amount_rao: str,
) -> bytes:
    """
    Build the canonical remark payload that will be included in the on-chain
    Utility.batch_all transaction.

    This data is signed by the fee-paying coldkey at payment time and becomes
    the authoritative on-chain binding between the payment and the specific
    submission.
    """
    lines = [
        FEE_BINDING_REMARK_VERSION,
        f"miner_hotkey:{miner_hotkey}",
        f"milestone_id:{milestone_id}",
        f"upload_endpoint_id:{upload_endpoint_id}",
        f"amount_rao:{amount_rao}",
    ]
    return "\n".join(lines).encode("utf-8")


def _extrinsic_hash_from_block(subtensor: bt.Subtensor, block_hash: str, extrinsic_id: str) -> str:
    """Resolve blake2 extrinsic hash from ``block_number-index`` via SubstrateInterface."""
    from async_substrate_interface.sync_substrate import SubstrateInterface

    try:
        idx = int(str(extrinsic_id).split("-")[-1])
    except ValueError as e:
        raise ValueError(f"unrecognized extrinsic_id {extrinsic_id!r}") from e

    endpoint = getattr(subtensor, "endpoint", None) or "ws://127.0.0.1:9944"
    substrate = getattr(subtensor, "substrate", None) or SubstrateInterface(url=endpoint)
    block = substrate.get_block(block_hash=block_hash)
    extrinsics = block.get("extrinsics") if isinstance(block, dict) else None
    if not isinstance(extrinsics, list) or not (0 <= idx < len(extrinsics)):
        raise ValueError(f"extrinsic index {idx} not in block {block_hash}")
    ex = extrinsics[idx]
    value = getattr(ex, "value", None) or {}
    h = value.get("extrinsic_hash") if isinstance(value, dict) else None
    if not h:
        raw = getattr(ex, "extrinsic_hash", None)
        if isinstance(raw, (bytes, bytearray)):
            h = "0x" + raw.hex()
    if not h:
        raise ValueError(f"could not read extrinsic_hash for {extrinsic_id} in {block_hash}")
    return str(h)


def transfer_fee_extrinsic_subtensor(
    *,
    subtensor: bt.Subtensor,
    source_ss58: str,
    keypair: Keypair,
    fee_tao: float,
    miner_hotkey: str,
    milestone_id: str,
    upload_endpoint_id: str,
    wallet: Optional[Any] = None,
) -> TransferProofTx:
    """
    Submit the fee payment as a Utility.batch_all containing:
      - Balances.transfer_keep_alive
      - System.remark_with_event (the canonical binding data)

    The provided Keypair (from the fee coldkey) signs the entire batch.

    fee_tao MUST come from the Challenges API.
    """
    if fee_tao is None or fee_tao <= 0:
        raise ValueError("fee_tao is required and must be obtained from the Challenges API")

    if keypair.ss58_address != source_ss58:
        raise ValueError("Source SS58 does not match the provided keypair's address.")

    amount_rao = str(int(bt.Balance.from_tao(fee_tao).rao))

    remark_data = build_fee_binding_remark(
        miner_hotkey=miner_hotkey,
        milestone_id=milestone_id,
        upload_endpoint_id=upload_endpoint_id,
        amount_rao=amount_rao,
    )

    # Bittensor SDK v11: compose/submit_call (no subtensor.substrate).
    # Prefer the legacy substrate path when tests inject a mock .substrate.
    if getattr(subtensor, "substrate", None) is None and hasattr(subtensor, "submit_call") and hasattr(bt, "calls"):
        if wallet is None:
            raise ValueError("wallet is required to submit the v11 fee-payment batch")
        transfer_call = subtensor.compose(
            bt.calls.Balances.transfer_keep_alive(
                dest=TRANSFER_DEST_SS58,
                value=int(amount_rao),
            )
        )
        remark_call = subtensor.compose(
            bt.calls.System.remark_with_event(remark=list(remark_data))
        )
        batch_call = bt.calls.Utility.batch_all(calls=[transfer_call, remark_call])
        result = subtensor.submit_call(batch_call, wallet)
        if not getattr(result, "success", False):
            raise ValueError(
                f"Fee payment batch failed: {getattr(result, 'error', None) or getattr(result, 'message', result)}"
            )
        block_hash = getattr(result, "block_hash", None)
        extrinsic_id = getattr(result, "extrinsic_id", None)
        if not block_hash or not extrinsic_id:
            raise ValueError("Fee payment succeeded but block_hash/extrinsic_id were not returned.")
        ex_hash = _extrinsic_hash_from_block(subtensor, str(block_hash), str(extrinsic_id))
        return TransferProofTx(extrinsic_hash=ex_hash, block_hash=str(block_hash))

    # Pre-v11 substrate path (kept for unit tests that mock .substrate).
    transfer_call = subtensor.substrate.compose_call(
        call_module="Balances",
        call_function="transfer_keep_alive",
        call_params={
            "dest": TRANSFER_DEST_SS58,
            "value": int(amount_rao),
        },
    )

    remark_call = subtensor.substrate.compose_call(
        call_module="System",
        call_function="remark_with_event",
        call_params={
            "remark": remark_data,
        },
    )

    # Batch them atomically. The coldkey signs the whole batch.
    batch_call = subtensor.substrate.compose_call(
        call_module="Utility",
        call_function="batch_all",
        call_params={
            "calls": [transfer_call, remark_call],
        },
    )

    extrinsic = subtensor.substrate.create_signed_extrinsic(
        call=batch_call,
        keypair=keypair,
    )

    response = subtensor.substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )

    if not response.is_success:
        raise ValueError(f"Fee payment batch failed: {response.error_message}")

    return transfer_proof_tx_from_receipt(response)


def transfer_tao_for_submission(
    *,
    console: Console,
    source_ss58: str,
    keypair: Keypair,
    network: str,
    fee_tao: float,
    miner_hotkey: str,
    milestone_id: str,
    upload_endpoint_id: str,
    wallet: Optional[Any] = None,
) -> TransferProofTx:
    """
    High-level helper used by the miner CLI.

    Performs the on-chain fee payment:
    - Batch containing transfer + remark
    - Signed by the fee coldkey (via the provided Keypair)

    All binding data is embedded in the on-chain remark at payment time.
    """
    if not milestone_id:
        raise click.ClickException("milestone_id is required")

    if not upload_endpoint_id:
        raise click.ClickException("upload_endpoint_id is required for the new proof format")

    if fee_tao is None or fee_tao <= 0:
        raise click.ClickException("fee_tao is required and must be greater than 0")

    console.print(
        Text(
            f"Using API priceTao={fee_tao} for milestone {milestone_id}",
            style="dim cyan",
        )
    )

    try:
        with bt.Subtensor(network=network) as subtensor:
            proof_tx = transfer_fee_extrinsic_subtensor(
                subtensor=subtensor,
                source_ss58=source_ss58,
                keypair=keypair,
                fee_tao=fee_tao,
                miner_hotkey=miner_hotkey,
                milestone_id=milestone_id,
                upload_endpoint_id=upload_endpoint_id,
                wallet=wallet,
            )
    except ValueError as e:
        raise click.ClickException(f"Fee payment transaction failed: {e}") from e
    except Exception as e:
        raise click.ClickException(f"Fee payment transaction failed: {e}") from e

    console.print(
        Text(
            f"Proof tx: {proof_tx.extrinsic_hash} (block {proof_tx.block_hash})",
            style="bold green",
        )
    )
    return proof_tx


def ensure_sufficient_balance_for_fee(
    source_ss58: str,
    network: str,
    fee_tao: float,
    buffer_tao: float = 0.0005,
) -> None:
    """Query the on-chain free balance for the fee-paying coldkey and ensure it is
    sufficient to cover the submission fee plus a small buffer for transaction fees
    and the existential deposit.

    This check is performed *before* requesting an upload slot or performing the
    storage upload of the solution .zip. If the check fails we raise a
    click.ClickException with a clear message so that ``mine-enigma`` never
    uploads a solution for which the payer cannot complete the fee transfer.

    The buffer (default 0.0005 TAO) is a conservative allowance that covers the
    existential deposit (~0.0000005) plus typical extrinsic fees for a small
    Utility.batch_all containing a Balances.transfer_keep_alive + remark.
    """
    import click

    if fee_tao is None or fee_tao <= 0:
        raise click.ClickException("fee_tao is required and must be > 0 for the balance pre-check")

    try:
        with bt.Subtensor(network=network) as subtensor:
            bal = subtensor.balances.get(source_ss58)
            if hasattr(bal, "tao"):
                available = float(bal.tao)
            elif hasattr(bal, "amount"):
                available = float(bal.amount)
            else:
                available = float(bal)
    except Exception as e:
        # Fail closed: do not proceed to upload if we cannot confirm the payer has funds.
        raise click.ClickException(
            f"Failed to query on-chain balance for fee wallet {source_ss58} "
            f"on network {network!r}: {e}\n\n"
            "Aborting submission to avoid uploading without confirmed funds."
        ) from e

    total_needed = fee_tao + buffer_tao
    if available < total_needed:
        raise click.ClickException(
            f"Insufficient balance in fee wallet {source_ss58} for this submission.\n"
            f"  Available free balance : {available:.9f} TAO\n"
            f"  Submission fee         : {fee_tao:.9f} TAO\n"
            f"  Safety buffer (fees+ED): {buffer_tao:.9f} TAO\n"
            f"  Total required         : {total_needed:.9f} TAO\n\n"
            "Please add TAO to the fee-paying coldkey and try the submission again."
        )
