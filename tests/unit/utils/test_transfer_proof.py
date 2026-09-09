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

"""
Solid unit tests for the fee payment verification logic.

These tests focus on the batch + remark parsing and verification helpers
used for fee payments. The fee coldkey signs a Utility.batch_all containing
the transfer + a remark with the binding data.

There is no legacy plain-transfer proof format.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from qbittensor.utils.transfer_proof import (
    _find_remark_data_in_calls,
    _find_transfer_in_calls,
    _get_calls_from_batch,
    _verify_batch_fee_payment_on_chain,
    parse_fee_binding_remark,
    FEE_BINDING_REMARK_VERSION,
)


def make_mock_call(module: str, function: str, args: dict) -> dict:
    return {
        "call_module": module,
        "call_function": function,
        "call_args": [{"name": k, "value": v} for k, v in args.items()],
    }


class TestBatchParsingHelpers:
    def test_get_calls_from_batch(self):
        # Use object with .value to match real substrate decoded extrinsics
        batch_ex = MagicMock()
        batch_ex.value = {
            "call": {
                "call_module": "Utility",
                "call_function": "batch_all",
                "call_args": [
                    {
                        "name": "calls",
                        "value": [
                            make_mock_call("Balances", "transfer_keep_alive", {"dest": "5D..", "value": 123}),
                            make_mock_call("System", "remark_with_event", {"remark": b"hello"}),
                        ],
                    }
                ],
            }
        }
        calls = _get_calls_from_batch(batch_ex)
        assert len(calls) == 2

    def test_find_transfer_in_calls_success(self):
        calls = [
            make_mock_call("System", "remark", {"remark": b"foo"}),
            make_mock_call("Balances", "transfer_keep_alive", {"dest": {"Id": b"\x00" * 32}, "value": 424242}),
        ]
        result = _find_transfer_in_calls(calls)
        assert result is not None
        dest, value = result
        assert value == 424242

    def test_find_remark_data_in_calls(self):
        remark_bytes = f"{FEE_BINDING_REMARK_VERSION}\nminer_hotkey:5H..".encode()
        calls = [
            make_mock_call("Balances", "transfer_keep_alive", {"dest": "x", "value": 1}),
            make_mock_call("System", "remark_with_event", {"remark": remark_bytes}),
        ]
        data = _find_remark_data_in_calls(calls)
        assert data == remark_bytes

    def test_parse_fee_binding_remark(self):
        raw = f"""{FEE_BINDING_REMARK_VERSION}
miner_hotkey:5Hotkey123
milestone_id:milestone-xyz
upload_endpoint_id:upload-abc123
amount_rao:424242
extra:ignored""".encode()

        parsed = parse_fee_binding_remark(raw)
        assert parsed["miner_hotkey"] == "5Hotkey123"
        assert parsed["milestone_id"] == "milestone-xyz"
        assert parsed["upload_endpoint_id"] == "upload-abc123"
        assert parsed["amount_rao"] == "424242"
        assert "extra" not in parsed


class TestBatchFeeVerification:
    def test_successful_batch_with_matching_remark(self):
        substrate = MagicMock()
        substrate.ss58_encode.side_effect = lambda x: "5Coldkey" if isinstance(x, (bytes, bytearray)) else str(x)

        # Build a fake successful receipt
        receipt = Mock()
        receipt.is_success = True
        receipt.error_message = None

        remark_data = f"{FEE_BINDING_REMARK_VERSION}\nminer_hotkey:5Hot\nmilestone_id:m1\nupload_endpoint_id:u1\namount_rao:12345".encode()

        # Wrap in an object with .value so the existing _signer_ss58 and _get_call_dict helpers work
        batch_ex = MagicMock()
        batch_ex.value = {
            "address": "5Coldkey",
            "call": {
                "call_module": "Utility",
                "call_function": "batch_all",
                "call_args": [
                    {
                        "name": "calls",
                        "value": [
                            make_mock_call("Balances", "transfer_keep_alive", {
                                "dest": "5D82xX2p14X7gCGKu2Hpf8feNAzeXefgoeh4UJgVRpVTbVP4",
                                "value": 12345,
                            }),
                            make_mock_call("System", "remark_with_event", {"remark": remark_data}),
                        ],
                    }
                ],
            },
        }

        receipt.extrinsic = batch_ex
        substrate.retrieve_extrinsic_by_hash.return_value = receipt

        ok, err = _verify_batch_fee_payment_on_chain(
            substrate=substrate,
            block_hash="0xblock",
            extrinsic_hash="0xtx",
            expected_signer_ss58="5Coldkey",
            expected_dest_ss58="5D82xX2p14X7gCGKu2Hpf8feNAzeXefgoeh4UJgVRpVTbVP4",
            expected_value_rao=12345,
        )

        assert ok is True, f"Expected success but got: {err}"
        assert err == ""

    def test_rejects_wrong_amount(self):
        substrate = MagicMock()
        substrate.ss58_encode.side_effect = lambda x: "5Coldkey" if isinstance(x, (bytes, bytearray)) else str(x)

        receipt = Mock(is_success=True, error_message=None)

        remark_data = f"{FEE_BINDING_REMARK_VERSION}\namount_rao:999".encode()

        batch_ex = MagicMock()
        batch_ex.value = {
            "address": "5Coldkey",
            "call": {
                "call_module": "Utility",
                "call_function": "batch_all",
                "call_args": [{
                    "name": "calls",
                    "value": [
                        make_mock_call("Balances", "transfer_keep_alive", {
                            "dest": "5D82xX2p14X7gCGKu2Hpf8feNAzeXefgoeh4UJgVRpVTbVP4",
                            "value": 12345,
                        }),
                        make_mock_call("System", "remark_with_event", {"remark": remark_data}),
                    ]
                }]
            }
        }
        receipt.extrinsic = batch_ex
        substrate.retrieve_extrinsic_by_hash.return_value = receipt

        ok, err = _verify_batch_fee_payment_on_chain(
            substrate=substrate,
            block_hash="0xb",
            extrinsic_hash="0xt",
            expected_signer_ss58="5Coldkey",
            expected_dest_ss58="5D82xX2p14X7gCGKu2Hpf8feNAzeXefgoeh4UJgVRpVTbVP4",
            expected_value_rao=12345,
        )

        assert ok is False
        assert "amount" in err.lower() or "value" in err.lower()

    def test_rejects_malformed_remark(self):
        """Adversarial test: remark exists but is garbage."""
        substrate = MagicMock()
        substrate.ss58_encode.side_effect = lambda x: "5Coldkey" if isinstance(x, (bytes, bytearray)) else str(x)

        receipt = Mock(is_success=True, error_message=None)

        garbage = b"this is not a valid binding remark at all"

        batch_ex = MagicMock()
        batch_ex.value = {
            "address": "5Coldkey",
            "call": {
                "call_module": "Utility",
                "call_function": "batch_all",
                "call_args": [{
                    "name": "calls",
                    "value": [
                        make_mock_call("Balances", "transfer_keep_alive", {
                            "dest": "5D82xX2p14X7gCGKu2Hpf8feNAzeXefgoeh4UJgVRpVTbVP4",
                            "value": 12345,
                        }),
                        make_mock_call("System", "remark_with_event", {"remark": garbage}),
                    ]
                }]
            }
        }
        receipt.extrinsic = batch_ex
        substrate.retrieve_extrinsic_by_hash.return_value = receipt

        ok, err = _verify_batch_fee_payment_on_chain(
            substrate=substrate,
            block_hash="0xb",
            extrinsic_hash="0xt",
            expected_signer_ss58="5Coldkey",
            expected_dest_ss58="5D82xX2p14X7gCGKu2Hpf8feNAzeXefgoeh4UJgVRpVTbVP4",
            expected_value_rao=12345,
        )

        assert ok is False
        assert "version" in err.lower() or "remark" in err.lower()

    def test_chain_get_block_fallback_when_state_discarded(self):
        """Lite/localnet nodes prune state; fee extrinsic is still in the block body."""
        substrate = MagicMock()
        substrate.ss58_encode.side_effect = lambda x: (
            "5Coldkey" if isinstance(x, (bytes, bytearray)) else str(x)
        )
        substrate.retrieve_extrinsic_by_hash.side_effect = Exception(
            "State discarded for 0xabc. This indicates the block is too old"
        )
        substrate.get_block.side_effect = Exception("State discarded")

        remark_data = (
            f"{FEE_BINDING_REMARK_VERSION}\nminer_hotkey:5Hot\nmilestone_id:m1\n"
            f"upload_endpoint_id:u1\namount_rao:12345"
        ).encode()
        decoded = MagicMock()
        decoded.value = {
            "address": "5Coldkey",
            "extrinsic_hash": "0xtx",
            "call": {
                "call_module": "Utility",
                "call_function": "batch_all",
                "call_args": [
                    {
                        "name": "calls",
                        "value": [
                            make_mock_call(
                                "Balances",
                                "transfer_keep_alive",
                                {
                                    "dest": "5D82xX2p14X7gCGKu2Hpf8feNAzeXefgoeh4UJgVRpVTbVP4",
                                    "value": 12345,
                                },
                            ),
                            make_mock_call(
                                "System", "remark_with_event", {"remark": remark_data}
                            ),
                        ],
                    }
                ],
            },
        }
        substrate.rpc_request.return_value = {
            "result": {"block": {"extrinsics": ["0xaaaa"]}}
        }
        substrate.decode_scale.return_value = decoded

        with patch(
            "qbittensor.utils.transfer_proof._get_archive_substrate", return_value=None
        ), patch(
            "qbittensor.utils.transfer_proof._fetch_extrinsic_details_via_subscan",
            return_value=None,
        ):
            ok, err = _verify_batch_fee_payment_on_chain(
                substrate=substrate,
                block_hash="0xblock",
                extrinsic_hash="0xtx",
                expected_signer_ss58="5Coldkey",
                expected_dest_ss58="5D82xX2p14X7gCGKu2Hpf8feNAzeXefgoeh4UJgVRpVTbVP4",
                expected_value_rao=12345,
            )

        assert ok is True, f"Expected success via chain_getBlock fallback but got: {err}"
        assert err == ""
        substrate.rpc_request.assert_called()
        substrate.decode_scale.assert_called()

    def test_rejects_failed_receipt(self):
        from async_substrate_interface.errors import ExtrinsicNotFound

        substrate = MagicMock()
        substrate.retrieve_extrinsic_by_hash.side_effect = ExtrinsicNotFound()
        ok, err = _verify_batch_fee_payment_on_chain(
            substrate=substrate,
            block_hash="0xb",
            extrinsic_hash="0xt",
            expected_signer_ss58="5Coldkey",
            expected_dest_ss58="5D82xX2p14X7gCGKu2Hpf8feNAzeXefgoeh4UJgVRpVTbVP4",
            expected_value_rao=1,
        )
        assert ok is False
        assert "not found" in err.lower()

    def test_rejects_wrong_destination(self):
        substrate = MagicMock()
        substrate.ss58_encode.side_effect = lambda x: str(x)
        remark_data = (
            f"{FEE_BINDING_REMARK_VERSION}\namount_rao:10"
        ).encode()
        batch_ex = MagicMock()
        batch_ex.value = {
            "address": "5Coldkey",
            "call": {
                "call_module": "Utility",
                "call_function": "batch_all",
                "call_args": [{
                    "name": "calls",
                    "value": [
                        make_mock_call(
                            "Balances",
                            "transfer_keep_alive",
                            {"dest": "5WrongDest", "value": 10},
                        ),
                        make_mock_call("System", "remark_with_event", {"remark": remark_data}),
                    ],
                }],
            },
        }
        receipt = Mock(is_success=True, error_message=None, extrinsic=batch_ex)
        substrate.retrieve_extrinsic_by_hash.return_value = receipt
        ok, err = _verify_batch_fee_payment_on_chain(
            substrate=substrate,
            block_hash="0xb",
            extrinsic_hash="0xt",
            expected_signer_ss58="5Coldkey",
            expected_dest_ss58="5D82xX2p14X7gCGKu2Hpf8feNAzeXefgoeh4UJgVRpVTbVP4",
            expected_value_rao=10,
        )
        assert ok is False
        assert "destination" in err.lower()


class TestProofHelpers:
    def test_build_transfer_proof_message_strips_0x(self):
        from qbittensor.utils.transfer_proof import (
            TRANSFER_PROOF_VERSION,
            build_transfer_proof_message,
        )

        msg = build_transfer_proof_message(
            miner_hotkey="5H",
            milestone_id="m",
            upload_id="u",
            tx_hash="0xABC",
            transfer_from_ss58="5From",
            transfer_to_ss58="5To",
            transfer_amount_rao="9",
        )
        assert msg.startswith(TRANSFER_PROOF_VERSION)
        assert "tx_hash:abc" in msg

    def test_dest_to_ss58_and_coerce_rao(self):
        from qbittensor.utils.transfer_proof import _coerce_int_rao, _dest_to_ss58

        substrate = MagicMock()
        substrate.ss58_encode.return_value = "5Encoded"
        assert _dest_to_ss58("5Plain", substrate) == "5Plain"
        assert _dest_to_ss58({"Id": b"\x00" * 32}, substrate) == "5Encoded"
        assert _dest_to_ss58(None, substrate) is None
        assert _coerce_int_rao(7) == 7
        assert _coerce_int_rao("8") == 8
        assert _coerce_int_rao(Mock(value=9)) == 9


class TestVerifyTransferProofForSynapse:
    def _proof(self, **overrides):
        from qbittensor.dto.challenge import SolutionCandidateProof, TransferProof
        from qbittensor.utils.transfer_proof import TRANSFER_DEST_SS58, build_transfer_proof_message

        miner = "5MinerHotkey"
        fields = dict(
            tx_hash="0xdead",
            transfer_block_hash="0xbeef",
            transfer_from_ss58="5Cold",
            transfer_to_ss58=TRANSFER_DEST_SS58,
            transfer_amount_rao="100",
            transfer_proof_signature_hex="aa",
            solution_candidate=SolutionCandidateProof(
                challenge_milestone_id="m1",
                upload_endpoint_id="u1",
            ),
        )
        fields.update(overrides)
        if "transfer_proof_message" not in fields:
            fields["transfer_proof_message"] = build_transfer_proof_message(
                miner_hotkey=miner,
                milestone_id="m1",
                upload_id="u1",
                tx_hash=fields["tx_hash"],
                transfer_from_ss58=fields["transfer_from_ss58"],
                transfer_to_ss58=fields["transfer_to_ss58"],
                transfer_amount_rao=fields["transfer_amount_rao"],
            )
        return TransferProof(**fields), miner

    def test_rejects_missing_fields(self):
        from qbittensor.utils.transfer_proof import verify_transfer_proof_for_synapse

        proof, miner = self._proof(tx_hash="   ")
        ok, err = verify_transfer_proof_for_synapse(proof, miner, Mock(), "100")
        assert ok is False
        assert "tx_hash" in err

    def test_rejects_wrong_amount_and_dest(self):
        from qbittensor.utils.transfer_proof import (
            TRANSFER_DEST_SS58,
            verify_transfer_proof_for_synapse,
        )

        proof, miner = self._proof()
        ok, err = verify_transfer_proof_for_synapse(proof, miner, Mock(), "999")
        assert ok is False
        assert "transfer_amount_rao" in err

        proof, miner = self._proof(transfer_to_ss58="5NotTheFeeDest")
        ok, err = verify_transfer_proof_for_synapse(proof, miner, Mock(), "100")
        assert ok is False
        assert TRANSFER_DEST_SS58[:8] in err or "fee destination" in err

    def test_rejects_bad_signature_hex(self):
        from qbittensor.utils.transfer_proof import verify_transfer_proof_for_synapse

        proof, miner = self._proof(transfer_proof_signature_hex="zzzz")
        ok, err = verify_transfer_proof_for_synapse(proof, miner, Mock(), "100")
        assert ok is False
        assert "hex" in err.lower()

    def test_happy_path_with_mocked_signature_and_chain(self):
        from qbittensor.utils.transfer_proof import verify_transfer_proof_for_synapse

        proof, miner = self._proof()
        subtensor = Mock()
        kp = Mock()
        kp.verify.return_value = True
        with (
            patch("qbittensor.utils.transfer_proof.Keypair", return_value=kp),
            patch(
                "qbittensor.utils.transfer_proof._get_hotkey_owner",
                return_value="5Cold",
            ),
            patch(
                "qbittensor.utils.transfer_proof._get_substrate",
                return_value=Mock(),
            ),
            patch(
                "qbittensor.utils.transfer_proof._verify_batch_fee_payment_on_chain",
                return_value=(True, ""),
            ) as verify_chain,
        ):
            ok, err = verify_transfer_proof_for_synapse(proof, miner, subtensor, "100")
        assert ok is True, err
        assert err == ""
        verify_chain.assert_called_once()
        kp.verify.assert_called_once()
