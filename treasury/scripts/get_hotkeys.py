#!/usr/bin/env python3

# The MIT License (MIT)
# Copyright © 2026 qBitTensor Labs

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Get associated hotkeys and voting power for an EVM identity.

uidLookup is keyed by the EOA that signed associate_evm_key — the validator
or treasury admin EOA, not the vault or governor contract.
"""

import argparse
import subprocess
import sys

try:
    from bittensor.wallet import Keypair
except ImportError:
    sys.exit("Please install: pip install bittensor")

from utils.common import DEFAULT_RPC_URL

BITTENSOR_VOTES_ADDRESS = "0x000000000000000000000000000000000000080D"


def _has_contract_code(address: str, rpc: str) -> bool:
    result = subprocess.run(
        ["cast", "code", address, "--rpc-url", rpc],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    code = result.stdout.strip().lower()
    return bool(code) and code not in ("0x", "0x0")


def main():
    parser = argparse.ArgumentParser(
        description="Lookup hotkeys and voting power for an associated EVM identity"
    )
    parser.add_argument("--contract", required=True, help="TreasuryController (governor) address")
    parser.add_argument(
        "--evm",
        "--treasury-admin",
        dest="evm",
        required=True,
        help=(
            "Associated EVM identity (the EOA that signed associate_evm). "
            "Not the vault or governor contract address."
        ),
    )
    parser.add_argument(
        "--rpc",
        default=DEFAULT_RPC_URL,
        help=f"RPC URL (default: {DEFAULT_RPC_URL})"
    )
    args = parser.parse_args()

    print(f"\n🔍 Looking up associated EVM identity: {args.evm}")

    # 1. Fetch Target NetUID from the Governor
    cmd_netuid = ["cast", "call", args.contract, "TARGET_NETUID()(uint16)", "--rpc-url", args.rpc]
    res_netuid = subprocess.run(cmd_netuid, capture_output=True, text=True)
    if res_netuid.returncode != 0:
        print(f"❌ Failed to fetch TARGET_NETUID from {args.contract}")
        sys.exit(1)

    netuid = res_netuid.stdout.strip()
    print(f"   Target NetUID: {netuid}")

    # 2. Get Associated Hotkeys
    cmd_hk = ["cast", "call", args.contract, "getHotkeysForAddress(address)(bytes32[])", args.evm, "--rpc-url", args.rpc]
    res_hk = subprocess.run(cmd_hk, capture_output=True, text=True)

    if res_hk.returncode != 0:
        print(f"❌ Failed to fetch hotkeys: {res_hk.stderr}")
        sys.exit(1)

    output = res_hk.stdout.strip()
    if not output or output == "[]":
        print("\n   ⚠️ No hotkeys associated with this EVM address on-chain.")
        if _has_contract_code(args.evm, args.rpc):
            print("   This address has contract code (vault/governor).")
            print("   uidLookup is keyed by the EOA that signed associate_evm,")
            print("   usually the validator or treasury admin — not the contract.")
        else:
            print("   Pass the EOA used as --private-key on associate_evm.py")
            print("   (not the vault or governor).")
        sys.exit(0)

    clean_output = output.strip("[]").replace('"', '').replace("'", "")
    hex_keys = [k.strip() for k in clean_output.split(",") if k.strip()]

    print(f"   Found {len(hex_keys)} associated hotkey(s).")

    total_power = 0

    # 3. Query Voting Power for each Hotkey
    for i, hk_hex in enumerate(hex_keys, 1):
        if hk_hex.startswith("0x") and len(hk_hex) == 66:
            try:
                kp = Keypair(public_key=bytes.fromhex(hk_hex[2:]), ss58_format=42)
                ss58 = kp.ss58_address
            except Exception:
                ss58 = "Unknown Error decoding SS58"

            cmd_power = [
                "cast", "call", BITTENSOR_VOTES_ADDRESS,
                "getVotingPower(uint16,bytes32)(uint256)",
                netuid, hk_hex, "--rpc-url", args.rpc
            ]
            res_power = subprocess.run(cmd_power, capture_output=True, text=True)

            power = 0
            if res_power.returncode == 0:
                power_str = res_power.stdout.strip().split()[0]
                if power_str.isdigit():
                    power = int(power_str)
            else:
                print(f"     ⚠️ EVM Precompile Failed: {res_power.stderr.strip() or res_power.stdout.strip()}")

            total_power += power

            print(f"\n   → Hotkey #{i}")
            print(f"     SS58:  {ss58}")
            print(f"     Hex:   {hk_hex}")
            print(f"     Power: {power / 1e9:,.4f} τ (Raw: {power})")
        else:
            print(f"\n   → Hotkey #{i}")
            print(f"     Invalid Hex Format: {hk_hex}")

    print("\n" + "=" * 50)
    print(f"✅ Total Combined Voting Power: {total_power / 1e9:,.4f} τ")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
