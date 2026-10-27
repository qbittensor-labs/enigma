"""Resolution of a winner address to the coldkey that receives a payout."""
import importlib.util
from pathlib import Path

import pytest

_COMMON = (
    Path(__file__).resolve().parents[3]
    / "treasury"
    / "scripts"
    / "utils"
    / "common.py"
)
_spec = importlib.util.spec_from_file_location("treasury_scripts_common", _COMMON)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

ZERO_ACCOUNT_SS58 = common.ZERO_ACCOUNT_SS58
resolve_payout_coldkey = common.resolve_payout_coldkey

# Any SS58 the API can return. Nothing here is tied to a particular wallet.
HOTKEY = "5WinnerHotkeyFromApi"
COLDKEY = "5OwnerColdkeyFromChain"


def test_hotkey_uses_owner_coldkey():
    coldkey, from_hotkey = resolve_payout_coldkey(HOTKEY, COLDKEY)
    assert coldkey == COLDKEY
    assert from_hotkey is True


def test_missing_owner_uses_address_as_coldkey():
    coldkey, from_hotkey = resolve_payout_coldkey(COLDKEY, None)
    assert coldkey == COLDKEY
    assert from_hotkey is False


def test_zero_owner_uses_address_as_coldkey():
    coldkey, from_hotkey = resolve_payout_coldkey(COLDKEY, ZERO_ACCOUNT_SS58)
    assert coldkey == COLDKEY
    assert from_hotkey is False


def test_empty_owner_uses_address_as_coldkey():
    coldkey, from_hotkey = resolve_payout_coldkey(COLDKEY, "")
    assert coldkey == COLDKEY
    assert from_hotkey is False


def test_zero_winner_address_is_rejected():
    with pytest.raises(ValueError, match="Invalid winner address"):
        resolve_payout_coldkey(ZERO_ACCOUNT_SS58, None)


def test_empty_winner_address_is_rejected():
    with pytest.raises(ValueError, match="Invalid winner address"):
        resolve_payout_coldkey("", COLDKEY)
