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

"""Wallet loading for the miner CLI.

The default Click path for --wallet.path is None. Bittensor v11 Wallet()
raises TypeError if that None is forwarded as path=. These tests pin the
default CLI path (no --wallet.path) against the real SDK.
"""

from pathlib import Path
from unittest.mock import MagicMock

import bittensor as bt
import click
import pytest
from rich.console import Console

from qbittensor.cli.miner.fee_wallet import (
    load_fee_keypair_from_keyfile,
    load_fee_keypair_from_wallet,
)
from qbittensor.cli.miner.mine_enigma import (
    CliApiAuth,
    _get_miner_hotkey_ss58,
    _load_signing_keypair,
    _resolve_cli_api_auth,
)


_CLEAR_ENV = (
    "WALLET_NAME",
    "WALLET_HOTKEY",
    "WALLET_PATH",
    "BUY_WALLET_COLDKEY",
    "BUY_WALLET_HOTKEY",
    "NETWORK",
    "NETUID",
    "MINER_HOTKEY_SS58",
    "MINER_KEYFILE_PATH",
)


@pytest.fixture
def clean_wallet_env(monkeypatch):
    for key in _CLEAR_ENV:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def temp_wallet(tmp_path):
    wallet = bt.Wallet(name="miner", hotkey="hk1", path=str(tmp_path))
    wallet.create_new_coldkey(use_password=False, suppress=True, overwrite=True)
    wallet.create_new_hotkey(use_password=False, suppress=True, overwrite=True)
    return wallet


def _auth(*, name="miner", hotkey="hk1", path=None) -> CliApiAuth:
    return CliApiAuth(
        wallet_name=name,
        wallet_hotkey=hotkey,
        network="finney",
        netuid=63,
        wallet_path=path,
    )


def test_resolve_cli_api_auth_defaults(clean_wallet_env):
    auth = _resolve_cli_api_auth(None, None)
    assert auth.wallet_name == "default"
    assert auth.wallet_hotkey == "default"
    assert auth.wallet_path is None
    assert auth.network == "finney"
    assert auth.netuid == 63


def test_resolve_cli_api_auth_reads_wallet_path_env(clean_wallet_env, monkeypatch):
    monkeypatch.setenv("WALLET_PATH", "/custom/wallets")
    monkeypatch.setenv("WALLET_NAME", "from-env")
    monkeypatch.setenv("WALLET_HOTKEY", "hot-env")
    auth = _resolve_cli_api_auth(None, None)
    assert auth.wallet_name == "from-env"
    assert auth.wallet_hotkey == "hot-env"
    assert auth.wallet_path == "/custom/wallets"


def test_resolve_cli_api_auth_cli_overrides_env(clean_wallet_env, monkeypatch):
    monkeypatch.setenv("WALLET_PATH", "/from-env")
    monkeypatch.setenv("WALLET_NAME", "env-name")
    monkeypatch.setenv("WALLET_HOTKEY", "env-hot")
    auth = _resolve_cli_api_auth("cli-name", "cli-hot", wallet_path="/from-cli")
    assert auth.wallet_name == "cli-name"
    assert auth.wallet_hotkey == "cli-hot"
    assert auth.wallet_path == "/from-cli"


def test_resolve_cli_api_auth_blank_wallet_path_becomes_none(clean_wallet_env, monkeypatch):
    monkeypatch.setenv("WALLET_PATH", "   ")
    auth = _resolve_cli_api_auth(None, None, wallet_path="")
    assert auth.wallet_path is None


def test_get_miner_hotkey_from_real_wallet(temp_wallet):
    auth = _auth(path=str(temp_wallet.path))
    assert _get_miner_hotkey_ss58(auth) == temp_wallet.hotkey.ss58_address


def test_load_signing_keypair_from_real_wallet(temp_wallet):
    auth = _auth(path=str(temp_wallet.path))
    keypair = _load_signing_keypair(Console(quiet=True), auth)
    assert keypair.ss58_address == temp_wallet.hotkey.ss58_address


def test_load_fee_keypair_from_real_wallet(temp_wallet):
    keypair = load_fee_keypair_from_wallet("miner", wallet_path=str(temp_wallet.path))
    assert keypair.ss58_address == temp_wallet.coldkey.ss58_address


def test_get_miner_hotkey_omitted_path_does_not_raise_nonetype_pathlike(
    clean_wallet_env,
):
    """Regression: listing submissions with no --wallet.path used to crash with

    TypeError: argument should be a str or an os.PathLike object ... NoneType
    wrapped in 'Please configure your wallet using: --wallet.path ...'
    """
    auth = _auth(
        name="enigma-pytest-missing-wallet",
        hotkey="enigma-pytest-missing-hotkey",
        path=None,
    )
    try:
        _get_miner_hotkey_ss58(auth)
    except click.ClickException as exc:
        text = str(exc)
        assert "NoneType" not in text
        assert "PathLike" not in text
        assert "os.PathLike" not in text


def test_load_signing_keypair_omitted_path_does_not_raise_nonetype_pathlike(
    clean_wallet_env,
):
    auth = _auth(
        name="enigma-pytest-missing-wallet",
        hotkey="enigma-pytest-missing-hotkey",
        path=None,
    )
    try:
        _load_signing_keypair(Console(quiet=True), auth)
    except click.ClickException as exc:
        text = str(exc)
        assert "NoneType" not in text
        assert "PathLike" not in text


def test_load_fee_keypair_omitted_path_does_not_raise_nonetype_pathlike():
    try:
        load_fee_keypair_from_wallet(
            "enigma-test-missing-wallet-xyz", wallet_path=None
        )
    except click.ClickException as exc:
        text = str(exc)
        assert "NoneType" not in text
        assert "PathLike" not in text


def test_get_miner_hotkey_does_not_forward_none_path_to_wallet(monkeypatch):
    captured: dict = {}

    def fake_wallet(**kwargs):
        captured.update(kwargs)
        if "path" in kwargs and kwargs["path"] is None:
            raise TypeError(
                'argument should be a str or an os.PathLike object where __fspath__ '
                'returns a str, not "NoneType"'
            )
        wallet = MagicMock()
        wallet.hotkey.ss58_address = "5FakeHotkey"
        return wallet

    monkeypatch.setattr("qbittensor.cli.miner.mine_enigma.bt.Wallet", fake_wallet)
    ss58 = _get_miner_hotkey_ss58(_auth(path=None))
    assert ss58 == "5FakeHotkey"
    assert "path" not in captured


def test_get_miner_hotkey_forwards_custom_path(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_wallet(**kwargs):
        captured.update(kwargs)
        wallet = MagicMock()
        wallet.hotkey.ss58_address = "5FakeHotkey"
        return wallet

    monkeypatch.setattr("qbittensor.cli.miner.mine_enigma.bt.Wallet", fake_wallet)
    ss58 = _get_miner_hotkey_ss58(_auth(path=str(tmp_path)))
    assert ss58 == "5FakeHotkey"
    assert captured["path"] == str(tmp_path)


def test_get_miner_hotkey_env_override_skips_wallet(monkeypatch, clean_wallet_env):
    monkeypatch.setenv("MINER_HOTKEY_SS58", "5ExplicitHotkey")
    assert _get_miner_hotkey_ss58(_auth(path=None)) == "5ExplicitHotkey"


def test_load_signing_keypair_env_keyfile(monkeypatch, temp_wallet, clean_wallet_env):
    keyfile = Path(temp_wallet.path) / "miner" / "hotkeys" / "hk1"
    assert keyfile.is_file()
    monkeypatch.setenv("MINER_KEYFILE_PATH", str(keyfile))
    keypair = _load_signing_keypair(Console(quiet=True), _auth(path=None))
    assert keypair.ss58_address == temp_wallet.hotkey.ss58_address


def test_load_fee_keypair_from_real_keyfile(temp_wallet):
    coldkey_path = Path(temp_wallet.path) / "miner" / "coldkey"
    keypair = load_fee_keypair_from_keyfile(coldkey_path)
    assert keypair.ss58_address == temp_wallet.coldkey.ss58_address


def test_load_validator_keypair_omitted_path_does_not_raise_nonetype_pathlike():
    from qbittensor.cli.validator.upload_diagnostics import _load_validator_keypair

    console = Console(record=True, width=120, force_terminal=True, no_color=True)
    with pytest.raises(click.Abort):
        _load_validator_keypair(
            console,
            "enigma-pytest-missing-wallet",
            "enigma-pytest-missing-hotkey",
            None,
        )
    text = console.export_text()
    assert "NoneType" not in text
    assert "PathLike" not in text
