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

from types import SimpleNamespace

import bittensor as bt
import pytest

from qbittensor.base.neuron import _wallet_from_config
from qbittensor.bt_compat import wallet_kwargs


def test_sdk_wallet_rejects_explicit_none_path():
    """Bittensor v11 does Path(path) in Wallet.__init__; path=None is a TypeError.

    This is the SDK constraint the miner CLI used to hit when --wallet.path was
    omitted (Click default None).
    """
    with pytest.raises(TypeError, match="PathLike|NoneType"):
        bt.Wallet(name="default", hotkey="default", path=None)


def test_wallet_kwargs_omits_none_and_empty_path():
    assert wallet_kwargs(name="n", hotkey="h", path=None) == {
        "name": "n",
        "hotkey": "h",
    }
    assert wallet_kwargs(name="n", hotkey="h", path="") == {
        "name": "n",
        "hotkey": "h",
    }
    assert wallet_kwargs(name="n", hotkey="h", path="/custom") == {
        "name": "n",
        "hotkey": "h",
        "path": "/custom",
    }


def test_wallet_kwargs_none_path_constructs_real_sdk_wallet():
    wallet = bt.Wallet(**wallet_kwargs(name="default", hotkey="default", path=None))
    assert isinstance(wallet.path, str)
    assert wallet.path
    assert wallet.name == "default"
    assert wallet.hotkey_str == "default"


def test_wallet_from_config_omits_none_path():
    cfg = SimpleNamespace(
        wallet=SimpleNamespace(name="miner", hotkey="hk", path=None)
    )
    wallet = _wallet_from_config(cfg)
    assert wallet.name == "miner"
    assert wallet.hotkey_str == "hk"
    assert isinstance(wallet.path, str)
    assert wallet.path


def test_wallet_from_config_passes_custom_path(tmp_path):
    cfg = SimpleNamespace(
        wallet=SimpleNamespace(name="miner", hotkey="hk", path=str(tmp_path))
    )
    wallet = _wallet_from_config(cfg)
    assert wallet.path == str(tmp_path)
