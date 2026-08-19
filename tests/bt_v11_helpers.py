# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
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

from types import SimpleNamespace
from unittest.mock import Mock

from qbittensor.bt_compat import MetagraphAdapter


def make_hyperparameters(
    *,
    min_allowed_weights: int = 1,
    max_weight_limit: float = 1.0,
):
    class _HP:
        def min_allowed_weights(self, netuid: int = None, **_):
            return min_allowed_weights

        def max_weight_limit(self, netuid: int = None, **_):
            return max_weight_limit

    return _HP()


def _metagraph_from_hotkeys(
    hotkeys: list[str],
    *,
    netuid: int,
    subtensor,
    stakes: list[float] | None = None,
) -> MetagraphAdapter:
    neurons = []
    for uid, hk in enumerate(hotkeys):
        stake = 1.0 if stakes is None else float(stakes[uid])
        neurons.append(
            SimpleNamespace(
                uid=uid,
                hotkey=hk,
                coldkey=f"ck{uid}",
                validator_permit=False,
                last_update=0,
                total_stake=SimpleNamespace(amount=stake, tao=stake),
                axon=f"127.0.0.1:{8091 + uid}",
                emission=SimpleNamespace(amount=0.0),
            )
        )
    raw = SimpleNamespace(
        netuid=netuid,
        neurons=neurons,
        hotkeys=list(hotkeys),
        coldkeys=[f"ck{i}" for i in range(len(hotkeys))],
        block=0,
        num_uids=len(neurons),
    )
    return MetagraphAdapter(raw, netuid=netuid, subtensor=subtensor)


def wire_v11_subtensor(
    mock_subtensor: Mock,
    *,
    hotkeys: list[str],
    netuid: int = 1,
    registered: bool = True,
    block: int = 1000,
    min_allowed_weights: int = 1,
    max_weight_limit: float = 1.0,
    stakes: list[float] | None = None,
    last_update=None,
) -> MetagraphAdapter:
    """Configure Mock subtensor with v11-only namespaces. Returns the adapted metagraph."""
    mock_subtensor.block = block
    mock_subtensor.network = getattr(mock_subtensor, "network", None) or "finney"
    mock_subtensor.endpoint = getattr(mock_subtensor, "endpoint", None) or "finney"
    mock_subtensor.hyperparameters = make_hyperparameters(
        min_allowed_weights=min_allowed_weights,
        max_weight_limit=max_weight_limit,
    )

    adapted = _metagraph_from_hotkeys(
        hotkeys, netuid=netuid, subtensor=mock_subtensor, stakes=stakes
    )
    if last_update is not None:
        adapted.last_update = last_update

    mock_subtensor.subnets.metagraph.return_value = adapted

    def _uid(hotkey_ss58: str, netuid: int = None, **_):
        if not registered:
            return None
        try:
            return hotkeys.index(hotkey_ss58)
        except ValueError:
            return None

    mock_subtensor.neurons.uid.side_effect = _uid

    mock_result = Mock()
    mock_result.success = True
    mock_result.message = "ok"
    mock_result.error = None
    mock_subtensor.execute.return_value = mock_result
    return adapted
