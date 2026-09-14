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

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from qbittensor.utils.services.challenges import ChallengesClient

from neurons.validator import (
    PRIVATE_MINER_HOTKEY,
    TREASURY_HOTKEY,
    MIN_DUST_FLOOR,
    WEIGHTS_MIN_INTERVAL_BLOCKS,
    Validator,
)
from qbittensor.utils.treasury_sinks import TREASURY_SINK_HOTKEYS
from qbittensor.utils.burn_sinks import BURN_SINK_HOTKEYS
from qbittensor.utils.treasury_cap import CapDecision

from qbittensor.base.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
)


@pytest.fixture
def mock_config():
    """Mock config for validator."""
    config = Validator.config()
    config.neuron.forward_sleep_interval = 5
    config.neuron.epoch_length = 100
    config.neuron.disable_set_weights = False
    config.neuron.moving_average_alpha = 0.1
    config.neuron.axon_off = True
    config.netuid = 1
    config.mock = False
    return config


@pytest.fixture
def mock_validator(mock_config):
    """Create a mock validator instance with mocked dependencies.

    Patch strategy (post platform API consolidation):
    - Primary patches target the canonical locations under qbittensor.*.
    - Secondary patches on neurons.* are still required for classes that
      neurons/validator.py imports locally (the name lookup happens in that
      module's globals). This is transitional technical debt.
    - ChallengesClient is now explicitly patched (and wired as .platform_client)
      so that Validator construction and forward() paths are properly isolated.
    """
    with (
        patch("qbittensor.base.neuron.bt.Wallet") as mock_wallet,
        patch("qbittensor.base.neuron.bt.Subtensor") as mock_subtensor,
        patch("qbittensor.base.neuron.bt.Metagraph") as mock_metagraph,
        patch("qbittensor.base.validator.bt.Dendrite") as mock_dendrite,
        patch("qbittensor.base.validator.bt.Axon") as mock_axon,
        patch("qbittensor.base.neuron.BaseNeuron.sync") as mock_sync,
        patch("qbittensor.base.validator.BaseValidatorNeuron.load_state") as mock_load_state,
        patch("qbittensor.base.neuron.check_config"),
        patch("qbittensor.utils.services.telemetry.TelemetryService") as mock_telemetry_service,
        patch("neurons.validator.TelemetryService"),
        # Explicitly isolate the platform client (new in post-consolidation Validator)
        patch("qbittensor.utils.services.challenges.ChallengesClient") as mock_challenges_client,
        patch("qbittensor.database.db_connection.DBConnection") as mock_db_connection,
        patch("qbittensor.validator.synapse.process_responses.ResponseProcessor") as mock_response_processor_cls,
        patch("qbittensor.validator.solution.solution_container_manager.SolutionContainerManager") as mock_solution_container_manager_cls,
        patch("qbittensor.validator.solution.solution_cross_check.SolutionCrossChecker") as mock_cross_check_cls,
        patch("neurons.validator.test_gpu_container", return_value=True),
        patch("neurons.validator.is_docker_available"),
    ):

        mock_wallet.return_value = Mock()
        mock_wallet.return_value.hotkey.ss58_address = "test_hotkey"
        mock_subtensor.return_value = Mock()
        from tests.bt_v11_helpers import wire_v11_subtensor

        mock_mg = wire_v11_subtensor(
            mock_subtensor.return_value,
            hotkeys=["test_hotkey", "hotkey1", "miner_hotkey"],
            netuid=1,
            stakes=[1000.0, 100.0, 50.0],
            last_update=[0, 0, 0],
            max_weight_limit=1.0,
        )
        mock_metagraph.return_value = mock_mg

        mock_dendrite.return_value = Mock()
        mock_dendrite.return_value._session = None
        mock_dendrite.return_value.aclose_session = AsyncMock()
        mock_dendrite.return_value.forward = AsyncMock(return_value=Mock())
        mock_axon.return_value = Mock()

        mock_telemetry = Mock()
        mock_telemetry.record_startup_metrics = Mock()
        mock_telemetry.heartbeat_timer = Mock()
        mock_telemetry.system_metrics_timer = Mock()
        mock_telemetry_service.return_value = mock_telemetry

        mock_db_connection.return_value = Mock()
        mock_db_connection.return_value.db_query.get_miner_submission_statuses.return_value = []
        mock_db_connection.return_value.db_query.get_active_miners.return_value = []
        mock_db_connection.return_value.db_query.prune_old_miner_solutions.return_value = None
        mock_response_processor_cls.return_value = Mock()
        mock_solution_container_manager_cls.return_value = Mock()
        mock_solution_container_manager_cls.return_value.validator_is_busy.return_value = False
        mock_cross_check_cls.return_value = Mock()

        validator = Validator(config=mock_config)

        # Wire key post-consolidation mocks onto the live instance
        validator.database_connection = mock_db_connection.return_value
        validator.response_processor = mock_response_processor_cls.return_value
        validator.platform_client = mock_challenges_client.return_value

        # Give the ChallengesClient mock some sensible defaults used by forward/cross-check paths
        mock_challenges_client.return_value.submit_solution.return_value = None
        mock_challenges_client.return_value.get_next_cross_check_submission.return_value = None
        mock_challenges_client.return_value.get_milestone_price_tao.return_value = 0.1

        validator.sync = Mock()
        validator.load_state = Mock()
        validator.save_state = Mock()
        yield validator


class TestSetWeights:
    """Test cases for weight distribution via the canonical Validator.set_weights().

    This now follows the standard Bittensor pattern (override set_weights, populate
    self.scores, call super().set_weights()). The old calculate_weights + custom timer
    mechanism has been removed.
    """

    @staticmethod
    def _stub_sink_jwt(
        mock_validator,
        sink_hotkey=TREASURY_HOTKEY,
        burn_sink_hotkey=None,
        *,
        error=None,
        tempo_id=None,
        cached_jwt=None,
    ):
        rm = Mock()
        if error is not None:
            rm.refresh_jwt.side_effect = error
        else:
            jwt = Mock()
            jwt.sink_hotkey = sink_hotkey
            jwt.burn_hotkey = burn_sink_hotkey
            jwt.tempo_id = tempo_id
            rm.refresh_jwt.return_value = jwt
        rm.jwt = cached_jwt
        mock_validator.platform_client.request_manager = rm
        return rm

    def test_set_weights_distributes_maintenance_and_treasury(self, mock_validator):
        """Every maintenance miner gets at least the floor; treasury takes (nearly) all remaining mass."""
        mock_validator.metagraph.hotkeys = [TREASURY_HOTKEY, "miner1", "miner2"]
        mock_validator.database_connection.db_query.get_active_miners.return_value = ["miner1"]
        self._stub_sink_jwt(mock_validator)

        with patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super:
            mock_validator.set_weights()

        weights = mock_validator.scores
        treasury_uid = mock_validator.metagraph.hotkeys.index(TREASURY_HOTKEY)

        # New floor-based policy: maintenance miners get the guaranteed floor (or more)
        assert weights[1] >= MIN_DUST_FLOOR
        assert weights[2] == 0.0
        # Treasury gets almost everything left after the tiny floors
        assert weights[treasury_uid] >= 0.999
        mock_validator.database_connection.db_query.prune_old_miner_solutions.assert_called_once()
        mock_super.assert_called_once()

    def test_set_weights_includes_private_miner_when_not_in_db(self, mock_validator):
        """Private miner hotkey always receives the guaranteed floor (even with zero DB miners)."""
        mock_validator.metagraph.hotkeys = [TREASURY_HOTKEY, "miner1", PRIVATE_MINER_HOTKEY]
        mock_validator.database_connection.db_query.get_active_miners.return_value = []
        self._stub_sink_jwt(mock_validator)

        with patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super:
            mock_validator.set_weights()

        weights = mock_validator.scores
        treasury_uid = mock_validator.metagraph.hotkeys.index(TREASURY_HOTKEY)
        private_miner_uid = mock_validator.metagraph.hotkeys.index(PRIVATE_MINER_HOTKEY)

        assert weights[private_miner_uid] >= MIN_DUST_FLOOR
        # Treasury still dominates
        assert weights[treasury_uid] >= 0.999
        mock_super.assert_called_once()

    def test_floor_protects_many_miners_at_high_treasury_from_quantization(self, mock_validator):
        """With MIN_DUST_FLOOR, even with many maintenance miners, every maintained
        UID survives the full processing + u16 quantization with >0 weight.
        """
        n = 256
        treasury_uid = 87

        # Realistic metagraph
        hotkeys = [f"hk_{i:03d}" for i in range(n)]
        hotkeys[treasury_uid] = TREASURY_HOTKEY

        # 60 maintenance miners (typical active set size) + the private one
        db_maintain = [f"hk_{i:03d}" for i in range(20, 80)]  # 60
        all_maintain = list(db_maintain) + [PRIVATE_MINER_HOTKEY]

        # Place the private miner at a plausible UID
        private_uid = 171
        hotkeys[private_uid] = PRIVATE_MINER_HOTKEY

        mock_validator.metagraph.hotkeys = hotkeys
        mock_validator.metagraph.n = n
        mock_validator.metagraph.uids = np.arange(n, dtype=np.int64)
        mock_validator.database_connection.db_query.get_active_miners.return_value = db_maintain
        self._stub_sink_jwt(mock_validator)

        with patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super:
            mock_validator.set_weights()

        scores = mock_validator.scores

        # Replicate exactly what BaseValidatorNeuron.set_weights does
        norm = np.linalg.norm(scores, ord=1)
        if norm == 0 or np.isnan(norm):
            norm = 1.0
        raw_weights = scores / norm

        mock_st = mock_validator.subtensor
        from tests.bt_v11_helpers import make_hyperparameters
        mock_st.hyperparameters = make_hyperparameters(
            min_allowed_weights=1, max_weight_limit=1.0
        )

        processed_uids, processed_w = process_weights_for_netuid(
            uids=mock_validator.metagraph.uids,
            weights=raw_weights,
            netuid=63,
            subtensor=mock_st,
            metagraph=mock_validator.metagraph,
        )

        uint_uids, uint_weights = convert_weights_and_uids_for_emit(
            uids=processed_uids, weights=processed_w
        )
        emitted = dict(zip([int(u) for u in uint_uids], uint_weights))

        # Proof: every single maintenance hotkey must have a non-zero emitted weight
        zeroed = []
        for hk in all_maintain:
            if hk in hotkeys:
                uid = hotkeys.index(hk)
                if emitted.get(uid, 0) == 0:
                    zeroed.append(uid)

        assert not zeroed, (
            f"With floor={MIN_DUST_FLOOR}, these maintenance UIDs were zeroed after "
            f"quantization: {zeroed}"
        )

        # Also sanity: treasury itself must be present and large
        assert emitted.get(treasury_uid, 0) > 10000  # comfortably non-zero
        mock_super.assert_called_once()

    def test_one_maintenance_miner_dust_survives_full_pipeline(self, mock_validator):
        """1 maintenance miner (only the forced private miner, zero from DB).

        This is the *most* stressful dust case: the single floor is at its smallest
        relative size after max-scaling against a near-1.0 treasury weight.
        The final u16 quantization in convert_weights_and_uids_for_emit must still
        produce a non-zero weight for it.
        """
        n = 256
        treasury_uid = 42

        hotkeys = [f"hk_{i:03d}" for i in range(n)]
        hotkeys[treasury_uid] = TREASURY_HOTKEY

        # Only the private miner is maintained (simulates get_active_miners() returning [])
        private_uid = 17
        hotkeys[private_uid] = PRIVATE_MINER_HOTKEY

        mock_validator.metagraph.hotkeys = hotkeys
        mock_validator.metagraph.n = n
        mock_validator.metagraph.uids = np.arange(n, dtype=np.int64)
        mock_validator.database_connection.db_query.get_active_miners.return_value = []
        self._stub_sink_jwt(mock_validator)

        with patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super:
            mock_validator.set_weights()

        scores = mock_validator.scores

        # Full on-chain pipeline (identical to BaseValidatorNeuron.set_weights + utils)
        norm = np.linalg.norm(scores, ord=1)
        if norm == 0 or np.isnan(norm):
            norm = 1.0
        raw_weights = scores / norm

        mock_st = mock_validator.subtensor
        from tests.bt_v11_helpers import make_hyperparameters
        mock_st.hyperparameters = make_hyperparameters(
            min_allowed_weights=1, max_weight_limit=1.0
        )

        processed_uids, processed_w = process_weights_for_netuid(
            uids=mock_validator.metagraph.uids,
            weights=raw_weights,
            netuid=63,
            subtensor=mock_st,
            metagraph=mock_validator.metagraph,
        )
        uint_uids, uint_weights = convert_weights_and_uids_for_emit(
            uids=processed_uids, weights=processed_w
        )
        emitted = dict(zip([int(u) for u in uint_uids], uint_weights))

        # The single maintenance miner (private) must survive with non-zero u16 weight
        assert emitted.get(private_uid, 0) > 0, (
            f"Private miner (only maintenance UID) was zeroed in u16 emit. "
            f"floor={MIN_DUST_FLOOR}, emitted={emitted.get(private_uid, 0)}"
        )

        # Treasury must also be present and dominant
        assert emitted.get(treasury_uid, 0) > 10000

        mock_super.assert_called_once()

    def test_255_maintenance_miners_dust_survives_full_pipeline(self, mock_validator):
        """255 maintenance miners (maximum possible on a 256-UID subnet with 1 treasury).

        Every one of the 255 floors must survive process_weights + u16 quantization.
        This stresses the path with the largest number of tiny non-zero weights.
        """
        n = 256
        treasury_uid = 0

        hotkeys = [f"hk_{i:03d}" for i in range(n)]
        hotkeys[treasury_uid] = TREASURY_HOTKEY

        # 255 maintenance hotkeys (all except treasury). Include the canonical private one.
        maintenance_hotkeys = [f"hk_{i:03d}" for i in range(1, 256)]
        # Make the last one the private miner hotkey so we also prove it is protected
        private_uid = 255
        hotkeys[private_uid] = PRIVATE_MINER_HOTKEY
        # Replace the last maintenance entry with the real private hotkey string for realism
        maintenance_hotkeys[-1] = PRIVATE_MINER_HOTKEY

        # DB returns all except the private one (proves the "always append private" path)
        db_maintain = maintenance_hotkeys[:-1]

        mock_validator.metagraph.hotkeys = hotkeys
        mock_validator.metagraph.n = n
        mock_validator.metagraph.uids = np.arange(n, dtype=np.int64)
        mock_validator.database_connection.db_query.get_active_miners.return_value = db_maintain
        self._stub_sink_jwt(mock_validator)

        with patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super:
            mock_validator.set_weights()

        scores = mock_validator.scores

        # Verify in scores (pre-normalization) that all 255 got the floor
        for hk in maintenance_hotkeys:
            uid = hotkeys.index(hk)
            assert scores[uid] >= MIN_DUST_FLOOR - 1e-12

        # Full pipeline to emitted u16 weights
        norm = np.linalg.norm(scores, ord=1)
        if norm == 0 or np.isnan(norm):
            norm = 1.0
        raw_weights = scores / norm

        mock_st = mock_validator.subtensor
        from tests.bt_v11_helpers import make_hyperparameters
        mock_st.hyperparameters = make_hyperparameters(
            min_allowed_weights=1, max_weight_limit=1.0
        )

        processed_uids, processed_w = process_weights_for_netuid(
            uids=mock_validator.metagraph.uids,
            weights=raw_weights,
            netuid=63,
            subtensor=mock_st,
            metagraph=mock_validator.metagraph,
        )
        uint_uids, uint_weights = convert_weights_and_uids_for_emit(
            uids=processed_uids, weights=processed_w
        )
        emitted = dict(zip([int(u) for u in uint_uids], uint_weights))

        # Every single one of the 255 maintenance UIDs must have positive emitted weight
        zeroed = []
        for hk in maintenance_hotkeys:
            uid = hotkeys.index(hk)
            if emitted.get(uid, 0) == 0:
                zeroed.append(uid)

        assert not zeroed, (
            f"With 255 maintenance miners and floor={MIN_DUST_FLOOR}, "
            f"these UIDs were zeroed after full quantization: {zeroed}"
        )

        # Treasury must still receive a large share
        assert emitted.get(treasury_uid, 0) > 10000

        mock_super.assert_called_once()

    def test_raw_uint_weights_sum_never_exceeds_u16_max(self, mock_validator):
        """Regression test: emitted uint16 weights must sum to <= 65535.

        The previous max-upscale + round logic in convert_weights_and_uids_for_emit
        could produce sums like 65537 (treasury=65535 + dust=2). This violates the
        documented contract and triggers on-chain warnings. The correction logic
        must shave excess off the dominant weight while preserving all dust.
        """
        n = 256
        treasury_uid = 87
        private_uid = 171

        hotkeys = [f"hk_{i:03d}" for i in range(n)]
        hotkeys[treasury_uid] = TREASURY_HOTKEY
        hotkeys[private_uid] = PRIVATE_MINER_HOTKEY

        mock_validator.metagraph.hotkeys = hotkeys
        mock_validator.metagraph.n = n
        mock_validator.metagraph.uids = np.arange(n, dtype=np.int64)
        # Only the private miner gets the floor (worst-case single dust scenario)
        mock_validator.database_connection.db_query.get_active_miners.return_value = []
        self._stub_sink_jwt(mock_validator)

        with patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super:
            mock_validator.set_weights()

        scores = mock_validator.scores

        # Replicate the exact pipeline that BaseValidatorNeuron.set_weights runs
        norm = np.linalg.norm(scores, ord=1)
        if norm == 0 or np.isnan(norm):
            norm = 1.0
        raw_weights = scores / norm

        mock_st = mock_validator.subtensor
        from tests.bt_v11_helpers import make_hyperparameters
        mock_st.hyperparameters = make_hyperparameters(
            min_allowed_weights=1, max_weight_limit=1.0
        )

        processed_uids, processed_w = process_weights_for_netuid(
            uids=mock_validator.metagraph.uids,
            weights=raw_weights,
            netuid=63,
            subtensor=mock_st,
            metagraph=mock_validator.metagraph,
        )
        uint_uids, uint_weights = convert_weights_and_uids_for_emit(
            uids=processed_uids, weights=processed_w
        )

        total_raw = sum(uint_weights)
        assert total_raw <= 65535, (
            f"Total raw weight {total_raw} exceeds U16_MAX=65535. "
            f"Emitted: {dict(zip([int(u) for u in uint_uids], uint_weights))}"
        )

        # The dust must still be present (keep-alive must not be broken by the correction)
        emitted = dict(zip([int(u) for u in uint_uids], uint_weights))
        assert emitted.get(private_uid, 0) > 0, "Private miner dust was zeroed"

        mock_super.assert_called_once()

    def test_set_weights_dusts_idle_treasury_sinks(self, mock_validator):
        """Idle sinks stay above 0 so a full-subnet recycle cannot pick them."""
        active = TREASURY_SINK_HOTKEYS[0]
        idle = list(TREASURY_SINK_HOTKEYS[1:])
        hotkeys = [active, *idle, PRIVATE_MINER_HOTKEY]
        mock_validator.metagraph.hotkeys = hotkeys
        mock_validator.metagraph.n = len(hotkeys)
        mock_validator.metagraph.uids = np.arange(len(hotkeys), dtype=np.int64)
        mock_validator.database_connection.db_query.get_active_miners.return_value = []
        self._stub_sink_jwt(mock_validator, active)

        with patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super:
            mock_validator.set_weights()

        weights = mock_validator.scores
        assert weights[mock_validator.metagraph.hotkeys.index(active)] >= 0.999
        for hk in idle:
            assert weights[mock_validator.metagraph.hotkeys.index(hk)] >= MIN_DUST_FLOOR
        mock_super.assert_called_once()

    def test_set_weights_uses_platform_sink_when_listed(self, mock_validator):
        listed = TREASURY_SINK_HOTKEYS[2]
        mock_validator.metagraph.hotkeys = [listed, "miner1", PRIVATE_MINER_HOTKEY]
        mock_validator.database_connection.db_query.get_active_miners.return_value = []
        rm = self._stub_sink_jwt(mock_validator, listed)

        with patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super:
            mock_validator.set_weights()

        weights = mock_validator.scores
        sink_uid = mock_validator.metagraph.hotkeys.index(listed)
        assert weights[sink_uid] >= 0.999
        rm.refresh_jwt.assert_called_once()
        mock_super.assert_called_once()

    def test_set_weights_skips_chain_when_platform_sink_unknown(self, mock_validator):
        unknown = "5NotATreasurySinkHotkeyXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        mock_validator.metagraph.hotkeys = [TREASURY_HOTKEY, unknown, PRIVATE_MINER_HOTKEY]
        mock_validator.database_connection.db_query.get_active_miners.return_value = []
        self._stub_sink_jwt(mock_validator, unknown)

        with patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super:
            mock_validator.set_weights()

        mock_super.assert_not_called()

    def test_set_weights_skips_chain_when_sink_missing(self, mock_validator):
        mock_validator.metagraph.hotkeys = [TREASURY_HOTKEY, "miner1", PRIVATE_MINER_HOTKEY]
        mock_validator.database_connection.db_query.get_active_miners.return_value = []
        self._stub_sink_jwt(mock_validator, error=RuntimeError("tensorauth down"))

        with patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super:
            mock_validator.set_weights()

        mock_super.assert_not_called()

    def test_set_weights_skips_chain_when_request_manager_missing(self, mock_validator):
        mock_validator.metagraph.hotkeys = [TREASURY_HOTKEY, "miner1", PRIVATE_MINER_HOTKEY]
        mock_validator.database_connection.db_query.get_active_miners.return_value = []
        mock_validator.platform_client.request_manager = None

        with patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super:
            mock_validator.set_weights()

        mock_super.assert_not_called()

    def test_set_weights_always_refreshes_jwt(self, mock_validator):
        """Cached JWT is ignored even when it looks current, including default sink."""
        cached_sink = TREASURY_SINK_HOTKEYS[0]
        live_sink = TREASURY_SINK_HOTKEYS[2]
        mock_validator.metagraph.hotkeys = [
            cached_sink, live_sink, PRIVATE_MINER_HOTKEY
        ]
        mock_validator.database_connection.db_query.get_active_miners.return_value = []

        cached = Mock()
        cached.sink_hotkey = cached_sink
        cached.burn_hotkey = BURN_SINK_HOTKEYS[0]
        cached.tempo_id = 2
        rm = self._stub_sink_jwt(
            mock_validator, live_sink, BURN_SINK_HOTKEYS[0], tempo_id=2, cached_jwt=cached
        )
        mock_validator.subtensor.block = 720  # tempo 2 — cache would look fresh

        with patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super:
            mock_validator.set_weights()

        weights = mock_validator.scores
        assert weights[mock_validator.metagraph.hotkeys.index(live_sink)] >= 0.999
        assert weights[mock_validator.metagraph.hotkeys.index(cached_sink)] >= MIN_DUST_FLOOR
        rm.refresh_jwt.assert_called_once()
        mock_super.assert_called_once()

    def test_set_weights_dusts_idle_burn_sinks(self, mock_validator, monkeypatch):
        sink = TREASURY_SINK_HOTKEYS[0]
        burn = "5SecondBurnSinkHotkeyXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        monkeypatch.setenv("BURN_SINK_HOTKEYS", f"{BURN_SINK_HOTKEYS[0]},{burn}")
        mock_validator.metagraph.hotkeys = [sink, burn, PRIVATE_MINER_HOTKEY]
        mock_validator.database_connection.db_query.get_active_miners.return_value = []
        self._stub_sink_jwt(mock_validator, sink, BURN_SINK_HOTKEYS[0])

        with patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super:
            mock_validator.set_weights()

        weights = mock_validator.scores
        assert weights[mock_validator.metagraph.hotkeys.index(sink)] >= 0.999
        assert weights[mock_validator.metagraph.hotkeys.index(burn)] >= MIN_DUST_FLOOR
        mock_super.assert_called_once()

    def test_set_weights_skips_chain_when_treasury_missing_from_metagraph(self, mock_validator):
        mock_validator.metagraph.hotkeys = ["miner1", PRIVATE_MINER_HOTKEY]
        mock_validator.database_connection.db_query.get_active_miners.return_value = ["miner1"]
        self._stub_sink_jwt(mock_validator, TREASURY_HOTKEY)

        with patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super:
            mock_validator.set_weights()

        mock_super.assert_not_called()

    def test_successful_set_weights_stamps_local_last_update(self, mock_validator):
        """A successful submit must close the window so the next 5s loop does not resubmit."""
        from qbittensor.base.validator import BaseValidatorNeuron

        mock_validator.uid = 0
        mock_validator.step = 1
        mock_validator.scores = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_validator.metagraph.last_update = [0, 0, 0]
        mock_validator.metagraph.uids = np.arange(3, dtype=np.int64)
        mock_validator.subtensor.block = 500
        mock_validator.subtensor.execute.return_value = Mock(success=True, error=None)

        with (
            patch(
                "qbittensor.base.validator.process_weights_for_netuid",
                return_value=(np.array([0]), np.array([1.0])),
            ),
            patch(
                "qbittensor.base.validator.convert_weights_and_uids_for_emit",
                return_value=([0], [1.0]),
            ),
        ):
            BaseValidatorNeuron.set_weights(mock_validator)

        assert mock_validator.metagraph.last_update[0] == 500

    def test_failed_set_weights_does_not_stamp_last_update(self, mock_validator):
        from qbittensor.base.validator import BaseValidatorNeuron

        mock_validator.uid = 0
        mock_validator.step = 1
        mock_validator.scores = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_validator.metagraph.last_update = [0, 0, 0]
        mock_validator.metagraph.uids = np.arange(3, dtype=np.int64)
        mock_validator.subtensor.block = 500
        mock_validator.subtensor.execute.return_value = Mock(success=False, error=None, message="pool full")

        with (
            patch(
                "qbittensor.base.validator.process_weights_for_netuid",
                return_value=(np.array([0]), np.array([1.0])),
            ),
            patch(
                "qbittensor.base.validator.convert_weights_and_uids_for_emit",
                return_value=([0], [1.0]),
            ),
        ):
            BaseValidatorNeuron.set_weights(mock_validator)

        assert mock_validator.metagraph.last_update[0] == 0

    def test_set_weights_execute_exception_does_not_stamp(self, mock_validator):
        from qbittensor.base.validator import BaseValidatorNeuron

        mock_validator.uid = 0
        mock_validator.step = 1
        mock_validator.scores = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_validator.metagraph.last_update = [0, 0, 0]
        mock_validator.metagraph.uids = np.arange(3, dtype=np.int64)
        mock_validator.subtensor.block = 500
        mock_validator.subtensor.execute.side_effect = RuntimeError("rpc down")

        with (
            patch(
                "qbittensor.base.validator.process_weights_for_netuid",
                return_value=(np.array([0]), np.array([1.0])),
            ),
            patch(
                "qbittensor.base.validator.convert_weights_and_uids_for_emit",
                return_value=([0], [1.0]),
            ),
        ):
            BaseValidatorNeuron.set_weights(mock_validator)

        assert mock_validator.metagraph.last_update[0] == 0

    def test_set_weights_failed_result_uses_error_remediation(self, mock_validator):
        from qbittensor.base.validator import BaseValidatorNeuron

        mock_validator.uid = 0
        mock_validator.step = 1
        mock_validator.scores = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_validator.metagraph.last_update = [0, 0, 0]
        mock_validator.metagraph.uids = np.arange(3, dtype=np.int64)
        err = Mock()
        err.remediation = "wait for inclusion"
        mock_validator.subtensor.execute.return_value = Mock(success=False, error=err)

        with (
            patch(
                "qbittensor.base.validator.process_weights_for_netuid",
                return_value=(np.array([0]), np.array([1.0])),
            ),
            patch(
                "qbittensor.base.validator.convert_weights_and_uids_for_emit",
                return_value=([0], [1.0]),
            ),
        ):
            BaseValidatorNeuron.set_weights(mock_validator)

        assert mock_validator.metagraph.last_update[0] == 0

    def test_mark_local_weights_submitted_skips_missing_last_update(self, mock_validator):
        from qbittensor.base.validator import BaseValidatorNeuron

        mock_validator.uid = 0
        mock_validator.metagraph.last_update = None
        mock_validator.subtensor.block = 9
        BaseValidatorNeuron._mark_local_weights_submitted(mock_validator)

    def test_successful_emit_records_recipient(self, mock_validator):
        mock_validator.uid = 0
        mock_validator.step = 1
        mock_validator.metagraph.hotkeys = [TREASURY_HOTKEY, PRIVATE_MINER_HOTKEY]
        mock_validator.metagraph.last_update = [0, 0]
        mock_validator.subtensor.block = 500
        mock_validator.subtensor.execute.return_value = Mock(success=True, error=None)
        self._stub_sink_jwt(mock_validator, TREASURY_HOTKEY)

        with (
            patch(
                "qbittensor.base.validator.process_weights_for_netuid",
                return_value=(np.array([0]), np.array([1.0])),
            ),
            patch(
                "qbittensor.base.validator.convert_weights_and_uids_for_emit",
                return_value=([0], [1.0]),
            ),
        ):
            mock_validator.set_weights()

        assert mock_validator._last_emitted_recipient == TREASURY_HOTKEY
        assert mock_validator.metagraph.last_update[0] == 500


class TestShouldSetWeights:
    def _ready(self, mock_validator, *, block=720, last_update=0, sink=None):
        mock_validator.uid = 0
        mock_validator.step = 1
        mock_validator.subtensor.block = block
        mock_validator.metagraph.last_update = [last_update, 0, 0]
        sink = sink or TREASURY_SINK_HOTKEYS[2]
        mock_validator.metagraph.hotkeys = [sink, TREASURY_HOTKEY, PRIVATE_MINER_HOTKEY]
        TestSetWeights._stub_sink_jwt(mock_validator, sink)
        return sink

    def test_step_zero_never_sets(self, mock_validator):
        self._ready(mock_validator)
        mock_validator.step = 0
        assert mock_validator.should_set_weights() is False

    def test_first_jwt_claim_sets(self, mock_validator):
        sink = self._ready(mock_validator)
        assert mock_validator.should_set_weights() is True
        assert mock_validator._planned_weight_targets.recipient == sink

    def test_same_recipient_skips(self, mock_validator):
        sink = self._ready(mock_validator, last_update=0, block=720)
        mock_validator._last_emitted_recipient = sink
        mock_validator._last_seen_tempo = 720 // 360
        assert mock_validator.should_set_weights() is False

    def test_new_sink_after_tempo_change_sets(self, mock_validator):
        old = TREASURY_SINK_HOTKEYS[0]
        new = TREASURY_SINK_HOTKEYS[2]
        self._ready(mock_validator, block=720, last_update=710, sink=new)
        mock_validator._last_emitted_recipient = old
        mock_validator._last_seen_tempo = 1  # previous tempo
        assert mock_validator.should_set_weights() is True
        assert mock_validator._planned_weight_targets.recipient == new

    def test_cooldown_skips_even_if_sink_changed(self, mock_validator):
        old = TREASURY_SINK_HOTKEYS[0]
        new = TREASURY_SINK_HOTKEYS[2]
        self._ready(mock_validator, block=714, last_update=710, sink=new)
        mock_validator._last_emitted_recipient = old
        mock_validator._last_seen_tempo = 1
        assert (714 - 710) < WEIGHTS_MIN_INTERVAL_BLOCKS
        assert mock_validator.should_set_weights() is False

    def test_missing_jwt_does_not_set(self, mock_validator):
        self._ready(mock_validator)
        TestSetWeights._stub_sink_jwt(
            mock_validator, error=RuntimeError("tensorauth down")
        )
        assert mock_validator.should_set_weights() is False

    def test_unknown_jwt_sink_does_not_set(self, mock_validator):
        self._ready(mock_validator)
        TestSetWeights._stub_sink_jwt(mock_validator, "5NotATreasurySink")
        assert mock_validator.should_set_weights() is False

    def test_burn_switch_counts_as_change(self, mock_validator, monkeypatch):
        monkeypatch.setenv("TREASURY_CAP_ENABLED", "1")
        sink = self._ready(mock_validator, block=720, last_update=0)
        mock_validator._last_emitted_recipient = sink
        mock_validator._last_seen_tempo = 720 // 360
        mock_validator.metagraph.hotkeys = [sink, BURN_SINK_HOTKEYS[0], PRIVATE_MINER_HOTKEY]
        TestSetWeights._stub_sink_jwt(mock_validator, sink, BURN_SINK_HOTKEYS[0])
        decision = CapDecision(
            burn=True,
            reason="at_cap",
            active_challenges=1,
            treasury_alpha=400_000.0,
            effective_cap=200_000.0,
        )
        with patch("neurons.validator.decide_treasury_share", return_value=decision):
            # same tempo + cap on + elapsed > epoch_length (720-0)
            assert mock_validator.should_set_weights() is True
            assert (
                mock_validator._planned_weight_targets.recipient
                == BURN_SINK_HOTKEYS[0]
            )

    def test_same_tempo_without_cap_does_not_refresh(self, mock_validator):
        sink = self._ready(mock_validator, block=720, last_update=0)
        mock_validator._last_emitted_recipient = sink
        mock_validator._last_seen_tempo = 720 // 360
        rm = mock_validator.platform_client.request_manager
        assert mock_validator.should_set_weights() is False
        rm.refresh_jwt.assert_not_called()


class TestResyncAndScores:
    def test_resync_returns_early_when_axons_unchanged(self, mock_validator):
        from qbittensor.base.validator import BaseValidatorNeuron

        axon = Mock()
        mock_validator.metagraph.axons = [axon]
        mock_validator.metagraph.hotkeys = ["hk0"]
        mock_validator.hotkeys = ["hk0"]
        mock_validator.scores = np.array([0.5], dtype=np.float32)
        mock_validator.metagraph.sync = Mock()
        BaseValidatorNeuron.resync_metagraph(mock_validator)
        mock_validator.metagraph.sync.assert_called_once()
        assert mock_validator.scores[0] == 0.5
        assert mock_validator.hotkeys == ["hk0"]

    def test_resync_zeros_replaced_hotkey_and_grows_scores(self, mock_validator):
        from qbittensor.base.validator import BaseValidatorNeuron

        mock_validator.metagraph.axons = [Mock(), Mock()]
        mock_validator.metagraph.hotkeys = ["a", "b"]
        mock_validator.hotkeys = ["a", "b"]
        mock_validator.scores = np.array([1.0, 2.0], dtype=np.float32)

        def _sync(**_kwargs):
            mock_validator.metagraph.axons = [Mock(), Mock(), Mock()]
            mock_validator.metagraph.hotkeys = ["a", "c", "d"]
            mock_validator.metagraph.n = 3

        mock_validator.metagraph.sync = Mock(side_effect=_sync)
        BaseValidatorNeuron.resync_metagraph(mock_validator)
        assert mock_validator.scores[0] == 1.0
        assert mock_validator.scores[1] == 0.0
        assert len(mock_validator.scores) == 3
        assert mock_validator.hotkeys == ["a", "c", "d"]

    def test_update_scores_applies_moving_average(self, mock_validator):
        from qbittensor.base.validator import BaseValidatorNeuron

        mock_validator.scores = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_validator.config.neuron.moving_average_alpha = 0.5
        BaseValidatorNeuron.update_scores(mock_validator, np.array([0.0], dtype=np.float32), [0])
        assert mock_validator.scores[0] == pytest.approx(0.5)

    def test_update_scores_empty_is_noop(self, mock_validator):
        from qbittensor.base.validator import BaseValidatorNeuron

        mock_validator.scores = np.array([1.0, 2.0], dtype=np.float32)
        BaseValidatorNeuron.update_scores(mock_validator, np.array([]), [])
        np.testing.assert_array_equal(mock_validator.scores, np.array([1.0, 2.0], dtype=np.float32))

    def test_update_scores_rejects_shape_mismatch(self, mock_validator):
        from qbittensor.base.validator import BaseValidatorNeuron

        mock_validator.scores = np.array([1.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            BaseValidatorNeuron.update_scores(
                mock_validator, np.array([1.0, 2.0]), [0]
            )


class TestValidator:
    """Test cases for the Validator class."""

    def test_forward_heartbeat_sent_when_due(self, mock_validator):
        """Test that heartbeat is recorded when the timer is due."""
        mock_validator.telemetry_service.record_heartbeat = Mock()

        def check_timer():
            mock_validator.telemetry_service.record_heartbeat()

        mock_validator.telemetry_service.heartbeat_timer.check_timer = Mock(side_effect=check_timer)
        mock_validator.forward()

        mock_validator.telemetry_service.heartbeat_timer.check_timer.assert_called_once()
        mock_validator.telemetry_service.record_heartbeat.assert_called_once()

    def test_forward_heartbeat_not_sent_too_soon(self, mock_validator):
        """Test that heartbeat is not recorded when the timer is not due."""
        mock_validator.telemetry_service.record_heartbeat = Mock()

        mock_validator.forward()

        mock_validator.telemetry_service.heartbeat_timer.check_timer.assert_called_once()
        mock_validator.telemetry_service.record_heartbeat.assert_not_called()


class TestMinerQueryThrottling:
    """
    Tests for the throttled miner querying logic introduced to prevent
    blasting every miner on every 5-second forward tick.

    Covers:
    - is_valid_miner_axon filtering (via the public helper)
    - _select_miners_for_this_step batch sizing and staleness behavior
    - Integration of the guard inside the query path
    """

    def _make_good_axon(self):
        axon = Mock()
        axon.is_serving = True
        axon.ip = "203.0.113.10"
        axon.port = 8091
        return axon

    def _make_bad_axon(self, ip="0.0.0.0", port=0, serving=False):
        axon = Mock()
        axon.is_serving = serving
        axon.ip = ip
        axon.port = port
        return axon

    def test_selects_small_batch_under_default_sweep_interval(self, mock_validator):
        """With a 10-minute target sweep, we should only query a small fraction per forward."""
        n = 100
        mock_validator.metagraph.hotkeys = [f"hk_{i}" for i in range(n)]
        mock_validator.metagraph.n = n
        mock_validator.metagraph.axons = [self._make_good_axon() for _ in range(n)]

        # Default sweep is 600s, forward is 5s in the fixture
        selected = mock_validator._select_miners_for_this_step()

        # Expect a small batch (roughly 100 * 5 / 600 ≈ 1, but at least 1)
        assert 1 <= len(selected) <= 5
        assert all(0 <= uid < n for uid in selected)

    def test_very_aggressive_sweep_selects_more_miners(self, mock_validator):
        """Setting a low-but-valid sweep interval should cause larger batches than the default."""
        n = 120
        mock_validator.metagraph.hotkeys = [f"hk_{i}" for i in range(n)]
        mock_validator.metagraph.n = n
        mock_validator.metagraph.axons = [self._make_good_axon() for _ in range(n)]

        # The production code has a safety floor of ~30s for the sweep interval.
        # Use a value just above it to get a meaningfully larger batch than the default 600s case.
        mock_validator.config.neuron.miner_sweep_interval = 45
        mock_validator.config.neuron.forward_sleep_interval = 5

        selected = mock_validator._select_miners_for_this_step()
        # With n=120 and sweep=45, we expect noticeably more than the default tiny batch.
        assert len(selected) >= 5

    def test_filters_out_invalid_axons(self, mock_validator):
        """0.0.0.0 and non-serving axons must never be selected for querying."""
        n = 10
        mock_validator.metagraph.hotkeys = [f"hk_{i}" for i in range(n)]
        mock_validator.metagraph.n = n

        axons = [self._make_good_axon() for _ in range(6)]
        axons += [self._make_bad_axon() for _ in range(4)]  # 4 bad ones
        mock_validator.metagraph.axons = axons

        mock_validator.config.neuron.miner_sweep_interval = 60
        mock_validator.config.neuron.forward_sleep_interval = 5

        selected = mock_validator._select_miners_for_this_step()

        # We should only ever see the 6 good UIDs
        assert all(uid < 6 for uid in selected)
        assert len(selected) <= 6

    def test_staleness_causes_progress_over_multiple_calls(self, mock_validator):
        """Repeated calls should eventually cover different miners (staleness works)."""
        n = 20
        mock_validator.metagraph.hotkeys = [f"hk_{i}" for i in range(n)]
        mock_validator.metagraph.n = n
        mock_validator.metagraph.axons = [self._make_good_axon() for _ in range(n)]

        mock_validator.config.neuron.miner_sweep_interval = 120
        mock_validator.config.neuron.forward_sleep_interval = 5

        first = set(mock_validator._select_miners_for_this_step())
        second = set(mock_validator._select_miners_for_this_step())

        # With only ~1 miner per call, the two sets are likely disjoint
        # or have very small overlap. The important thing is we don't
        # hammer the exact same miner every single call.
        overlap = first & second
        assert len(overlap) <= 2  # very loose; mainly checking we make progress

    def test_last_queried_is_populated_as_side_effect(self, mock_validator):
        """Calling the selector should update last_queried for the chosen UIDs."""
        n = 15
        mock_validator.metagraph.hotkeys = [f"hk_{i}" for i in range(n)]
        mock_validator.metagraph.n = n
        mock_validator.metagraph.axons = [self._make_good_axon() for _ in range(n)]

        mock_validator.config.neuron.miner_sweep_interval = 300
        mock_validator.config.neuron.forward_sleep_interval = 5

        assert len(mock_validator.last_queried) == 0

        selected = mock_validator._select_miners_for_this_step()

        assert len(mock_validator.last_queried) == len(selected)
        assert all(uid in mock_validator.last_queried for uid in selected)

    def test_invalid_axons_are_skipped_in_gather_path(self, mock_validator):
        """
        When _gather_miner_synapses runs, UIDs with invalid axons should
        produce empty sentinel responses without ever calling dendrite.forward.
        """
        # Small metagraph with one good, one bad axon
        mock_validator.metagraph.hotkeys = ["good", "bad"]
        mock_validator.metagraph.n = 2
        good_axon = self._make_good_axon()
        bad_axon = self._make_bad_axon()
        mock_validator.metagraph.axons = [good_axon, bad_axon]

        # Force the selector to consider both (by making the sweep extremely aggressive)
        mock_validator.config.neuron.miner_sweep_interval = 1
        mock_validator.config.neuron.forward_sleep_interval = 1

        # Patch the actual dendrite call so we can count invocations
        with patch.object(mock_validator.dendrite, "forward", new_callable=AsyncMock) as mock_forward:
            responses = mock_validator._run_async(
                mock_validator._gather_miner_synapses(validator_busy=False)
            )

        # We should have exactly 2 responses (full length list)
        assert len(responses) == 2

        # The bad axon (index 1) should have produced a sentinel (no solution_candidate)
        assert responses[1].solution_candidate is None

        # dendrite.forward should only have been called for the good axon
        # (the selector may return 1 or 2, but never the bad UID)
        called_uids = []
        for call in mock_forward.call_args_list:
            axon_arg = call.kwargs.get("axons") or call.args[0]
            # In our code we pass the axon object directly in the throttled path
            if hasattr(axon_arg, "ip"):
                # We can't easily map back without the metagraph, but we can assert
                # that we never passed the bad_axon object.
                assert axon_arg is not bad_axon


class TestTreasuryCapHook:
    """JWT sink vs JWT burn routing inside Validator.set_weights()."""

    OWNER_HK = BURN_SINK_HOTKEYS[0]

    def _enable_cap(self, monkeypatch):
        monkeypatch.setenv("TREASURY_CAP_ENABLED", "1")

    def _run(self, mock_validator, *, decision: CapDecision):
        TestSetWeights._stub_sink_jwt(
            mock_validator, TREASURY_HOTKEY, self.OWNER_HK
        )
        with (
            patch("neurons.validator.decide_treasury_share", return_value=decision),
            patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super,
        ):
            mock_validator.set_weights()
        return mock_super

    def test_burn_routes_share_to_jwt_burn_and_dusts_sink(self, mock_validator, monkeypatch):
        self._enable_cap(monkeypatch)
        mock_validator.metagraph.hotkeys = [TREASURY_HOTKEY, self.OWNER_HK, "miner1"]
        mock_super = self._run(
            mock_validator,
            decision=CapDecision(
                burn=True,
                reason="at_cap",
                active_challenges=1,
                treasury_alpha=400_000.0,
                effective_cap=200_000.0,
            ),
        )

        weights = mock_validator.scores
        owner_uid = mock_validator.metagraph.hotkeys.index(self.OWNER_HK)
        sink_uid = mock_validator.metagraph.hotkeys.index(TREASURY_HOTKEY)
        assert weights[owner_uid] >= 0.99
        assert weights[sink_uid] >= MIN_DUST_FLOOR
        mock_super.assert_called_once()

    def test_below_cap_funds_treasury_and_dusts_burn(self, mock_validator, monkeypatch):
        self._enable_cap(monkeypatch)
        mock_validator.metagraph.hotkeys = [TREASURY_HOTKEY, self.OWNER_HK, "miner1"]
        mock_super = self._run(
            mock_validator,
            decision=CapDecision(
                burn=False,
                reason="below_cap",
                active_challenges=2,
                treasury_alpha=345_000.0,
                effective_cap=400_000.0,
            ),
        )

        weights = mock_validator.scores
        sink_uid = mock_validator.metagraph.hotkeys.index(TREASURY_HOTKEY)
        owner_uid = mock_validator.metagraph.hotkeys.index(self.OWNER_HK)
        assert weights[sink_uid] >= 0.99
        assert weights[owner_uid] >= MIN_DUST_FLOOR
        mock_super.assert_called_once()

    def test_burn_sink_missing_from_metagraph_funds(self, mock_validator, monkeypatch):
        self._enable_cap(monkeypatch)
        mock_validator.metagraph.hotkeys = [TREASURY_HOTKEY, "miner1"]
        mock_super = self._run(
            mock_validator,
            decision=CapDecision(
                burn=True,
                reason="at_cap",
                active_challenges=1,
                treasury_alpha=400_000.0,
                effective_cap=200_000.0,
            ),
        )

        sink_uid = mock_validator.metagraph.hotkeys.index(TREASURY_HOTKEY)
        assert mock_validator.scores[sink_uid] >= 0.99
        mock_super.assert_called_once()

    def test_unknown_count_funds(self, mock_validator, monkeypatch):
        self._enable_cap(monkeypatch)
        mock_validator.metagraph.hotkeys = [TREASURY_HOTKEY, self.OWNER_HK]
        mock_super = self._run(
            mock_validator,
            decision=CapDecision(burn=False, reason="unknown_count"),
        )

        sink_uid = mock_validator.metagraph.hotkeys.index(TREASURY_HOTKEY)
        assert mock_validator.scores[sink_uid] >= 0.99
        mock_super.assert_called_once()

    def test_missing_sink_refuses_even_when_burning(self, mock_validator, monkeypatch):
        self._enable_cap(monkeypatch)
        mock_validator.metagraph.hotkeys = [self.OWNER_HK, "miner1"]
        mock_super = self._run(
            mock_validator,
            decision=CapDecision(
                burn=True,
                reason="at_cap",
                active_challenges=1,
                treasury_alpha=400_000.0,
                effective_cap=200_000.0,
            ),
        )
        mock_super.assert_not_called()

    def test_disabled_cap_never_decides(self, mock_validator, monkeypatch):
        monkeypatch.delenv("TREASURY_CAP_ENABLED", raising=False)
        mock_validator.metagraph.hotkeys = [TREASURY_HOTKEY, "miner1"]
        TestSetWeights._stub_sink_jwt(mock_validator, TREASURY_HOTKEY, self.OWNER_HK)
        with (
            patch("neurons.validator.decide_treasury_share") as mock_decide,
            patch("neurons.validator.BaseValidatorNeuron.set_weights") as mock_super,
        ):
            mock_validator.set_weights()
        mock_decide.assert_not_called()
        sink_uid = mock_validator.metagraph.hotkeys.index(TREASURY_HOTKEY)
        assert mock_validator.scores[sink_uid] >= 0.99
        mock_super.assert_called_once()
