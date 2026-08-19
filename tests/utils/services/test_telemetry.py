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

import queue
from unittest.mock import Mock, patch

import numpy as np
import pytest

from qbittensor.utils.services.telemetry import TelemetryService


@pytest.fixture
def telemetry_service():
    mock_rm = Mock()
    with patch("qbittensor.utils.services.telemetry.RequestManager", return_value=mock_rm):
        with patch.object(TelemetryService, "_start_background_worker"):
            service = TelemetryService(keypair=Mock(), base_url="https://example.com")
    # The service now owns the mock_rm as its request_manager
    return service


class TestTelemetryServiceToPythonScalar:
    def test_none_and_primitives(self, telemetry_service):
        assert telemetry_service._to_python_scalar(None) is None
        assert telemetry_service._to_python_scalar(42) == 42
        assert telemetry_service._to_python_scalar(3.14) == 3.14
        assert telemetry_service._to_python_scalar("hello") == "hello"

    def test_numpy_scalars(self, telemetry_service):
        assert telemetry_service._to_python_scalar(np.int64(7)) == 7
        assert telemetry_service._to_python_scalar(np.float64(1.5)) == 1.5

    def test_object_with_item_method(self, telemetry_service):
        obj = Mock()
        obj.item.return_value = 99
        assert telemetry_service._to_python_scalar(obj) == 99

    def test_fallback_to_str(self, telemetry_service):
        assert telemetry_service._to_python_scalar({"a": 1}) == "{'a': 1}"


class TestTelemetryServiceEnqueue:
    def test_enqueue_numeric_datapoint(self, telemetry_service):
        ok = telemetry_service._enqueue_datapoint("cpu_usage", "2026-01-01T00:00:00Z", 42.0)
        assert ok is True
        item = telemetry_service.queue.get_nowait()
        assert item["type"] == "cpu_usage"
        assert item["value"] == 42.0

    def test_enqueue_string_datapoint(self, telemetry_service):
        ok = telemetry_service._enqueue_datapoint("heartbeat_version", "2026-01-01T00:00:00Z", "1.0.0")
        assert ok is True
        item = telemetry_service.queue.get_nowait()
        assert item["value"] == "1.0.0"

    def test_enqueue_drops_when_queue_full(self, telemetry_service):
        telemetry_service.max_queue_size = 1
        telemetry_service.queue = queue.Queue(maxsize=1)
        telemetry_service.queue.put_nowait({"type": "x", "timestamp": "t", "value": 1})
        ok = telemetry_service._enqueue_datapoint("overflow", "2026-01-01T00:00:00Z", 1.0)
        assert ok is False


class TestTelemetryServiceFlushBatch:
    def test_flush_batch_posts_datapoints(self, telemetry_service):
        telemetry_service.queue.task_done = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        telemetry_service.request_manager.post = Mock(return_value=mock_response)

        batch = [
            {
                "type": "heartbeat_version",
                "timestamp": "2026-01-01T00:00:00Z",
                "value": "1.2.3",
                "miner_uid": None,
                "miner_hotkey": None,
                "attributes": None,
            }
        ]
        telemetry_service._flush_batch(batch)

        telemetry_service.request_manager.post.assert_called_once()
        call_kwargs = telemetry_service.request_manager.post.call_args.kwargs
        assert call_kwargs["json"]["datapoints"][0]["string_value"] == "1.2.3"
        assert call_kwargs["json"]["datapoints"][0]["type"] == "heartbeat_version"
        assert call_kwargs.get("additional_headers") is not None


class TestCpuGpuLookupHelpers:
    def test_get_cpu_model_returns_something_reasonable(self):
        from qbittensor.utils.services.telemetry import _get_cpu_model
        model = _get_cpu_model()
        assert isinstance(model, str)
        assert len(model) > 0
        # Should not be the bare arch that was previously reported as "family"
        assert model.lower() not in {"x86_64", "amd64", "i386", "unknown"}

    def test_get_gpu_info_cpu_path(self):
        from qbittensor.utils.services.telemetry import _get_gpu_info
        count, models = _get_gpu_info("cpu")
        assert count == 0
        assert models == "none"

    def test_get_gpu_info_nonexistent_cuda(self):
        from qbittensor.utils.services.telemetry import _get_gpu_info
        # Should not explode; on this env will be nvidia-smi not found or error
        count, models = _get_gpu_info("cuda:99")
        assert isinstance(count, int)
        assert count >= 0
        assert isinstance(models, str)
        assert len(models) > 0

    def test_get_nvidia_driver_info_returns_strings(self):
        from qbittensor.utils.services.telemetry import _get_nvidia_driver_info
        driver, cuda = _get_nvidia_driver_info()
        assert isinstance(driver, str)
        assert isinstance(cuda, str)
        # On this machine (no NVIDIA) we expect "none" or a value; either is acceptable.
        assert len(driver) > 0
        assert len(cuda) > 0

    def test_get_docker_versions_parses_format_output(self):
        from qbittensor.utils.services import telemetry as tel

        class Result:
            returncode = 0
            stdout = "28.3.3\t28.3.2\n"
            stderr = ""

        with patch.object(tel.subprocess, "run", return_value=Result()):
            client, server = tel._get_docker_versions()
        assert client == "28.3.3"
        assert server == "28.3.2"

    def test_get_docker_versions_ignores_missing_server(self):
        from qbittensor.utils.services import telemetry as tel

        class Result:
            returncode = 1
            stdout = "28.3.3\t<no value>\n"
            stderr = "Cannot connect to the Docker daemon"

        with patch.object(tel.subprocess, "run", return_value=Result()):
            client, server = tel._get_docker_versions()
        assert client == "28.3.3"
        assert server == ""

    def test_get_docker_versions_falls_back_to_docker_dash_dash_version(self):
        from qbittensor.utils.services import telemetry as tel

        class EmptyFormat:
            returncode = 1
            stdout = ""
            stderr = "unknown flag"

        class Plain:
            returncode = 0
            stdout = "Docker version 27.5.1, build abc123\n"
            stderr = ""

        with patch.object(tel.subprocess, "run", side_effect=[EmptyFormat(), Plain()]):
            client, server = tel._get_docker_versions()
        assert client == "27.5.1"
        assert server == ""

    def test_get_docker_versions_raises_when_cli_missing(self):
        from qbittensor.utils.services import telemetry as tel

        with patch.object(tel.subprocess, "run", side_effect=FileNotFoundError("docker")):
            with pytest.raises(RuntimeError, match="docker CLI not found"):
                tel._get_docker_versions()


class TestGpuMetricHelpers:
    def test_decode_throttle_reasons_ignores_idle(self):
        from qbittensor.utils.services.telemetry import _decode_throttle_reasons

        assert _decode_throttle_reasons(0) == "none"
        assert _decode_throttle_reasons(0x1) == "none"  # gpu idle
        assert _decode_throttle_reasons(0x2) == "none"  # applications clocks
        assert _decode_throttle_reasons(0x4) == "sw_power_cap"
        assert _decode_throttle_reasons(0x4 | 0x20) == "sw_power_cap,sw_thermal_slowdown"
        assert _decode_throttle_reasons(0x1 | 0x8) == "hw_slowdown"

    def test_selected_gpu_indices(self):
        from qbittensor.utils.services.telemetry import _selected_gpu_indices

        assert _selected_gpu_indices("cpu", 2) == []
        assert _selected_gpu_indices("cuda:1", 2) == [1]
        assert _selected_gpu_indices("cuda:99", 2) == []
        assert _selected_gpu_indices("cuda", 2) == [0, 1]
        assert _selected_gpu_indices("cuda:0", 0) == []

    def test_get_gpu_static_info_cpu_path(self):
        from qbittensor.utils.services.telemetry import _get_gpu_static_info

        assert _get_gpu_static_info("cpu") == []

    def test_get_gpu_static_info_survives_per_field_failure(self):
        from qbittensor.utils.services import telemetry as tel

        handle = object()
        mem = Mock(total=96 * 1024 * 1024 * 1024)

        with patch.object(tel, "NVML_AVAILABLE", True), \
             patch.object(tel, "nvmlInit"), \
             patch.object(tel, "nvmlShutdown"), \
             patch.object(tel, "nvmlDeviceGetCount", return_value=1), \
             patch.object(tel, "nvmlDeviceGetHandleByIndex", return_value=handle), \
             patch.object(tel, "nvmlDeviceGetMemoryInfo", return_value=mem), \
             patch.object(tel, "nvmlDeviceGetCudaComputeCapability", side_effect=RuntimeError("no cc")), \
             patch.object(tel, "nvmlDeviceGetPowerManagementLimit", return_value=600_000), \
             patch.object(tel, "nvmlDeviceGetPowerManagementDefaultLimit", side_effect=RuntimeError("no default")), \
             patch.object(tel, "nvmlDeviceGetPowerManagementLimitConstraints", return_value=[300_000, 600_000]):
            infos = tel._get_gpu_static_info("cuda")

        assert len(infos) == 1
        assert infos[0].memory_total_bytes == 96 * 1024 * 1024 * 1024
        assert infos[0].compute_capability is None
        assert infos[0].power_limit_watts == 600.0
        assert infos[0].power_default_limit_watts is None
        assert infos[0].power_max_limit_watts == 600.0

    def test_get_gpu_static_info_raises_on_init_failure(self):
        from qbittensor.utils.services import telemetry as tel

        with patch.object(tel, "NVML_AVAILABLE", True), \
             patch.object(tel, "nvmlInit", side_effect=RuntimeError("libnvidia-ml.so")):
            with pytest.raises(RuntimeError, match="pynvml init failed"):
                tel._get_gpu_static_info("cuda")

    def test_get_gpu_runtime_info_empty_without_indices(self):
        from qbittensor.utils.services.telemetry import _get_gpu_runtime_info

        assert _get_gpu_runtime_info([]) == []

    def test_get_gpu_runtime_info_raises_when_all_handles_fail(self):
        from qbittensor.utils.services import telemetry as tel

        with patch.object(tel, "NVML_AVAILABLE", True), \
             patch.object(tel, "nvmlDeviceGetHandleByIndex", side_effect=RuntimeError("no handle")):
            with pytest.raises(RuntimeError, match="no runtime metrics"):
                tel._get_gpu_runtime_info([0])

    def test_get_gpu_runtime_info_reads_power_temp_throttle(self):
        from qbittensor.utils.services import telemetry as tel

        handle = object()
        mem = Mock(total=100, used=25)
        with patch.object(tel, "NVML_AVAILABLE", True), \
             patch.object(tel, "nvmlDeviceGetHandleByIndex", return_value=handle), \
             patch.object(tel, "nvmlDeviceGetUtilizationRates", return_value=Mock(gpu=80)), \
             patch.object(tel, "nvmlDeviceGetMemoryInfo", return_value=mem), \
             patch.object(tel, "nvmlDeviceGetPowerUsage", return_value=412_500), \
             patch.object(tel, "nvmlDeviceGetPowerManagementLimit", return_value=600_000), \
             patch.object(tel, "nvmlDeviceGetTemperature", return_value=62), \
             patch.object(tel, "nvmlDeviceGetCurrentClocksEventReasons", return_value=0x4):
            infos = tel._get_gpu_runtime_info([0])

        assert len(infos) == 1
        assert infos[0].utilization == 80.0
        assert infos[0].memory_usage_percent == 25.0
        assert infos[0].power_draw_watts == 412.5
        assert infos[0].power_limit_watts == 600.0
        assert infos[0].temperature_c == 62.0
        assert infos[0].throttle_reasons == "sw_power_cap"

    def test_discover_gpu_indices_cpu(self):
        from qbittensor.utils.services.telemetry import _discover_gpu_indices

        assert _discover_gpu_indices("cpu") == []


class TestGpuTelemetryRecording:
    def test_startup_enqueues_per_gpu_capability(self, telemetry_service):
        from qbittensor.utils.services import telemetry as tel

        info = tel.GpuStaticInfo(
            index=0,
            memory_total_bytes=96 * 1024 * 1024 * 1024,
            compute_capability="10.0",
            power_limit_watts=600.0,
            power_default_limit_watts=600.0,
            power_max_limit_watts=600.0,
        )
        vm = Mock(total=96 * 1024 * 1024 * 1024)
        disk = Mock(total=2 * 1024 * 1024 * 1024 * 1024)
        with patch.object(tel, "_get_cpu_model", return_value="Test CPU"), \
             patch.object(tel.psutil, "cpu_count", return_value=26), \
             patch.object(tel.psutil, "virtual_memory", return_value=vm), \
             patch.object(tel.psutil, "disk_usage", return_value=disk), \
             patch.object(tel, "_get_gpu_info", return_value=(1, "NVIDIA RTX PRO 6000")), \
             patch.object(tel, "_get_nvidia_driver_info", return_value=("570.00", "12.8")), \
             patch.object(tel, "_get_docker_versions", return_value=("28.3.3", "28.3.3")), \
             patch.object(tel, "_get_gpu_static_info", return_value=[info]), \
             patch.object(tel, "_discover_gpu_indices", return_value=[0]), \
             patch.object(tel, "NVML_AVAILABLE", False):
            telemetry_service.device = "cuda"
            telemetry_service.record_startup_metrics()

        items = []
        while not telemetry_service.queue.empty():
            items.append(telemetry_service.queue.get_nowait())
        by_type = {item["type"]: item for item in items}

        assert by_type["system_gpu_memory_bytes"]["value"] == float(96 * 1024 * 1024 * 1024)
        assert by_type["system_gpu_memory_bytes"]["attributes"]["gpu_index"] == 0
        assert by_type["system_gpu_compute_capability"]["value"] == "10.0"
        assert by_type["system_gpu_power_limit_watts"]["value"] == 600.0
        assert by_type["system_gpu_power_default_limit_watts"]["value"] == 600.0
        assert by_type["system_gpu_power_max_limit_watts"]["value"] == 600.0
        assert by_type["system_docker_version"]["value"] == "28.3.3"
        assert by_type["system_docker_server_version"]["value"] == "28.3.3"
        assert by_type["system_gpu_driver_version"]["value"] == "570.00"
        assert by_type["system_cuda_version"]["value"] == "12.8"
        assert telemetry_service.gpu_indices == [0]

    def test_startup_skips_missing_static_fields(self, telemetry_service):
        from qbittensor.utils.services import telemetry as tel

        info = tel.GpuStaticInfo(index=0, memory_total_bytes=1024, compute_capability=None)
        vm = Mock(total=1)
        disk = Mock(total=1)
        with patch.object(tel, "_get_cpu_model", return_value="cpu"), \
             patch.object(tel.psutil, "cpu_count", return_value=1), \
             patch.object(tel.psutil, "virtual_memory", return_value=vm), \
             patch.object(tel.psutil, "disk_usage", return_value=disk), \
             patch.object(tel, "_get_gpu_info", return_value=(1, "gpu")), \
             patch.object(tel, "_get_nvidia_driver_info", return_value=("1.0", "12.0")), \
             patch.object(tel, "_get_docker_versions", return_value=("28.0.0", "")), \
             patch.object(tel, "_get_gpu_static_info", return_value=[info]), \
             patch.object(tel, "NVML_AVAILABLE", False):
            telemetry_service.device = "cuda"
            telemetry_service.record_startup_metrics()

        types = {item["type"] for item in _drain(telemetry_service)}
        assert "system_gpu_memory_bytes" in types
        assert "system_gpu_compute_capability" not in types
        assert "system_gpu_power_limit_watts" not in types

    def test_system_metrics_enqueues_power_temp_throttle(self, telemetry_service):
        from qbittensor.utils.services import telemetry as tel

        info = tel.GpuRuntimeInfo(
            index=0,
            utilization=80.0,
            memory_usage_percent=50.0,
            power_draw_watts=412.5,
            power_limit_watts=600.0,
            temperature_c=62.0,
            throttle_reasons="sw_power_cap",
        )
        vm = Mock(percent=10.0)
        disk = Mock(percent=20.0)
        telemetry_service.gpu_indices = [0]
        telemetry_service._pynvml_initialized = True
        with patch.object(tel.psutil, "cpu_percent", return_value=5.0), \
             patch.object(tel.psutil, "virtual_memory", return_value=vm), \
             patch.object(tel.psutil, "disk_usage", return_value=disk), \
             patch.object(tel, "_get_docker_versions", return_value=("28.3.3", "28.3.3")), \
             patch.object(tel, "_get_nvidia_driver_info", return_value=("570.00", "12.8")), \
             patch.object(tel, "_get_gpu_runtime_info", return_value=[info]) as runtime:
            telemetry_service.record_system_metrics()

        runtime.assert_called_once_with([0])
        by_type = {item["type"]: item for item in _drain(telemetry_service)}
        assert by_type["system_gpu_utilization"]["value"] == 80.0
        assert by_type["system_gpu_memory_usage"]["value"] == 50.0
        assert by_type["system_gpu_power_draw_watts"]["value"] == 412.5
        assert by_type["system_gpu_power_limit_watts"]["value"] == 600.0
        assert by_type["system_gpu_temperature_c"]["value"] == 62.0
        assert by_type["system_gpu_throttle_reasons"]["value"] == "sw_power_cap"
        assert by_type["system_gpu_throttle_reasons"]["attributes"]["gpu_index"] == 0
        assert by_type["system_docker_version"]["value"] == "28.3.3"
        assert by_type["system_gpu_driver_version"]["value"] == "570.00"
        assert by_type["system_cuda_version"]["value"] == "12.8"

    def test_startup_collection_error_does_not_block_other_collectors(self, telemetry_service):
        from qbittensor.utils.services import telemetry as tel

        vm = Mock(total=1)
        with patch.object(tel, "_get_cpu_model", return_value="Test CPU"), \
             patch.object(tel.psutil, "cpu_count", return_value=8), \
             patch.object(tel.psutil, "virtual_memory", return_value=vm), \
             patch.object(tel.psutil, "disk_usage", side_effect=OSError("disk dead")), \
             patch.object(tel, "_get_gpu_info", return_value=(1, "NVIDIA RTX PRO 6000")), \
             patch.object(tel, "_get_nvidia_driver_info", return_value=("570.00", "12.8")), \
             patch.object(tel, "_get_docker_versions", side_effect=RuntimeError("docker CLI not found")), \
             patch.object(tel, "_get_gpu_static_info", return_value=[]), \
             patch.object(tel, "_discover_gpu_indices", return_value=[]), \
             patch.object(tel, "NVML_AVAILABLE", False):
            telemetry_service.device = "cuda"
            telemetry_service.record_startup_metrics()

        items = _drain(telemetry_service)
        by_type = {}
        errors = []
        for item in items:
            if item["type"] == "system_collection_error":
                errors.append(item)
            else:
                by_type[item["type"]] = item

        assert by_type["system_cpu_family"]["value"] == "Test CPU"
        assert by_type["system_gpu_models"]["value"] == "NVIDIA RTX PRO 6000"
        assert by_type["system_gpu_driver_version"]["value"] == "570.00"
        assert "system_disk_bytes" not in by_type
        assert "system_docker_version" not in by_type
        collectors = {item["value"] for item in errors}
        assert "disk" in collectors
        assert "docker_version" in collectors
        docker_err = next(item for item in errors if item["value"] == "docker_version")
        assert "docker CLI not found" in docker_err["attributes"]["error"]

    def test_system_metrics_survives_gpu_runtime_failure(self, telemetry_service):
        from qbittensor.utils.services import telemetry as tel

        vm = Mock(percent=10.0)
        disk = Mock(percent=20.0)
        telemetry_service.device = "cuda"
        telemetry_service.gpu_indices = [0]
        telemetry_service._pynvml_initialized = True
        with patch.object(tel.psutil, "cpu_percent", return_value=5.0), \
             patch.object(tel.psutil, "virtual_memory", return_value=vm), \
             patch.object(tel.psutil, "disk_usage", return_value=disk), \
             patch.object(tel, "_get_docker_versions", return_value=("28.3.3", "")), \
             patch.object(tel, "_get_nvidia_driver_info", return_value=("none", "none")), \
             patch.object(tel, "_get_gpu_runtime_info", side_effect=RuntimeError("nvml exploded")):
            telemetry_service.record_system_metrics()

        items = _drain(telemetry_service)
        by_type = {item["type"]: item for item in items if item["type"] != "system_collection_error"}
        errors = [item for item in items if item["type"] == "system_collection_error"]
        assert by_type["system_cpu_usage"]["value"] == 5.0
        assert by_type["system_docker_version"]["value"] == "28.3.3"
        collectors = {item["value"] for item in errors}
        assert "gpu_runtime" in collectors
        assert "nvidia_driver" in collectors


def _drain(service):
    items = []
    while not service.queue.empty():
        items.append(service.queue.get_nowait())
    return items
