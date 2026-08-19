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

import time
import bittensor as bt
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
import queue
import threading
import os
import psutil
import platform
import re
import subprocess

from qbittensor.utils.request.request_manager import RequestManager
from qbittensor.utils.timer import Timer
import qbittensor
from qbittensor.utils.time import timestamp_iso

try:
    from pynvml import *
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    bt.logging.warning("pynvml not available, GPU metrics will be skipped")


def _get_cpu_model() -> str:
    """Best-effort human readable CPU model / family.

    platform.processor() often returns bare arch (e.g. "x86_64") on Linux/macOS.
    We try /proc/cpuinfo, sysctl, wmic, then sensible fallbacks.
    """
    try:
        system = platform.system().lower()
        if system == "linux":
            try:
                with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if "model name" in line.lower():
                            val = line.split(":", 1)[1].strip()
                            if val:
                                return val
            except Exception:
                pass
        elif system == "darwin":
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                ).decode("utf-8", errors="ignore").strip()
                if out:
                    return out
            except Exception:
                pass
        elif system == "windows":
            try:
                out = subprocess.check_output(
                    ["wmic", "cpu", "get", "name", "/value"],
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).decode("utf-8", errors="ignore")
                for line in out.splitlines():
                    if line.lower().startswith("name="):
                        val = line.split("=", 1)[1].strip()
                        if val:
                            return val
            except Exception:
                pass

        # Better-than-nothing fallbacks (avoid returning bare "x86_64" etc as "family")
        proc = (platform.processor() or "").strip()
        bad_archs = {"", "x86_64", "amd64", "i386", "i686", "arm64", "aarch64", "unknown"}
        if proc and proc.lower() not in bad_archs:
            return proc

        mach = (platform.machine() or "").strip()
        if mach and mach.lower() not in bad_archs:
            return mach

        return platform.platform(aliased=True, terse=True) or "unknown"
    except Exception:
        return "unknown"


def _get_gpu_info(device: str) -> tuple[int, str]:
    """Return (gpu_count, models_str) using pynvml when it works at runtime,
    with nvidia-smi subprocess fallback (more reliable in some envs).

    This is *only* for the startup model/count strings.
    Periodic utilization still requires working pynvml + self.gpu_indices.
    """
    if device == "cpu":
        return 0, "none"

    # Try pynvml first (for consistency with periodic metrics path)
    if NVML_AVAILABLE:
        try:
            nvmlInit()
            try:
                device_count = nvmlDeviceGetCount()
                if device.startswith("cuda:"):
                    try:
                        gpu_index = int(device.split(":", 1)[1])
                        if 0 <= gpu_index < device_count:
                            handle = nvmlDeviceGetHandleByIndex(gpu_index)
                            name = nvmlDeviceGetName(handle)
                            if isinstance(name, (bytes, bytearray)):
                                name = name.decode("utf-8", errors="ignore")
                            return 1, str(name).strip()
                        else:
                            return 0, "invalid device"
                    except Exception as e:
                        bt.logging.warning(f"Single-GPU pynvml query failed for {device}: {e}")
                        # fall through to nvidia-smi
                else:
                    # "cuda" or other -> report all
                    gpu_models_list = []
                    for i in range(device_count):
                        handle = nvmlDeviceGetHandleByIndex(i)
                        name = nvmlDeviceGetName(handle)
                        if isinstance(name, (bytes, bytearray)):
                            name = name.decode("utf-8", errors="ignore")
                        gpu_models_list.append(str(name).strip())
                    models = ", ".join(gpu_models_list) if gpu_models_list else "none"
                    return device_count, models
            finally:
                try:
                    nvmlShutdown()
                except Exception:
                    pass
        except Exception as e:
            bt.logging.warning(f"pynvml runtime init/query failed ({e}); trying nvidia-smi fallback for models")

    # Fallback to nvidia-smi (doesn't require pynvml python package or its .so quirks)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = [line.strip() for line in (result.stdout or "").strip().splitlines() if line.strip()]
            if not lines:
                return 0, "none"
            if device.startswith("cuda:"):
                try:
                    idx = int(device.split(":", 1)[1])
                    if 0 <= idx < len(lines):
                        return 1, lines[idx]
                    else:
                        return 0, "invalid device"
                except Exception:
                    return 0, "error"
            else:
                # all GPUs
                return len(lines), ", ".join(lines)
        else:
            bt.logging.warning(f"nvidia-smi query failed (rc={result.returncode}): {result.stderr[:200] if result.stderr else ''}")
    except FileNotFoundError:
        return 0, "nvidia-smi not found"
    except Exception as e:
        bt.logging.warning(f"nvidia-smi fallback error: {e}")

    return 0, "error"


def _get_nvidia_driver_info() -> tuple[str, str]:
    """Return (driver_version, cuda_version) best-effort from the host.

    Primary source: `nvidia-smi` banner line (contains both "Driver Version: X.Y.Z"
    and "CUDA Version: A.B" -- the latter is the max CUDA runtime the driver supports).

    Fallbacks:
    - pynvml nvmlSystemGetDriverVersion() for driver only.
    - Returns ("none", "none") when no NVIDIA stack is present or detectable.

    This is reported via telemetry at startup so we can track what CUDA/driver
    combinations validators are actually running (required for --gpus passthrough
    of miner solution containers and for the GPU smoke test).
    """
    driver = ""
    cuda = ""

    # Best signal: the nvidia-smi banner (works even without per-gpu queries)
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.splitlines():
                if "Driver Version" in line and "CUDA Version" in line:
                    dm = re.search(r"Driver Version:\s*([0-9.]+)", line)
                    cm = re.search(r"CUDA Version:\s*([0-9.]+)", line)
                    if dm:
                        driver = dm.group(1)
                    if cm:
                        cuda = cm.group(1)
                    break
            # Secondary: explicit driver query (some nvidia-smi builds vary in banner)
            if not driver:
                qres = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if qres.returncode == 0:
                    lines = [line.strip() for line in (qres.stdout or "").strip().splitlines() if line.strip()]
                    if lines:
                        driver = lines[0]
    except FileNotFoundError:
        pass
    except Exception as e:
        bt.logging.debug(f"nvidia-smi version parse error: {e}")

    # pynvml driver fallback (no CUDA version from pynvml AFAIK)
    if not driver and NVML_AVAILABLE:
        try:
            nvmlInit()
            try:
                drv = nvmlSystemGetDriverVersion()
                if isinstance(drv, (bytes, bytearray)):
                    driver = drv.decode("utf-8", errors="ignore").strip()
                elif drv:
                    driver = str(drv).strip()
            finally:
                try:
                    nvmlShutdown()
                except Exception:
                    pass
        except Exception as e:
            bt.logging.debug(f"pynvml driver version query failed: {e}")

    return (driver or "none"), (cuda or "none")


_DOCKER_VERSION_RE = re.compile(r"Docker version\s+(\S+)", re.IGNORECASE)
_DOCKER_TEMPLATE_NA = {"", "<no value>", "<nil>", "<none>", "n/a"}


def _clean_docker_version(raw: Optional[str]) -> str:
    token = (raw or "").strip()
    if token.lower() in _DOCKER_TEMPLATE_NA:
        return ""
    return token


def _get_docker_versions() -> tuple[str, str]:
    """Return (client_version, server_version).

    Raises if the Docker CLI is missing, times out, or produces no client version.
    Server may be empty when the CLI works but the daemon is down — that is not
    a hard failure; the caller still reports the client.
    """
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Client.Version}}\t{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError as e:
        raise RuntimeError("docker CLI not found in PATH") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("docker version timed out") from e
    except Exception as e:
        raise RuntimeError(f"docker version failed: {e}") from e

    line = (result.stdout or "").strip()
    if "\t" in line:
        client_raw, server_raw = line.split("\t", 1)
    else:
        client_raw, server_raw = line, ""
    client = _clean_docker_version(client_raw)
    server = _clean_docker_version(server_raw)
    if client:
        return client, server

    # Older CLIs may not support `docker version --format`.
    try:
        fallback = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as e:
        raise RuntimeError(f"docker --version failed: {e}") from e
    match = _DOCKER_VERSION_RE.search(fallback.stdout or "")
    if match:
        return match.group(1).rstrip(","), ""

    err = (result.stderr or fallback.stderr or "").strip()
    raise RuntimeError(
        f"could not read docker version (rc={result.returncode}): {err[:200] or 'empty output'}"
    )


# NVML milliwatts -> watts. NVML_TEMPERATURE_GPU is 0; hardcoded so this module
# still imports when pynvml is missing.
_MW_PER_WATT = 1000.0
_NVML_TEMPERATURE_GPU = 0

# Only reasons that indicate the GPU is being held back. Idle / app-clock /
# display bits are normal and would spam "gpu_idle" between solution runs.
_THROTTLE_REASON_BITS = (
    ("sw_power_cap", 0x0000000000000004),
    ("hw_slowdown", 0x0000000000000008),
    ("sw_thermal_slowdown", 0x0000000000000020),
    ("hw_thermal_slowdown", 0x0000000000000040),
    ("hw_power_brake_slowdown", 0x0000000000000080),
)


@dataclass
class GpuStaticInfo:
    """Per-GPU capability snapshot (startup). Missing fields stay None."""

    index: int
    memory_total_bytes: Optional[int] = None
    compute_capability: Optional[str] = None
    power_limit_watts: Optional[float] = None
    power_default_limit_watts: Optional[float] = None
    power_max_limit_watts: Optional[float] = None


@dataclass
class GpuRuntimeInfo:
    """Per-GPU live snapshot (periodic). Missing fields stay None."""

    index: int
    utilization: Optional[float] = None
    memory_usage_percent: Optional[float] = None
    power_draw_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None
    temperature_c: Optional[float] = None
    throttle_reasons: Optional[str] = None


def _selected_gpu_indices(device: str, device_count: int) -> list[int]:
    """Apply --neuron.device filtering: cpu -> [], cuda:N -> [N], else all."""
    if device == "cpu" or device_count <= 0:
        return []
    if device.startswith("cuda:"):
        try:
            idx = int(device.split(":", 1)[1])
        except ValueError:
            return []
        if 0 <= idx < device_count:
            return [idx]
        return []
    return list(range(device_count))


def _mw_to_watts(milliwatts: Any) -> Optional[float]:
    try:
        return float(milliwatts) / _MW_PER_WATT
    except (TypeError, ValueError):
        return None


def _decode_throttle_reasons(mask: int) -> str:
    """Return comma-separated active limiter names, or 'none'.

    Ignores idle / application-clocks / display bits so an idle card does not
    look throttled.
    """
    names = [name for name, bit in _THROTTLE_REASON_BITS if mask & bit]
    return ",".join(names) if names else "none"


def _get_gpu_static_info(device: str) -> list[GpuStaticInfo]:
    """Per-GPU VRAM / compute cap / power limits via NVML.

    Only safe to call before TelemetryService's long-lived nvmlInit(): this path
    does its own init/shutdown, matching _get_gpu_info.

    Raises on NVML init/query failure so the caller can record a collection error.
    Per-field NVML gaps stay None (the GPU may not expose that sensor).
    """
    if not NVML_AVAILABLE or device == "cpu":
        return []
    try:
        nvmlInit()
    except Exception as e:
        raise RuntimeError(f"pynvml init failed: {e}") from e
    try:
        device_count = nvmlDeviceGetCount()
        results: list[GpuStaticInfo] = []
        handle_errors: list[str] = []
        for i in _selected_gpu_indices(device, device_count):
            try:
                handle = nvmlDeviceGetHandleByIndex(i)
            except Exception as e:
                handle_errors.append(f"gpu {i}: {e}")
                continue
            info = GpuStaticInfo(index=i)
            try:
                info.memory_total_bytes = int(nvmlDeviceGetMemoryInfo(handle).total)
            except Exception:
                pass
            try:
                major, minor = nvmlDeviceGetCudaComputeCapability(handle)
                info.compute_capability = f"{int(major)}.{int(minor)}"
            except Exception:
                pass
            try:
                info.power_limit_watts = _mw_to_watts(nvmlDeviceGetPowerManagementLimit(handle))
            except Exception:
                pass
            try:
                info.power_default_limit_watts = _mw_to_watts(
                    nvmlDeviceGetPowerManagementDefaultLimit(handle)
                )
            except Exception:
                pass
            try:
                constraints = nvmlDeviceGetPowerManagementLimitConstraints(handle)
                info.power_max_limit_watts = _mw_to_watts(constraints[1])
            except Exception:
                pass
            results.append(info)
        if not results and handle_errors:
            raise RuntimeError(
                "pynvml returned no GPU capability snapshot: " + "; ".join(handle_errors)
            )
        return results
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"pynvml static GPU query failed: {e}") from e
    finally:
        try:
            nvmlShutdown()
        except Exception:
            pass


def _get_gpu_runtime_info(indices: list[int]) -> list[GpuRuntimeInfo]:
    """Per-GPU util / memory / power / temp / throttle via an already-initialized NVML.

    Raises on total failure. Per-field NVML gaps stay None.
    """
    if not NVML_AVAILABLE or not indices:
        return []
    try:
        results: list[GpuRuntimeInfo] = []
        handle_errors: list[str] = []
        for i in indices:
            try:
                handle = nvmlDeviceGetHandleByIndex(i)
            except Exception as e:
                handle_errors.append(f"gpu {i}: {e}")
                continue
            info = GpuRuntimeInfo(index=i)
            try:
                info.utilization = float(nvmlDeviceGetUtilizationRates(handle).gpu)
            except Exception:
                pass
            try:
                mem = nvmlDeviceGetMemoryInfo(handle)
                if mem.total:
                    info.memory_usage_percent = (mem.used / mem.total) * 100.0
            except Exception:
                pass
            try:
                info.power_draw_watts = _mw_to_watts(nvmlDeviceGetPowerUsage(handle))
            except Exception:
                pass
            try:
                info.power_limit_watts = _mw_to_watts(nvmlDeviceGetPowerManagementLimit(handle))
            except Exception:
                pass
            try:
                info.temperature_c = float(
                    nvmlDeviceGetTemperature(handle, _NVML_TEMPERATURE_GPU)
                )
            except Exception:
                pass
            try:
                try:
                    mask = nvmlDeviceGetCurrentClocksEventReasons(handle)
                except Exception:
                    mask = nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                info.throttle_reasons = _decode_throttle_reasons(int(mask))
            except Exception:
                pass
            results.append(info)
        if not results and handle_errors:
            raise RuntimeError(
                "pynvml returned no runtime metrics: " + "; ".join(handle_errors)
            )
        return results
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"pynvml runtime query failed: {e}") from e


def _discover_gpu_indices(device: str) -> list[int]:
    """Indices matching --neuron.device, via NVML."""
    if not NVML_AVAILABLE or device == "cpu":
        return []
    try:
        nvmlInit()
        try:
            return _selected_gpu_indices(device, nvmlDeviceGetCount())
        finally:
            try:
                nvmlShutdown()
            except Exception:
                pass
    except Exception as e:
        bt.logging.warning(f"Could not populate GPU indices for periodic metrics (pynvml): {e}")
        return []


class TelemetryService:
    def __init__(
        self,
        device: str = "cpu",
        export_interval_millis=5000,
        max_queue_size=1000,
        batch_size=10,
        retry_attempts=3,
        retry_delay=1,
        service_name: Optional[str] = None,
        network: Optional[str] = None,
        *,
        keypair: Optional[Any] = None,
        base_url: Optional[str] = None,
        tensorauth_url: Optional[str] = None,
        netuid: Optional[int] = None,
    ):
        """
        Telemetry / metrics service for validators and miners.

        Pass keypair + base_url (telemetry) + tensorauth_url + netuid
        and the service will create its own RequestManager (one RM per client).
        """
        if keypair is not None and base_url is not None:
            self.request_manager = RequestManager(
                keypair,
                base_url=base_url,
                tensorauth_url=tensorauth_url,
                netuid=netuid,
            )
        else:
            raise ValueError("TelemetryService requires keypair + base_url")
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.flush_interval = export_interval_millis / 1000.0
        self.device = device
        self.gpu_indices = []
        self._pynvml_initialized = False
        self.queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker_thread = None
        self.heartbeat_timer = Timer(timeout=timedelta(minutes=5), run=self.record_heartbeat, run_on_start=True)
        self.system_metrics_timer = Timer(timeout=timedelta(minutes=5), run=self.record_system_metrics, run_on_start=True)

        self._service_name = service_name
        self._network = network

        bt.logging.info("TelemetryService initialized (using RequestManager for all API calls)")
        self._start_background_worker()

    def _to_python_scalar(self, x: Any) -> Any:
        """Convert NumPy or Torch scalars to JSON-serializable Python types."""
        if x is None:
            return None
        if isinstance(x, (int, float, str)):
            return x
        if hasattr(x, 'item'):  # Handles torch.Tensor scalars and NumPy arrays
            return x.item()
        if isinstance(x, (np.integer, np.floating, np.number)):
            return x.item()
        return str(x)  # Fallback for other types

    def _start_background_worker(self):
        """Start the background thread for flushing the queue."""
        def worker():
            while not self._stop_event.is_set():
                try:
                    # Flush every interval or when batch_size reached
                    start_time = time.time()
                    batch = []
                    while len(batch) < self.batch_size and not self._stop_event.is_set():
                        try:
                            item = self.queue.get(timeout=0.1)
                            batch.append(item)
                        except queue.Empty:
                            break
                    if batch:
                        self._flush_batch(batch)
                    sleep_time = max(0, self.flush_interval - (time.time() - start_time))
                    if sleep_time > 0:
                        self._stop_event.wait(sleep_time)
                except Exception as e:
                    bt.logging.error(f"Background worker error: {e}")
                    time.sleep(self.retry_delay)

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _flush_batch(self, batch: list[Dict[str, Any]]) -> None:
        """Flush a batch of datapoints via the internal RequestManager (single POST to /v1/datapoints)."""
        # Build the payload once
        datapoints = []
        for item in batch:
            payload_item = {
                "type": item["type"],
                "timestamp": item["timestamp"],
            }
            if item.get("miner_uid") is not None:
                payload_item["miner_uid"] = item["miner_uid"]
            if item.get("miner_hotkey"):
                payload_item["miner_hotkey"] = item["miner_hotkey"]
            if isinstance(item["value"], (int, float)):
                payload_item["numeric_value"] = item["value"]
            else:
                payload_item["string_value"] = item["value"]
            if item.get("attributes"):
                payload_item["attributes"] = item["attributes"]
            datapoints.append(payload_item)

        additional_headers = []
        if self._service_name:
            additional_headers.append(("X-Service-Name", self._service_name))
        if self._network:
            additional_headers.append(("X-Network", self._network))

        # Retry loop around the internal RequestManager call
        for attempt in range(self.retry_attempts):
            try:
                response = self.request_manager.post(
                    "v1/datapoints",
                    json={"datapoints": datapoints},
                    additional_headers=additional_headers,
                )
                if 200 <= response.status_code <= 299:
                    for _ in batch:
                        self.queue.task_done()
                    return
                else:
                    raise RuntimeError(f"Telemetry POST returned {response.status_code}")
            except Exception as e:
                bt.logging.warning(f"Batch send attempt {attempt + 1} failed (size {len(batch)}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    bt.logging.error(
                        f"Failed to send batch of {len(batch)} after {self.retry_attempts} attempts; dropping."
                    )
                    for _ in batch:
                        self.queue.task_done()

    def _enqueue_datapoint(self, type: str, timestamp: str, value: float | str, miner_uid: Optional[int] = None, miner_hotkey: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None) -> bool:
        """Enqueue a datapoint; return True if enqueued, False if queue full (dropped)."""
        # CHANGE: timestamp now str (ISO)
        try:
            if self.queue.full():
                bt.logging.warning(f"Queue full (size {self.max_queue_size}); dropping datapoint {type}")
                return False
            # Convert value to ensure it's a Python scalar (handles NumPy/Torch)
            safe_value = self._to_python_scalar(value)
            if isinstance(safe_value, (int, float)):
                safe_value = float(safe_value)  # Ensure float for numericValue

            # onvert miner_uid
            safe_miner_uid = self._to_python_scalar(miner_uid) if miner_uid is not None else None
            if safe_miner_uid is not None:
                safe_miner_uid = int(safe_miner_uid)

            # Convert attributes values
            safe_attributes = None
            if attributes:
                safe_attributes = {
                    k: self._to_python_scalar(v)
                    for k, v in attributes.items()
                }

            item = {
                'type': type,
                'timestamp': timestamp,
                'value': safe_value,
                'miner_uid': safe_miner_uid,
                'miner_hotkey': miner_hotkey,
                'attributes': safe_attributes,
            }
            self.queue.put_nowait(item)

            bt.logging.debug(f"Recorded datapoint: {item}")

            return True
        except queue.Full:
            bt.logging.warning(f"Queue full; dropping datapoint {type}")
            return False

    def _enqueue_gpu_metric(
        self,
        type: str,
        timestamp: str,
        value: float | str | None,
        gpu_index: int,
    ) -> None:
        """Enqueue a per-GPU datapoint; skip fields the collector could not read."""
        if value is None:
            return
        if isinstance(value, str) and value == "":
            return
        self._enqueue_datapoint(type, timestamp, value, attributes={"gpu_index": gpu_index})

    def _report_collection_error(self, timestamp: str, collector: str, error: BaseException) -> None:
        """Log and emit a datapoint so a failed collector is visible without crashing."""
        msg = f"{type(error).__name__}: {error}"
        bt.logging.warning(f"Telemetry collection failed ({collector}): {msg}")
        try:
            self._enqueue_datapoint(
                "system_collection_error",
                timestamp,
                collector,
                attributes={"error": msg[:500]},
            )
        except Exception as enqueue_err:
            bt.logging.warning(
                f"Could not enqueue collection error for {collector}: {enqueue_err}"
            )

    def _try_record(self, timestamp: str, collector: str, fn) -> None:
        """Run one collector. Never raises; records system_collection_error on failure."""
        try:
            fn()
        except Exception as e:
            self._report_collection_error(timestamp, collector, e)

    def _record_updatable_host_info(self, timestamp: str) -> None:
        """Docker / NVIDIA driver / CUDA — can change under a long-lived process."""
        def docker_version():
            client, server = _get_docker_versions()
            self._enqueue_datapoint("system_docker_version", timestamp, client)
            if server:
                self._enqueue_datapoint("system_docker_server_version", timestamp, server)

        def nvidia_versions():
            driver_ver, cuda_ver = _get_nvidia_driver_info()
            self._enqueue_datapoint("system_gpu_driver_version", timestamp, driver_ver)
            self._enqueue_datapoint("system_cuda_version", timestamp, cuda_ver)
            if self.device != "cpu" and driver_ver == "none" and cuda_ver == "none":
                raise RuntimeError("could not read NVIDIA driver or CUDA version")

        self._try_record(timestamp, "docker_version", docker_version)
        self._try_record(timestamp, "nvidia_driver", nvidia_versions)

    def record_heartbeat(self):
        version = qbittensor.__version__
        bt.logging.info(f"🫀 Recording heartbeat version: {version}")
        try:
            timestamp: str = timestamp_iso()
            # Record version as string
            self._enqueue_datapoint("heartbeat_version", timestamp, version)
        except Exception as e:
            bt.logging.info(f"Failed to enqueue heartbeat: {e}")

    def record_startup_metrics(self):
        """Record startup system metrics (CPU, RAM, disk, GPU identity/capability)."""
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            bt.logging.warning(f"Startup metrics recording failed: {e}")
            return

        gpu_count = 0
        gpu_models = "none"
        self.gpu_indices = []

        def cpu_ram():
            cpu_family = _get_cpu_model()
            cpu_count = psutil.cpu_count() or os.cpu_count() or 1
            total_ram = psutil.virtual_memory().total
            self._enqueue_datapoint("system_cpu_family", timestamp, cpu_family)
            self._enqueue_datapoint("system_cpu_count", timestamp, cpu_count)
            self._enqueue_datapoint("system_ram_bytes", timestamp, total_ram)

        def disk():
            disk_path = os.path.abspath(os.sep)
            total_disk = psutil.disk_usage(disk_path).total
            self._enqueue_datapoint("system_disk_bytes", timestamp, total_disk)

        def gpu_identity():
            nonlocal gpu_count, gpu_models
            gpu_count, gpu_models = _get_gpu_info(self.device)
            self._enqueue_datapoint("system_gpu_count", timestamp, gpu_count)
            self._enqueue_datapoint("system_gpu_models", timestamp, gpu_models)
            if self.device != "cpu" and gpu_models in {"error", "nvidia-smi not found"}:
                raise RuntimeError(f"GPU identity collection failed: {gpu_models}")

        def gpu_capability():
            # Must run before the long-lived nvmlInit() below (this path init/shutdowns NVML).
            static_infos = _get_gpu_static_info(self.device)
            for info in static_infos:
                self._enqueue_gpu_metric(
                    "system_gpu_memory_bytes", timestamp, info.memory_total_bytes, info.index
                )
                self._enqueue_gpu_metric(
                    "system_gpu_compute_capability", timestamp, info.compute_capability, info.index
                )
                self._enqueue_gpu_metric(
                    "system_gpu_power_limit_watts", timestamp, info.power_limit_watts, info.index
                )
                self._enqueue_gpu_metric(
                    "system_gpu_power_default_limit_watts",
                    timestamp,
                    info.power_default_limit_watts,
                    info.index,
                )
                self._enqueue_gpu_metric(
                    "system_gpu_power_max_limit_watts",
                    timestamp,
                    info.power_max_limit_watts,
                    info.index,
                )
            self.gpu_indices = [info.index for info in static_infos] or _discover_gpu_indices(
                self.device
            )
            if gpu_count > 0 and not static_infos and self.device != "cpu":
                raise RuntimeError("had GPU identity but no capability snapshot")

        def gpu_nvml_session():
            if self.gpu_indices and NVML_AVAILABLE and not self._pynvml_initialized:
                try:
                    nvmlInit()
                    self._pynvml_initialized = True
                except Exception as e:
                    self.gpu_indices = []
                    self._pynvml_initialized = False
                    raise RuntimeError(f"failed to keep pynvml initialized: {e}") from e

        self._try_record(timestamp, "cpu_ram", cpu_ram)
        self._try_record(timestamp, "disk", disk)
        self._try_record(timestamp, "gpu_identity", gpu_identity)
        self._record_updatable_host_info(timestamp)
        self._try_record(timestamp, "gpu_capability", gpu_capability)
        self._try_record(timestamp, "gpu_nvml_session", gpu_nvml_session)

    def record_system_metrics(self):
        """Record periodic system metrics (CPU/RAM/disk, GPU live, docker/driver/CUDA)."""
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            bt.logging.warning(f"System metrics recording failed: {e}")
            return

        def cpu_ram():
            cpu_usage = psutil.cpu_percent(interval=1)
            ram_usage = psutil.virtual_memory().percent
            self._enqueue_datapoint("system_cpu_usage", timestamp, cpu_usage)
            self._enqueue_datapoint("system_ram_usage", timestamp, ram_usage)

        def disk():
            disk_path = os.path.abspath(os.sep)
            disk = psutil.disk_usage(disk_path)
            self._enqueue_datapoint("system_disk_usage", timestamp, float(disk.percent))

        def gpu_runtime():
            if not (self.gpu_indices and self._pynvml_initialized):
                return
            infos = _get_gpu_runtime_info(self.gpu_indices)
            for info in infos:
                self._enqueue_gpu_metric(
                    "system_gpu_utilization", timestamp, info.utilization, info.index
                )
                self._enqueue_gpu_metric(
                    "system_gpu_memory_usage",
                    timestamp,
                    info.memory_usage_percent,
                    info.index,
                )
                self._enqueue_gpu_metric(
                    "system_gpu_power_draw_watts",
                    timestamp,
                    info.power_draw_watts,
                    info.index,
                )
                self._enqueue_gpu_metric(
                    "system_gpu_power_limit_watts",
                    timestamp,
                    info.power_limit_watts,
                    info.index,
                )
                self._enqueue_gpu_metric(
                    "system_gpu_temperature_c", timestamp, info.temperature_c, info.index
                )
                self._enqueue_gpu_metric(
                    "system_gpu_throttle_reasons",
                    timestamp,
                    info.throttle_reasons,
                    info.index,
                )

        self._try_record(timestamp, "cpu_ram", cpu_ram)
        self._try_record(timestamp, "disk", disk)
        self._record_updatable_host_info(timestamp)
        self._try_record(timestamp, "gpu_runtime", gpu_runtime)
        bt.logging.info("System metrics sent")

    def record_event(
        self,
        event_type: str,
        value: float | str = 1,
        miner_hotkey: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Record a business/operational event (e.g. solution_received, platform_submission, etc.).

        This is the recommended public API for custom telemetry.
        Always try to include useful identifiers in attributes, especially submission_id when available.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        return self._enqueue_datapoint(
            type=event_type,
            timestamp=timestamp,
            value=value,
            miner_hotkey=miner_hotkey,
            attributes=attributes,
        )

    def shutdown(self):
        """
        Stop the background worker, flush any remaining datapoints, and shut down.
        """
        try:
            bt.logging.info("Shutting down metrics service...")
            self._stop_event.set()
            if self._worker_thread:
                self._worker_thread.join(timeout=5.0)

            # Force flush remaining items
            batch = []
            while not self.queue.empty():
                try:
                    batch.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            if batch:
                self._flush_batch(batch)

            # Shutdown pynvml if we initialized it for long-lived GPU metrics
            if getattr(self, '_pynvml_initialized', False):
                try:
                    nvmlShutdown()
                except Exception:
                    pass
                self._pynvml_initialized = False

            bt.logging.info("Metrics service shutdown complete. ✅")
        except Exception as e:
            bt.logging.warning(f"Error during shutdown: {e}")
