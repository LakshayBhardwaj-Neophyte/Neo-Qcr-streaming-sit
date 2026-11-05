"""
Centralised Prometheus metric definitions for the NeoQCR service.

Import this module wherever you need to record a metric:

    from src.utils.prom_metrics import (
        REQUEST_COUNT, VLM_LATENCY, YOLO_GATE, ...
    )

All metrics are defined here so that prometheus_client never sees
duplicate registrations (even when modules are imported multiple times).
"""

import os
import sys
import threading
import time

from prometheus_client import Counter, Histogram, Gauge

# ────────────────────────────────────────────────────────────────────────
# 1.  REQUEST COUNTERS
# ────────────────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "neoqcr_requests_total",
    "Total API requests by endpoint, client, and outcome",
    ["endpoint", "client_name", "status_code"],
)

REQUEST_ERRORS = Counter(
    "neoqcr_request_errors_total",
    "Total request errors (exceptions / non-200)",
    ["endpoint", "client_name", "error_type"],
)

# ────────────────────────────────────────────────────────────────────────
# 2.  LATENCY HISTOGRAMS  (values in seconds)
# ────────────────────────────────────────────────────────────────────────

PROCESS_LATENCY = Histogram(
    "neoqcr_process_duration_seconds",
    "End-to-end /process latency",
    ["client_name"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0],
)

VLM_LATENCY = Histogram(
    "neoqcr_vlm_inference_seconds",
    "VLM (Qwen) inference latency",
    ["endpoint", "client_name"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
)

REGEX_LATENCY = Histogram(
    "neoqcr_regex_extraction_seconds",
    "Regex metadata extraction latency",
    ["client_name"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
)

PARSE_LATENCY = Histogram(
    "neoqcr_request_parse_seconds",
    "Request parsing latency (image decode + JWT)",
    ["endpoint", "client_name"],
)

USER_VAL_LATENCY = Histogram(
    "neoqcr_user_validation_seconds",
    "User validation DB lookup latency",
    ["client_name"],
)

DB_SAVE_LATENCY = Histogram(
    "neoqcr_db_save_seconds",
    "Background metadata save latency",
    ["client_name", "operation"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

UPLOAD_LATENCY = Histogram(
    "neoqcr_image_upload_seconds",
    "Image upload to blob storage latency",
    ["client_name", "storage_backend"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# ────────────────────────────────────────────────────────────────────────
# 3.  YOLO QUALITY GATE
# ────────────────────────────────────────────────────────────────────────

YOLO_GATE = Counter(
    "neoqcr_yolo_quality_gate_total",
    "YOLO quality gate results",
    ["endpoint", "client_name", "result", "class_name"],
)

YOLO_LATENCY = Histogram(
    "neoqcr_yolo_inference_seconds",
    "YOLO classifier inference latency",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

# ────────────────────────────────────────────────────────────────────────
# 4.  WEBSOCKET STREAMING
# ────────────────────────────────────────────────────────────────────────

WS_SESSIONS = Counter(
    "neoqcr_ws_sessions_total",
    "Total WebSocket streaming sessions",
    ["client_name", "outcome"],
)

WS_ACTIVE_SESSIONS = Gauge(
    "neoqcr_ws_active_sessions",
    "Currently active WebSocket sessions",
)

WS_SESSION_DURATION = Histogram(
    "neoqcr_ws_session_duration_seconds",
    "WebSocket session duration from connect to close",
    ["client_name", "outcome"],
    buckets=[1, 2, 5, 10, 15, 20, 25, 30],
)

WS_FRAMES_RECEIVED = Counter(
    "neoqcr_ws_frames_received_total",
    "Total frames received via WebSocket",
    ["client_name", "classification"],
)

WS_VLM_TASKS_FIRED = Counter(
    "neoqcr_ws_vlm_tasks_fired_total",
    "VLM tasks launched for good frames",
    ["client_name"],
)

WS_DATE_FOUND_LATENCY = Histogram(
    "neoqcr_ws_date_found_seconds",
    "Time from WS connect to date found (winning VLM task)",
    ["client_name"],
    buckets=[0.5, 1, 2, 3, 5, 8, 10, 15, 20],
)

# ────────────────────────────────────────────────────────────────────────
# 5.  BUSINESS METRICS
# ────────────────────────────────────────────────────────────────────────

METADATA_FIELDS = Counter(
    "neoqcr_metadata_field_extracted_total",
    "Count of each metadata field successfully extracted",
    ["field", "client_name"],
)

METADATA_FIELDS_MISSING = Counter(
    "neoqcr_metadata_field_missing_total",
    "Count of each metadata field that was NOT extracted",
    ["field", "client_name"],
)

AUTH_ATTEMPTS = Counter(
    "neoqcr_auth_attempts_total",
    "JWT authentication attempts",
    ["client_name", "result"],
)

# ────────────────────────────────────────────────────────────────────────
# 6.  SYSTEM / GPU
# ────────────────────────────────────────────────────────────────────────

GPU_MEMORY_USED = Gauge(
    "neoqcr_gpu_memory_used_bytes",
    "GPU memory used",
    ["gpu_id"],
)

GPU_UTILIZATION = Gauge(
    "neoqcr_gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_id"],
)

GPU_MEMORY_TOTAL = Gauge(
    "neoqcr_gpu_memory_total_bytes",
    "GPU memory total (whole GPU, or the MIG slice)",
    ["gpu_id"],
)

# Memory held by THIS process via PyTorch — works on MIG / vGPU where NVML
# readings are "Not Supported".
GPU_PROCESS_MEMORY = Gauge(
    "neoqcr_gpu_process_memory_bytes",
    "GPU memory held by this process (PyTorch allocator)",
    ["gpu_id", "kind"],   # kind = allocated | reserved
)

# ────────────────────────────────────────────────────────────────────────
# 7.  FUNCTION-LEVEL (used by the time_calculate decorator)
# ────────────────────────────────────────────────────────────────────────

FUNCTION_DURATION = Histogram(
    "neoqcr_function_duration_seconds",
    "Duration of decorated functions",
    ["function_name"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# ────────────────────────────────────────────────────────────────────────
# 8.  HELPERS
# ────────────────────────────────────────────────────────────────────────

METADATA_FIELD_NAMES = ("mrp", "mfg_date", "expiry_date", "batch_no")


def record_metadata_fields(metadata: dict, client_name: str) -> None:
    """Count which metadata fields were / were not extracted."""
    for field in METADATA_FIELD_NAMES:
        val = metadata.get(field)
        missing = val is None or (isinstance(val, str) and val.strip().lower() in ("", "nan", "none"))
        (METADATA_FIELDS_MISSING if missing else METADATA_FIELDS).labels(field, client_name or "unknown").inc()


_gpu_monitor_started = False


def _nvml_targets(pynvml) -> list:
    """
    Return [(gpu_id_label, handle)] to poll. On a MIG-enabled GPU the parent
    handle cannot report usage, so each visible MIG slice is polled instead
    (label "<gpu>-mig<slice>").
    """
    targets = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        try:
            name = pynvml.nvmlDeviceGetName(h)
            name = name.decode() if isinstance(name, bytes) else name
        except Exception:
            name = "unknown"

        mig_on = False
        try:
            current, _pending = pynvml.nvmlDeviceGetMigMode(h)
            mig_on = current == getattr(pynvml, "NVML_DEVICE_MIG_ENABLE", 1)
        except Exception:
            pass  # not a MIG-capable GPU

        if mig_on:
            slices = 0
            try:
                max_mig = pynvml.nvmlDeviceGetMaxMigDeviceCount(h)
            except Exception:
                max_mig = 0
            for j in range(max_mig):
                try:
                    mh = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(h, j)
                except Exception:
                    continue  # slice not present / not visible to this container
                targets.append((f"{i}-mig{j}", mh))
                slices += 1
            print(f"[prom_metrics] GPU {i} ({name}): MIG enabled, {slices} slice(s) visible")
        else:
            targets.append((str(i), h))
            print(f"[prom_metrics] GPU {i} ({name}): MIG disabled")
    return targets


def start_gpu_monitor(interval_s: float = 5.0) -> None:
    """
    Poll GPU stats in a daemon thread and publish GPU gauges. No-op when
    GPU_METRICS_ENABLED=false (e.g. on K8s where DCGM exporter is used).

    * NVML: utilization, memory used/total — per GPU, or per MIG slice.
      A reading the device reports as "Not Supported" (MIG/vGPU do this for
      utilization) is logged ONCE for that GPU and then skipped.
    * PyTorch: memory allocated/reserved by this process — always available
      once CUDA is in use, including on MIG/vGPU.
    """
    global _gpu_monitor_started
    if _gpu_monitor_started:
        return
    if os.getenv("GPU_METRICS_ENABLED", "true").strip().lower() == "false":
        print("[prom_metrics] GPU monitor disabled by GPU_METRICS_ENABLED=false")
        return

    pynvml = None
    readings: dict = {}  # (gpu_id, reading_name) -> (gauge, read_fn, handle)
    try:
        import pynvml as _pynvml
        _pynvml.nvmlInit()
        pynvml = _pynvml
        for gpu_id, h in _nvml_targets(pynvml):
            readings[(gpu_id, "utilization")] = (
                GPU_UTILIZATION, lambda h: pynvml.nvmlDeviceGetUtilizationRates(h).gpu, h)
            readings[(gpu_id, "memory_used")] = (
                GPU_MEMORY_USED, lambda h: pynvml.nvmlDeviceGetMemoryInfo(h).used, h)
            readings[(gpu_id, "memory_total")] = (
                GPU_MEMORY_TOTAL, lambda h: pynvml.nvmlDeviceGetMemoryInfo(h).total, h)
    except Exception as exc:
        print(f"[prom_metrics] NVML unavailable ({exc}) — only PyTorch memory will be reported")

    not_supported_code = getattr(pynvml, "NVML_ERROR_NOT_SUPPORTED", 3) if pynvml else 3
    reported: set = set()

    def _is_not_supported(exc: Exception) -> bool:
        return getattr(exc, "value", None) == not_supported_code or "Not Supported" in str(exc)

    def _poll_torch() -> None:
        # Only read if the app already initialised CUDA — never initialise it here.
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return
        for d in range(torch.cuda.device_count()):
            GPU_PROCESS_MEMORY.labels(str(d), "allocated").set(torch.cuda.memory_allocated(d))
            GPU_PROCESS_MEMORY.labels(str(d), "reserved").set(torch.cuda.memory_reserved(d))

    def _poll():
        while True:
            for key, (gauge, read, h) in list(readings.items()):
                gpu_id, name = key
                try:
                    value = read(h)   # read first: a failed read must not create a 0-valued series
                    gauge.labels(gpu_id).set(value)
                except Exception as exc:
                    if _is_not_supported(exc):
                        print(f"[prom_metrics] GPU {gpu_id} {name}: Not Supported on this device "
                              f"(MIG/vGPU) — skipping it")
                        del readings[key]
                    elif key not in reported:
                        print(f"[prom_metrics] GPU {gpu_id} {name} poll error: {exc}")
                        reported.add(key)
            try:
                _poll_torch()
            except Exception as exc:
                if "torch" not in reported:
                    print(f"[prom_metrics] PyTorch GPU memory poll error: {exc}")
                    reported.add("torch")
            time.sleep(interval_s)

    threading.Thread(target=_poll, name="gpu-monitor", daemon=True).start()
    _gpu_monitor_started = True
    print(f"[prom_metrics] GPU monitor started — {len(readings)} NVML reading(s) + PyTorch memory")
