"""
Centralised Prometheus metric definitions for the NeoQCR service.

Import this module wherever you need to record a metric:

    from src.utils.prom_metrics import (
        REQUEST_COUNT, VLM_LATENCY, YOLO_GATE, ...
    )

All metrics are defined here so that prometheus_client never sees
duplicate registrations (even when modules are imported multiple times).
"""

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


def start_gpu_monitor(interval_s: float = 5.0) -> None:
    """Poll NVML in a daemon thread and publish GPU gauges. No-op without a GPU."""
    global _gpu_monitor_started
    if _gpu_monitor_started:
        return
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
    except Exception as exc:
        print(f"[prom_metrics] GPU monitor disabled: {exc}")
        return

    def _poll():
        while True:
            try:
                for i in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    GPU_UTILIZATION.labels(str(i)).set(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    GPU_MEMORY_USED.labels(str(i)).set(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            except Exception as exc:
                print(f"[prom_metrics] GPU poll error: {exc}")
            time.sleep(interval_s)

    threading.Thread(target=_poll, name="gpu-monitor", daemon=True).start()
    _gpu_monitor_started = True
    print(f"[prom_metrics] GPU monitor started for {count} device(s)")
