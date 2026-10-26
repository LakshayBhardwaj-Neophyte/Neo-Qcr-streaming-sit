from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import jwt
from dotenv import load_dotenv
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from PIL import Image

from src.utils.prom_metrics import (
    WS_SESSIONS, WS_ACTIVE_SESSIONS, WS_SESSION_DURATION,
    WS_FRAMES_RECEIVED, WS_DATE_FOUND_LATENCY, AUTH_ATTEMPTS,
)

load_dotenv("src/configs/.env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum time the stream runs before giving up (no MFG/EXP found → 422).
STREAM_MAX_DURATION_S: int = int(os.getenv("STREAM_MAX_DURATION_S", "20"))

# 2-class patch classifier: 0 = good (usable), 1 = bad (discard, keep streaming).
GOOD_CLASS_ID: int = 0

# Internal HTTP endpoints (same host, same port).
PROCESS_SERVER_URL: str   = os.getenv("PROCESS_SERVER_URL",   "http://localhost:4097/process")
RECAPTURE_SERVER_URL: str = os.getenv("RECAPTURE_SERVER_URL", "http://localhost:4097/recapture")

# ---------------------------------------------------------------------------
# Frame-saving toggles (debug / review)
# ---------------------------------------------------------------------------
SAVE_FRAMES: bool      = os.getenv("SAVE_FRAMES",      "false").lower() == "true"
SAVE_FRAMES_BLOB: bool = os.getenv("SAVE_FRAMES_BLOB", "false").lower() == "true"
FRAMES_SAVE_DIR: str   = os.getenv("WS_FRAMES_SAVE_DIR", "/home/shared/saved_frames_src")

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
stream_router = APIRouter()


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _ws_validate_token(token: str, client_name: str) -> bool:
    """Validate a JWT token for the given client. Mirrors api.py logic."""
    secret_key        = os.getenv("JWT_SECRET_KEY")
    algorithm         = os.getenv("JWT_ALGORITHM", "HS256")
    shipsy_secret_key = os.getenv("SHIPSY_SECRET_KEY")
    shipsy_algorithm  = os.getenv("SHIPSY_ALGORITHM", "HS256")

    if (client_name or "").strip().lower() == "shipsy":
        key, algo = shipsy_secret_key, shipsy_algorithm
    else:
        key, algo = secret_key, algorithm

    try:
        header = jwt.get_unverified_header(token)
        if header.get("alg") == "none":
            return False
        if "jku" in header or "jwk" in header:
            return False
        if "kid" in header and "../" in header.get("kid", ""):
            return False

        jwt.decode(
            token, key, algorithms=[algo],
            audience="valid-audience",
            issuer="trusted-issuer",
        )
        return True

    except Exception as exc:
        print(f"[WS auth] Token validation failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Frame parsing
# ---------------------------------------------------------------------------

def _parse_frame(data: bytes) -> Tuple[int, int, bytes]:
    """
    Parse SDK binary frame: [4B big-endian size][4B big-endian seq][JPEG bytes].
    Returns (size, seq, jpeg_bytes).  Raises ValueError on bad header.
    """
    if len(data) < 8:
        raise ValueError(f"Frame too short ({len(data)} B) — need ≥ 8 for header")
    size = int.from_bytes(data[0:4], "big")
    seq  = int.from_bytes(data[4:8], "big")
    jpeg = data[8:]
    return size, seq, jpeg


def _decode_jpeg(jpeg_bytes: bytes) -> Optional[Image.Image]:
    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        return img.convert("RGB") if img.mode != "RGB" else img
    except Exception as exc:
        print(f"[WS] JPEG decode error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Frame savers (daemon threads — never block the event loop)
# ---------------------------------------------------------------------------

def _save_frame(jpeg_bytes: bytes, save_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img = Image.open(io.BytesIO(jpeg_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(save_path, format="JPEG", quality=60)
        print(f"[WS][Save] Written → {save_path}")
    except Exception as exc:
        print(f"[WS][Save] Failed: {exc}")


def _save_frame_blob(img: Image.Image, blob_name: str, client_name: str, user_id: str) -> None:
    try:
        from src.utils.storage_router import StorageRouter
        StorageRouter().upload_image(img, blob_name, client_name, user_id)
        print(f"[WS][Save Blob] Uploaded → {blob_name}")
    except Exception as exc:
        print(f"[WS][Save Blob] Failed: {exc}")


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _is_missing(val: Any) -> bool:
    """True when a metadata field is absent / empty / 'nan'."""
    if val is None:
        return True
    if isinstance(val, str) and (not val.strip() or val.strip().lower() == "nan"):
        return True
    return False


def _parse_ddmmyyyy(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str, "%d-%m-%Y")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------

async def _apply_decision_tree(
    winner_result: Dict[str, Any],
    orchestrator: Any,
) -> Tuple[Image.Image, bool]:
    """
    Post-VLM decision tree — determines which image to send to /process.

    Returns
    -------
    (image_to_use, used_full_frame)
        image_to_use    PIL image to POST to /process for inference + DB save
        used_full_frame True  → decision required the full original frame
                        False → patch is sufficient
    """
    data         = winner_result["data"]
    patch_img    = winner_result["patch_img"]
    original_img = winner_result["original_img"]

    mrp   = data.get("mrp")
    mfg   = data.get("mfg_date")
    exp   = data.get("expiry_date")
    batch = data.get("batch_no")

    _ocr_query = (
        "what is the maximum retail price, manufacturing date, "
        "expiry date, batch number in the retail image?"
    )

    async def _full_vlm() -> Dict[str, Any]:
        """Run VLM on the full original frame → return metadata dict."""
        try:
            result = await orchestrator.qwen_model.process_batch(original_img, _ocr_query)
            full_mrp, full_dates, full_batch = orchestrator._extract_metadata(result)
            from src.utils.metadata_processor import MetadataProcessor
            return MetadataProcessor.create_metadata(full_mrp, full_dates, full_batch)
        except Exception as exc:
            print(f"[DT] Full-image VLM error: {exc}")
            return {}

    # ── Priority 1: MRP or Batch missing → Case 2 ────────────────────────
    if _is_missing(mrp) or _is_missing(batch):
        print("[DT] Case 2 — MRP/Batch missing → full image")
        return original_img, True

    # ── Priority 2a: All features present → Case 1 ───────────────────────
    if not _is_missing(mfg) and not _is_missing(exp):
        print("[DT] Case 1 — All features present → patch")
        return patch_img, False

    curr_date        = datetime.now()
    curr_date_minus7 = curr_date - timedelta(days=7)

    # ── Case 3: MFG missing, EXP present ─────────────────────────────────
    if _is_missing(mfg) and not _is_missing(exp):
        exp_dt = _parse_ddmmyyyy(exp)

        if exp_dt is None:
            print(f"[DT] Case 3 — EXP '{exp}' unparseable → patch as-is")
            return patch_img, False

        if exp_dt > curr_date:
            # 3a — clearly not expired → no full-image call needed
            print(f"[DT] Case 3a — EXP {exp} > today → patch")
            return patch_img, False

        if exp_dt > curr_date_minus7:
            # 3b — expired within 7-day grace window → still valid
            print(f"[DT] Case 3b — EXP {exp} within 7-day grace → patch")
            return patch_img, False

        # 3c — expired > 7 days ago, suspicious → verify against full image
        print(f"[DT] Case 3c — EXP {exp} > 7 days ago → verifying with full image")
        full_data = await _full_vlm()
        full_exp  = full_data.get("expiry_date")

        if full_exp == exp:
            print(f"[DT] Case 3c-i  — same EXP ({exp}) → patch")
            return patch_img, False
        else:
            print(f"[DT] Case 3c-ii — diff EXP (patch={exp}, full={full_exp}) → full")
            return original_img, True

    # ── Case 4: EXP missing, MFG present ─────────────────────────────────
    if not _is_missing(mfg) and _is_missing(exp):
        mfg_dt = _parse_ddmmyyyy(mfg)

        if mfg_dt is None:
            print(f"[DT] Case 4 — MFG '{mfg}' unparseable → full image")
            return original_img, True

        if mfg_dt <= curr_date_minus7:
            # 4a — old enough that expiry should be on label → use full image
            print(f"[DT] Case 4a — MFG {mfg} > 7 days old → full image")
            return original_img, True

        # 4b — very fresh product → verify against full image
        print(f"[DT] Case 4b — MFG {mfg} recent → verifying with full image")
        full_data = await _full_vlm()
        full_mfg  = full_data.get("mfg_date")

        if full_mfg == mfg:
            print(f"[DT] Case 4b-i  — same MFG ({mfg}) → patch")
            return patch_img, False
        else:
            print(f"[DT] Case 4b-ii — diff MFG (patch={mfg}, full={full_mfg}) → full")
            return original_img, True

    # Fallback (should not be reached with valid data)
    print("[DT] Fallback → patch")
    return patch_img, False


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@stream_router.websocket("/ws/stream")
async def stream_endpoint(
    websocket:    WebSocket,
    token:        str = Query(..., description="JWT bearer token"),
    image_name:   str = Query(..., description="SDK image name, e.g. store_user_device_token.jpg"),
    ean_code:     str = Query(..., description="Product EAN / barcode"),
    env_id:       str = Query(..., description="Environment / shelf ID"),
    client_name:  str = Query("reliance", description="Client identifier"),
    request_id:   str = Query(..., description="SDK request ID for session tracking"),
    capture_type: str = Query("false", description="'true' for multicapture second shot"),
    is_recapture: Optional[Union[bool, str]] = Query(False),
    metadata_id:  Optional[str]              = Query(""),
    roi_x1:       Optional[float] = Query(None, description="ROI top-left X"),
    roi_y1:       Optional[float] = Query(None, description="ROI top-left Y"),
    roi_x2:       Optional[float] = Query(None, description="ROI bottom-right X"),
    roi_y2:       Optional[float] = Query(None, description="ROI bottom-right Y"),
) -> None:
    """
    WebSocket endpoint — live frame streaming with 2-class patch quality gate.

    Pipeline
    --------
    1. Each incoming JPEG is classified (class 0 = good, class 1 = bad).
    2. Every good frame has its patch cropped and sent to the VLM concurrently.
    3. The FIRST VLM call that returns MFG **or** EXP sets a flag and wins.
       All other in-flight VLM calls silently discard their results on return.
    4. The decision tree determines whether the patch or the full original frame
       should be forwarded to /process for authoritative inference + DB save.
    5. If 20 s elapse without any MFG/EXP found → 422 error to SDK.
    """
    await websocket.accept()
    session_start = time.monotonic()
    client_label  = (client_name or "unknown").strip().lower()

    # ── Authentication ───────────────────────────────────────────────────
    if not _ws_validate_token(token, client_name):
        AUTH_ATTEMPTS.labels(client_label, "failure").inc()
        WS_SESSIONS.labels(client_label, "unauthorized").inc()
        await websocket.send_json({"error": "Unauthorized — invalid or expired token"})
        await websocket.close(code=1008)
        return

    AUTH_ATTEMPTS.labels(client_label, "success").inc()
    WS_ACTIVE_SESSIONS.inc()

    def _end_session_metrics(outcome: str) -> None:
        WS_ACTIVE_SESSIONS.dec()
        WS_SESSIONS.labels(client_label, outcome).inc()
        WS_SESSION_DURATION.labels(client_label, outcome).observe(time.monotonic() - session_start)

    # ── App-state models ─────────────────────────────────────────────────
    classifier   = websocket.app.state.classifier    # 2-class ImageClassifier
    orchestrator = websocket.app.state.orchestrator  # Orchestrator (VLM + DB)

    # ── Per-session state ────────────────────────────────────────────────
    seen_seqs:    set                      = set()
    save_threads: List[threading.Thread]   = []
    vlm_tasks:    List[asyncio.Task]       = []

    # ── Concurrency primitives (first-win flag) ──────────────────────────
    date_found_event: asyncio.Event  = asyncio.Event()
    winner_result:    Dict[str, Any] = {}
    winner_lock:      asyncio.Lock   = asyncio.Lock()

    # ── Frame-save directory ─────────────────────────────────────────────
    # Structure: /home/shared/saved_frames_src/<YYYY-MM-DD>/<EAN>/images_<session_id_or_time>/
    date_folder = time.strftime("%Y-%m-%d")
    ean_folder  = ean_code.strip() or "unknown_ean"
    timestamp   = time.strftime("%Y%m%d_%H%M%S")

    session_tag = (request_id or env_id or "").strip()
    if session_tag and session_tag.lower() != "none":
        images_folder = f"images_{session_tag}_{timestamp}"
    else:
        images_folder = f"images_{timestamp}"

    session_dir = str(Path(FRAMES_SAVE_DIR) / date_folder / ean_folder / images_folder)

    parts   = image_name.split("_")
    user_id = parts[1] if len(parts) >= 4 else "unknown_user"

    is_recapture_bool = str(is_recapture).strip().lower() in ("true", "1", "yes")

    print(
        f"[WS /ws/stream] Session opened — env_id={env_id}  client={client_name}  "
        f"request_id={request_id}  ean={ean_code}  is_recapture={is_recapture_bool}  "
        f"max_duration={STREAM_MAX_DURATION_S}s  save_frames={SAVE_FRAMES}\n"
        f"  → Frames save dir: {session_dir}"
    )

    # ── Shared OCR query ─────────────────────────────────────────────────
    _ocr_query = (
        "what is the maximum retail price, manufacturing date, "
        "expiry date, batch number in the retail image?"
    )

    # ── Nested coroutine: VLM on a single good patch ─────────────────────

    async def process_good_patch_vlm(
        patch_img:    Image.Image,
        original_img: Image.Image,
        bbox:         Optional[list],
        seq:          int,
    ) -> None:
        """
        Run VLM on one cropped patch.

        * Gate 1 (before call): if flag already set → skip.
        * Gate 2 (after call): if flag was set while we ran → discard result.
        * If MFG or EXP found → acquire lock → set flag → store winner.
        """
        nonlocal winner_result

        # Gate 1
        if date_found_event.is_set():
            return

        try:
            ocr_result = await orchestrator.qwen_model.process_batch(patch_img, _ocr_query)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            print(f"[WS][VLM] Frame {seq} error: {exc}")
            return

        # Gate 2
        if date_found_event.is_set():
            print(f"[WS][VLM] Frame {seq} — already won by another frame, discarding")
            return

        if not ocr_result or not isinstance(ocr_result, list):
            print(f"[WS][VLM] Frame {seq} — empty OCR result")
            return

        raw = str(ocr_result[0])
        if raw.startswith("ERROR"):
            print(f"[WS][VLM] Frame {seq} — VLM error string: {raw[:80]}")
            return

        try:
            mrp, dates, batch_no = orchestrator._extract_metadata([raw])
        except Exception as exc:
            print(f"[WS][VLM] Frame {seq} metadata extraction error: {exc}")
            return

        from src.utils.metadata_processor import MetadataProcessor
        data = MetadataProcessor.create_metadata(mrp, dates, batch_no)

        mfg = data.get("mfg_date")
        exp = data.get("expiry_date")

        print(
            f"[WS][VLM] Frame {seq} — mrp={mrp}  mfg={mfg}  exp={exp}  "
            f"batch={batch_no}  raw={raw[:80]}"
        )

        # Flag is set ONLY when at least one date (MFG or EXP) is present.
        if _is_missing(mfg) and _is_missing(exp):
            print(f"[WS][VLM] Frame {seq} — no date found, continuing stream")
            return

        # Race-free flag set
        async with winner_lock:
            if date_found_event.is_set():
                print(f"[WS][VLM] Frame {seq} — lost the race, discarding")
                return

            winner_result = {
                "data":         data,
                "patch_img":    patch_img,
                "original_img": original_img,
                "bbox":         bbox,
                "ocr_raw":      raw,
                "mrp":          mrp,
                "dates":        dates,
                "batch_no":     batch_no,
            }
            date_found_event.set()
            WS_DATE_FOUND_LATENCY.labels(client_label).observe(time.monotonic() - session_start)
            print(f"[WS][VLM] Frame {seq} WON ✓ — mfg={mfg}  exp={exp}")

    # ── Receive loop ─────────────────────────────────────────────────────
    try:
        while True:
            elapsed           = time.monotonic() - session_start
            timeout_remaining = max(0.0, STREAM_MAX_DURATION_S - elapsed)

            if timeout_remaining == 0:
                print(f"[WS] {STREAM_MAX_DURATION_S}s timeout reached")
                break

            if date_found_event.is_set():
                print("[WS] Date found — exiting receive loop")
                break

            # Short poll (0.5 s) so we notice date_found_event quickly.
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=min(0.5, timeout_remaining),
                )
            except asyncio.TimeoutError:
                # No frame in this window — re-check timeout and event.
                continue

            msg_type = message.get("type", "")

            if msg_type == "websocket.disconnect":
                print("[WS] Client disconnected gracefully")
                break

            if msg_type != "websocket.receive":
                continue

            # ── Text message ──────────────────────────────────────────────
            text = message.get("text")
            if text is not None:
                if text.strip().upper() == "DONE":
                    print("[WS] DONE received")
                    break
                continue

            # ── Binary frame ──────────────────────────────────────────────
            data_bytes = message.get("bytes")
            if not data_bytes:
                continue

            try:
                size, seq, jpeg = _parse_frame(data_bytes)
            except ValueError as exc:
                print(f"[WS] Malformed frame: {exc}")
                continue

            if seq in seen_seqs:
                continue
            seen_seqs.add(seq)

            img = _decode_jpeg(jpeg)
            if img is None:
                continue

            arrival_ts = time.monotonic() - session_start

            # ── Optional: persist raw JPEG to disk ────────────────────────
            if SAVE_FRAMES:
                save_path = os.path.join(session_dir, f"frame_{seq:06d}.jpg")
                t = threading.Thread(
                    target=_save_frame, args=(jpeg, save_path), daemon=True
                )
                t.start()
                save_threads.append(t)

            if SAVE_FRAMES_BLOB:
                blob_name = (
                    f"{client_name}/{user_id}/{ean_folder}_{timestamp}/frame_{seq:06d}.jpg"
                )
                t_blob = threading.Thread(
                    target=_save_frame_blob,
                    args=(img, blob_name, client_name, user_id),
                    daemon=True,
                )
                t_blob.start()
                save_threads.append(t_blob)

            # ── Classify (thread-pool — does not block event loop) ─────────
            loop       = asyncio.get_event_loop()
            detections = await loop.run_in_executor(None, classifier.predict, img)

            is_good      = False
            current_bbox = None

            # Good = top detection has class_id == 0
            if detections and detections[0].get("class_id") == GOOD_CLASS_ID:
                current_bbox = detections[0].get("bbox")

                if (
                    roi_x1 is not None and roi_y1 is not None
                    and roi_x2 is not None and roi_y2 is not None
                    and current_bbox
                ):
                    px1, py1, px2, py2 = current_bbox
                    cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
                    if roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2:
                        is_good = True
                else:
                    is_good = True

            WS_FRAMES_RECEIVED.labels(client_label, "good" if is_good else "bad").inc()

            print(
                f"[WS] Frame seq={seq}  ts={arrival_ts:.2f}s  "
                f"top={detections[0]['class_name'] if detections else 'none'}  "
                f"good={is_good}  vlm_tasks={len(vlm_tasks)}"
            )

            # ── Send per-frame result to SDK ──────────────────────────────
            try:
                await websocket.send_json({
                    "type":       "frame_result",
                    "seq":        seq,
                    "frame_ts":   round(arrival_ts, 3),
                    "detections": detections,
                    "is_good":    is_good,
                })
            except Exception as send_exc:
                print(f"[WS] Send error on frame {seq}: {send_exc}")
                break

            # ── Good frame + flag not yet set → crop patch + fire VLM ─────
            if is_good and not date_found_event.is_set():
                if current_bbox:
                    x1, y1, x2, y2 = current_bbox
                    pad = float(os.getenv("BBOX_PADDING_PERCENT", "0.1"))
                    if pad > 0:
                        w, h = x2 - x1, y2 - y1
                        img_w, img_h = img.size
                        x1 = max(0,     x1 - w * pad)
                        y1 = max(0,     y1 - h * pad)
                        x2 = min(img_w, x2 + w * pad)
                        y2 = min(img_h, y2 + h * pad)
                    patch = img.crop((int(x1), int(y1), int(x2), int(y2)))
                else:
                    patch = img.copy()

                task = asyncio.create_task(
                    process_good_patch_vlm(patch, img.copy(), current_bbox, seq)
                )
                vlm_tasks.append(task)

    except WebSocketDisconnect:
        print("[WS] WebSocketDisconnect exception")
    except Exception as exc:
        print(f"[WS] Receive loop error: {exc}")
        traceback.print_exc()

    # ── Cancel all still-running VLM tasks ───────────────────────────────
    for task in vlm_tasks:
        if not task.done():
            task.cancel()
    if vlm_tasks:
        await asyncio.gather(*vlm_tasks, return_exceptions=True)

    # ── Wait for frame-save threads ───────────────────────────────────────
    if save_threads:
        print(f"[WS][Save] Waiting for {len(save_threads)} save thread(s)...")
        for t in save_threads:
            t.join(timeout=5)

    print(
        f"\n[WS] === Session ended — {len(seen_seqs)} total frames  "
        f"vlm_tasks_fired={len(vlm_tasks)}  date_found={date_found_event.is_set()} ==="
    )

    # ── Send stop_stream so the app stops sending frames immediately ──────
    # The socket stays open — the app waits for process_result or error.
    if date_found_event.is_set():
        try:
            await websocket.send_json({
                "type":    "stop_stream",
                "message": "Date found — processing result, please stop sending frames",
            })
            print("[WS] stop_stream signal sent to SDK")
        except Exception as sig_exc:
            print(f"[WS] Could not send stop_stream: {sig_exc}")

    # ── Case T: 20 s timeout — no MFG or EXP found ───────────────────────
    if not date_found_event.is_set():
        print("[WS] Case T — no MFG/EXP found within 20 s → 422 error")
        _end_session_metrics("timeout")
        try:
            await websocket.send_json({
                "type":    "error",
                "code":    422,
                "message": (
                    "Could not extract manufacturing or expiry date from the product "
                    "label. Please ensure the label is clearly visible, well-lit, "
                    "and try again."
                ),
            })
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
        return

    # ── Post-session pipeline ─────────────────────────────────────────────
    ws_outcome = "error"
    try:
        last_clean_img = winner_result.get("original_img")

        # ── Step 1: Upload original frame in background ───────────────────
        try:
            parts_split = image_name.split("_")
            if len(parts_split) >= 4:
                img_user_id = parts_split[1]
                img_token   = parts_split[3].split(".jpg")[0]
                final_metadata_id = metadata_id if metadata_id else f"{env_id}{img_token}"

                def _upload_original() -> None:
                    try:
                        from src.utils.storage_router import StorageRouter
                        from src.orchestrator.orchestrator_neoqcr import build_blob_name
                        storage   = StorageRouter()
                        blob_name = build_blob_name(
                            f"{final_metadata_id}_original",
                            client_name=client_name,
                            user_id=img_user_id,
                        )
                        storage.upload_image(
                            last_clean_img, blob_name, client_name, img_user_id
                        )
                        print(f"[WS] Original frame uploaded → {blob_name}")
                    except Exception as exc:
                        print(f"[WS] Original frame upload failed: {exc}")

                threading.Thread(target=_upload_original, daemon=True).start()
        except Exception as exc:
            print(f"[WS] Could not start original upload thread: {exc}")

        # ── Step 2: Decision tree → which image to send to /process ───────
        _dt_t0 = time.perf_counter()
        final_image, used_full = await _apply_decision_tree(winner_result, orchestrator)
        _dt_ms = (time.perf_counter() - _dt_t0) * 1000
        print(
            f"[WS] Decision tree done in {_dt_ms:.1f}ms  "
            f"used_full_image={used_full}"
        )

        # ── Step 3: Encode final image → JPEG bytes ───────────────────────
        jpeg_buf = io.BytesIO()
        final_image.save(jpeg_buf, format="JPEG", quality=95)
        jpeg_buf.seek(0)
        final_jpeg_bytes = jpeg_buf.read()
        print(f"[WS] Encoded final image: {len(final_jpeg_bytes)} bytes")

        # ── Step 4: Build patch_info for /process ─────────────────────────
        patch_info_dict: Dict[str, Any] = {}
        if not used_full:
            winner_bbox = winner_result.get("bbox")
            if winner_bbox:
                patch_info_dict["last_clean_bbox"] = winner_bbox
        patch_info = json.dumps(patch_info_dict) if patch_info_dict else "{}"

        # ── Step 5: POST to /process or /recapture ────────────────────────
        target_url     = RECAPTURE_SERVER_URL if is_recapture_bool else PROCESS_SERVER_URL
        endpoint_label = "/recapture"         if is_recapture_bool else "/process"

        multipart_data = {
            "token":              (None, token),
            "image_name":         (None, image_name),
            "ean_code":           (None, ean_code),
            "env_id":             (None, env_id),
            "metadata_id":        (None, metadata_id if metadata_id else ""),
            "client_name":        (None, client_name),
            "request_id":         (None, request_id),
            "capture_type":       (None, capture_type),
            "skip_quality_check": (None, "true"),
            "session_start":      (None, str(session_start)),
            "patch_info":         (None, patch_info),
            "is_streaming":       (None, "true"),
            "image":              (image_name, final_jpeg_bytes, "image/jpeg"),
        }

        print(
            f"[WS] POSTing to {endpoint_label} — "
            f"ean={ean_code}  env={env_id}  client={client_name}  "
            f"used_full={used_full}"
        )

        async with httpx.AsyncClient(timeout=60.0) as client_http:
            _proc_t0 = time.perf_counter()
            resp     = await client_http.post(target_url, files=multipart_data)
            _proc_ms = (time.perf_counter() - _proc_t0) * 1000

        print(f"[WS] {endpoint_label} HTTP {resp.status_code} in {_proc_ms:.1f}ms")
        process_result = resp.json()
        ws_outcome = "success" if resp.status_code == 200 else "error"

        # ── Step 6: Forward result over WebSocket ─────────────────────────
        try:
            await websocket.send_json({
                "type":       "process_result",
                "status":     resp.status_code,
                "result":     process_result,
                "request_id": request_id,
                "timings": {
                    "decision_tree_ms": round(_dt_ms,   1),
                    "process_ms":       round(_proc_ms, 1),
                },
            })
            print("[WS] Process result sent to SDK")
        except Exception as send_exc:
            print(f"[WS] Could not send process result: {send_exc}")

    except httpx.TimeoutException:
        print("[WS] /process call timed out")
        try:
            await websocket.send_json({"error": "Processing timed out — please retry"})
        except Exception:
            pass

    except Exception as exc:
        print(f"[WS] Pipeline error: {exc}")
        traceback.print_exc()
        try:
            await websocket.send_json({"error": f"Processing failed: {str(exc)}"})
        except Exception:
            pass

    finally:
        _end_session_metrics(ws_outcome)
        try:
            await websocket.close()
        except Exception:
            pass
        print("[WS] Session closed")
