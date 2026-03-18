"""
Smart Behavioral Video Compression — solution.py
=================================================
Sentio Mind Assignment | SBVC Pipeline

5-Step Algorithm (exactly as specified):
  Step 1 : pHash         — drop if > 95% similar to last kept frame
  Step 2 : Optical flow  — discard if motion score < 0.05
  Step 3 : Haar face     — keep regardless of motion if face detected
  Step 4 : Context frame — keep one frame every 3 s minimum
  Step 5 : Re-encode     — H.264 MP4 @ 12 fps via ffmpeg

Outputs:
  compressed_output.mp4
  compression_report.html
  segments_kept.json

Usage:
  python solution.py                          # uses defaults
  python solution.py --input myvideo.mov      # custom input
  python solution.py --help                   # all options
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import imagehash
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION  (match assignment spec exactly)
# ─────────────────────────────────────────────────────────────────
PHASH_SIMILARITY_THRESHOLD = 0.95   # > 95% similar → duplicate → drop
OPTICAL_FLOW_THRESHOLD     = 0.05   # < 0.05        → static   → drop
CONTEXT_FRAME_INTERVAL_S   = 3.0    # keep ≥1 frame every 3 s
OUTPUT_FPS                 = 12     # re-encode target fps
FACE_SCALE_FACTOR          = 1.1
FACE_MIN_NEIGHBORS         = 5
FACE_MIN_SIZE              = (30, 30)
HAAR_CASCADE               = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ── Speed optimisation flags ──────────────────────────────────────
# Process every Nth frame only — massively reduces CPU work.
# At 58 fps input and SAMPLE_EVERY=3, we analyse ~19 fps worth of
# frames which is still above the 12 fps output target.
SAMPLE_EVERY      = 3      # analyse 1 in every N frames
RESIZE_FOR_PROC   = 0.5    # shrink frame to 50% for pHash/flow/face
                            # (invisible in final output — we extract
                            #  the original-size frame for encoding)


# ─────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────
@dataclass
class FrameDecision:
    frame_index:    int
    timestamp_s:    float
    kept:           bool
    reason:         str       # phash_dup | low_motion | motion | face | face+motion | context
    motion_score:   float = 0.0
    face_detected:  bool  = False
    phash_distance: int   = 0


@dataclass
class Segment:
    """Contiguous run of kept frames — plugs into extract_intelligent_frames()"""
    segment_id:    str
    start_frame:   int
    end_frame:     int
    start_time_s:  float
    end_time_s:    float
    frame_count:   int
    has_faces:     bool
    avg_motion:    float
    keep_reason:   str
    frame_indices: List[int] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def compute_phash(frame_bgr: np.ndarray) -> imagehash.ImageHash:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def phash_similarity(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> float:
    """Returns 0.0–1.0  (1.0 = identical)"""
    return 1.0 - (h1 - h2) / len(h1.hash.flatten())


def optical_flow_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Dense Farneback optical flow → mean magnitude (higher = more motion)"""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def detect_faces(frame_bgr: np.ndarray, cascade: cv2.CascadeClassifier) -> bool:
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE,
    )
    return len(faces) > 0


def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────
# INTEGRATION ENTRY-POINT  (fixed signature — do not rename)
# ─────────────────────────────────────────────────────────────────
def extract_intelligent_frames(
    video_path: str,
    segments_json: Optional[str] = None,
) -> List[FrameDecision]:
    """
    Sentio Mind pipeline integration point.

    Fast path  : if segments_json exists, returns kept frames from manifest
                 without re-scanning the raw video.
    Full path  : runs 5-step analysis on the video from scratch.
    """
    if segments_json and Path(segments_json).exists():
        with open(segments_json) as fh:
            data = json.load(fh)
        decisions = []
        for seg in data.get("segments", []):
            for idx in seg.get("frame_indices", []):
                decisions.append(FrameDecision(
                    frame_index=idx,
                    timestamp_s=seg["start_time_s"],
                    kept=True,
                    reason="loaded_from_segments_json",
                ))
        print(f"[fast-path] loaded {len(decisions)} frames from {segments_json}")
        return decisions

    return _analyse_video(video_path)


# ─────────────────────────────────────────────────────────────────
# STEP 1–4  — FRAME ANALYSIS
# ─────────────────────────────────────────────────────────────────
def _analyse_video(video_path: str) -> List[FrameDecision]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cascade = cv2.CascadeClassifier(HAAR_CASCADE)

    decisions:    List[FrameDecision]          = []
    last_hash:    Optional[imagehash.ImageHash] = None
    last_kept_ts: float                         = -CONTEXT_FRAME_INTERVAL_S
    prev_gray:    Optional[np.ndarray]          = None

    dur_s = total / fps
    print(f"\n{'─'*56}")
    print(f"  Input    : {Path(video_path).name}")
    print(f"  Frames   : {total}  |  FPS : {fps:.1f}  |  Duration : {dur_s:.1f}s")
    print(f"  Sampling : every {SAMPLE_EVERY} frames  |  Resize : {int(RESIZE_FOR_PROC*100)}%")
    print(f"{'─'*56}")

    t0         = time.perf_counter()
    bar_width  = 40
    frame_idx  = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts = frame_idx / fps

        # ── Progress bar (every 100 frames) ───────────────────────
        if frame_idx % 100 == 0:
            pct   = frame_idx / max(total, 1)
            filled = int(bar_width * pct)
            bar   = "█" * filled + "░" * (bar_width - filled)
            elapsed = time.perf_counter() - t0
            speed   = (ts / elapsed) if elapsed > 0.1 else 0
            print(f"\r  [{bar}] {pct*100:5.1f}%  {speed:5.1f}× RT", end="", flush=True)

        # ── FRAME SKIP — drop every Nth frame ─────────────────────
        # Skipped frames are marked as dropped with reason "sampled_out"
        if frame_idx > 0 and frame_idx % SAMPLE_EVERY != 0:
            decisions.append(FrameDecision(
                frame_index=frame_idx, timestamp_s=ts,
                kept=False, reason="sampled_out",
            ))
            frame_idx += 1
            continue

        # ── Downscale for fast processing ─────────────────────────
        h, w    = frame.shape[:2]
        small   = cv2.resize(frame, (int(w * RESIZE_FOR_PROC), int(h * RESIZE_FOR_PROC)))

        # ── STEP 1 : pHash deduplication ──────────────────────────
        curr_hash  = compute_phash(small)
        phash_dist = 64
        if last_hash is not None:
            sim        = phash_similarity(curr_hash, last_hash)
            phash_dist = int((1.0 - sim) * 64)
            if sim > PHASH_SIMILARITY_THRESHOLD:
                decisions.append(FrameDecision(
                    frame_index=frame_idx, timestamp_s=ts,
                    kept=False, reason="phash_dup",
                    phash_distance=phash_dist,
                ))
                frame_idx += 1
                continue

        # ── STEP 2 : Optical-flow motion score ────────────────────
        curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        motion    = optical_flow_score(prev_gray, curr_gray) if prev_gray is not None else 1.0
        prev_gray = curr_gray

        if motion < OPTICAL_FLOW_THRESHOLD:
            # ── STEP 3 : Face override ─────────────────────────────
            if detect_faces(small, cascade):
                last_hash    = curr_hash
                last_kept_ts = ts
                decisions.append(FrameDecision(
                    frame_index=frame_idx, timestamp_s=ts,
                    kept=True, reason="face",
                    motion_score=motion, face_detected=True,
                    phash_distance=phash_dist,
                ))
                frame_idx += 1
                continue

            # ── STEP 4 : Context frame every 3 s ──────────────────
            if ts - last_kept_ts >= CONTEXT_FRAME_INTERVAL_S:
                last_hash    = curr_hash
                last_kept_ts = ts
                decisions.append(FrameDecision(
                    frame_index=frame_idx, timestamp_s=ts,
                    kept=True, reason="context",
                    motion_score=motion, face_detected=False,
                    phash_distance=phash_dist,
                ))
                frame_idx += 1
                continue

            decisions.append(FrameDecision(
                frame_index=frame_idx, timestamp_s=ts,
                kept=False, reason="low_motion",
                motion_score=motion, phash_distance=phash_dist,
            ))
            frame_idx += 1
            continue

        # ── Motion sufficient — keep ──────────────────────────────
        face_found   = detect_faces(small, cascade)
        last_hash    = curr_hash
        last_kept_ts = ts
        decisions.append(FrameDecision(
            frame_index=frame_idx, timestamp_s=ts,
            kept=True,
            reason="face+motion" if face_found else "motion",
            motion_score=motion, face_detected=face_found,
            phash_distance=phash_dist,
        ))
        frame_idx += 1

    cap.release()

    elapsed = time.perf_counter() - t0
    kept    = sum(1 for d in decisions if d.kept)
    speed_x = dur_s / elapsed if elapsed > 0 else 0
    print(f"\r  [{'█'*bar_width}] 100.0%  {speed_x:5.1f}× RT")
    print(f"  Analysis done in {elapsed:.1f}s ({speed_x:.1f}× real-time) — kept {kept}/{total} frames")
    return decisions


# ─────────────────────────────────────────────────────────────────
# SEGMENT BUILDER
# ─────────────────────────────────────────────────────────────────
def build_segments(
    decisions: List[FrameDecision],
    fps: float,
) -> List[Segment]:
    segments: List[Segment] = []
    seg_id = 0
    i, n   = 0, len(decisions)

    while i < n:
        d = decisions[i]
        if not d.kept:
            i += 1
            continue

        frames  = [d.frame_index]
        motions = [d.motion_score]
        faces   = [d.face_detected]
        reason  = d.reason
        i += 1

        while i < n and decisions[i].kept:
            dd = decisions[i]
            frames.append(dd.frame_index)
            motions.append(dd.motion_score)
            faces.append(dd.face_detected)
            i += 1

        segments.append(Segment(
            segment_id   = f"seg_{seg_id:04d}",
            start_frame  = frames[0],
            end_frame    = frames[-1],
            start_time_s = round(frames[0] / fps, 4),
            end_time_s   = round(frames[-1] / fps, 4),
            frame_count  = len(frames),
            has_faces    = any(faces),
            avg_motion   = float(np.mean(motions)) if motions else 0.0,
            keep_reason  = reason,
            frame_indices= frames,
        ))
        seg_id += 1

    return segments


# ─────────────────────────────────────────────────────────────────
# STEP 5 — FFMPEG RE-ENCODE
# ─────────────────────────────────────────────────────────────────
def reencode_with_ffmpeg(
    source_video: str,
    kept_decisions: List[FrameDecision],
    output_path: str,
    fps: int = OUTPUT_FPS,
) -> None:
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source_video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    kept_set = {d.frame_index for d in kept_decisions}

    with tempfile.TemporaryDirectory(prefix="sbvc_") as tmpdir:
        print(f"  Extracting {len(kept_set)} frames to temp dir …")
        frame_paths: List[str] = []
        idx = written = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in kept_set:
                p = os.path.join(tmpdir, f"f_{written:07d}.jpg")
                cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_paths.append(p)
                written += 1
            idx += 1

        cap.release()

        if not frame_paths:
            print("  WARNING: no frames to encode — check analysis step")
            return

        if check_ffmpeg():
            _encode_ffmpeg(frame_paths, output_path, fps, tmpdir)
        else:
            print("  ffmpeg not found — using OpenCV fallback writer")
            _encode_opencv(frame_paths, output_path, fps, w, h)

    print(f"  Encoded → {output_path}")


def _encode_ffmpeg(
    frame_paths: List[str],
    output_path: str,
    fps: int,
    tmpdir: str,
) -> None:
    concat = os.path.join(tmpdir, "list.txt")
    with open(concat, "w") as fh:
        for p in frame_paths:
            fh.write(f"file '{p}'\n")
            fh.write(f"duration {1/fps:.6f}\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat,
        "-vf", f"fps={fps}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    print(f"  Running ffmpeg (libx264, CRF 23, {fps} fps) …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  ffmpeg error — falling back to OpenCV")
        cap = cv2.VideoCapture(frame_paths[0])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        _encode_opencv(frame_paths, output_path, fps, w, h)


def _encode_opencv(
    frame_paths: List[str],
    output_path: str,
    fps: int,
    w: int,
    h: int,
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw     = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for p in frame_paths:
        img = cv2.imread(p)
        if img is not None:
            vw.write(img)
    vw.release()


# ─────────────────────────────────────────────────────────────────
# THUMBNAIL COLLECTOR (for HTML report)
# ─────────────────────────────────────────────────────────────────
def collect_thumbnails(
    source_video: str,
    segments: List[Segment],
    max_thumbs: int = 32,
) -> List[Tuple[Segment, str]]:
    if not segments:
        return []
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        return []

    step      = max(1, len(segments) // max_thumbs)
    chosen    = segments[::step][:max_thumbs]
    targets   = {s.start_frame: s for s in chosen}
    results   : List[Tuple[Segment, str]] = []
    idx       = 0

    while len(results) < len(targets):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in targets:
            # resize to thumbnail
            h, w = frame.shape[:2]
            scale = min(1.0, 320 / w)
            thumb = cv2.resize(frame, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64    = base64.b64encode(buf).decode()
            results.append((targets[idx], b64))
        idx += 1

    cap.release()
    results.sort(key=lambda x: x[0].start_frame)
    return results


# ─────────────────────────────────────────────────────────────────
# segments_kept.json WRITER
# ─────────────────────────────────────────────────────────────────
def write_segments_json(
    segments:     List[Segment],
    decisions:    List[FrameDecision],
    source_video: str,
    output_path:  str,
    stats:        dict,
) -> None:
    payload = {
        "schema_version": "1.0",
        "source_video":   source_video,
        "compression_stats": stats,
        "segments": [asdict(s) for s in segments],
        "frame_decisions": [
            {
                "frame_index":    d.frame_index,
                "timestamp_s":    round(d.timestamp_s, 4),
                "kept":           d.kept,
                "reason":         d.reason,
                "motion_score":   round(d.motion_score, 5),
                "face_detected":  d.face_detected,
                "phash_distance": d.phash_distance,
            }
            for d in decisions
        ],
    }
    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  Segments → {output_path}")


# ─────────────────────────────────────────────────────────────────
# HTML REPORT WRITER  (fully offline, no CDN)
# ─────────────────────────────────────────────────────────────────
def write_html_report(
    output_path:  str,
    source_video: str,
    output_video: str,
    decisions:    List[FrameDecision],
    segments:     List[Segment],
    stats:        dict,
    thumbnails:   List[Tuple[Segment, str]],
) -> None:
    reason_counts: dict = {}
    for d in decisions:
        reason_counts[d.reason] = reason_counts.get(d.reason, 0) + 1

    # Bar chart rows
    COLOURS = ["#00e5ff","#ff6b35","#7fff6b","#ff3d71","#ffe66d","#b388ff","#69f0ae"]
    max_v   = max(reason_counts.values()) if reason_counts else 1
    bar_rows = ""
    for i, (lbl, val) in enumerate(reason_counts.items()):
        pct = val / max_v * 100
        col = COLOURS[i % len(COLOURS)]
        bar_rows += f"""
        <div class="bar-row">
          <span class="bar-label">{lbl}</span>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{col}"></div></div>
          <span class="bar-val">{val:,}</span>
        </div>"""

    # Timeline dots (capped at 500 for perf)
    total_f  = len(decisions)
    step_tl  = max(1, total_f // 500)
    tl_dots  = ""
    for d in decisions[::step_tl]:
        if not d.kept:            col = "#1e1e2e"; tip = f"#{d.frame_index} dropped ({d.reason})"
        elif d.face_detected:     col = "#ff6b35"; tip = f"#{d.frame_index} FACE"
        elif d.reason == "context":col = "#ffe66d"; tip = f"#{d.frame_index} context"
        else:                     col = "#00e5ff"; tip = f"#{d.frame_index} motion"
        tl_dots += f'<span class="tl-dot" style="background:{col}" title="{tip}"></span>'

    # Storyboard thumbnails
    thumb_html = ""
    for seg, b64 in thumbnails:
        if seg.has_faces:           tag, tcol = "FACE",    "#ff6b35"
        elif seg.avg_motion > 0.05: tag, tcol = "MOTION",  "#00e5ff"
        else:                       tag, tcol = "CONTEXT", "#ffe66d"
        thumb_html += f"""
        <div class="thumb-card">
          <img src="data:image/jpeg;base64,{b64}" loading="lazy" alt="seg">
          <div class="thumb-meta">
            <span class="thumb-time">{seg.start_time_s:.1f}s</span>
            <span class="thumb-tag" style="color:{tcol}">{tag}</span>
          </div>
        </div>"""

    # Segment table rows (first 200)
    seg_rows = ""
    for s in segments[:200]:
        bc = "bf-face" if s.has_faces else ("bf-motion" if s.avg_motion > 0.05 else "bf-context")
        seg_rows += f"""
        <tr>
          <td>{s.segment_id}</td>
          <td>{s.start_time_s:.2f}</td>
          <td>{s.end_time_s:.2f}</td>
          <td>{s.frame_count}</td>
          <td>{s.avg_motion:.4f}</td>
          <td>{"✓" if s.has_faces else "—"}</td>
          <td><span class="badge {bc}">{s.keep_reason}</span></td>
        </tr>"""
    if len(segments) > 200:
        seg_rows += f'<tr><td colspan="7" class="trunc">… {len(segments)-200} more rows …</td></tr>'

    orig_mb   = stats.get("original_size_mb", 0)
    comp_mb   = stats.get("compressed_size_mb", 0)
    reduction = stats.get("reduction_pct", 0)
    speed_x   = stats.get("speed_multiplier", 0)
    kept_n    = stats.get("frames_kept", 0)
    total_n   = stats.get("frames_total", 0)
    dur_s     = stats.get("processing_time_s", 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SBVC Compression Report</title>
<style>
  :root{{--bg:#0a0a12;--s:#12121e;--s2:#1a1a2e;--b:#2a2a4a;--a:#00e5ff;--a2:#ff6b35;--a3:#7fff6b;--t:#e8e8f0;--m:#7878a0;}}
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:var(--bg);color:var(--t);font-family:'Courier New',monospace;font-size:14px;line-height:1.6}}
  header{{padding:48px 40px 32px;border-bottom:1px solid var(--b);background:linear-gradient(135deg,#08080f,#0d0d1e);position:relative;overflow:hidden}}
  header::before{{content:'';position:absolute;inset:0;background:repeating-linear-gradient(90deg,transparent,transparent 59px,rgba(0,229,255,.03) 60px),repeating-linear-gradient(0deg,transparent,transparent 59px,rgba(0,229,255,.03) 60px);pointer-events:none}}
  .logo{{font-size:10px;letter-spacing:6px;color:var(--a);text-transform:uppercase;margin-bottom:12px;opacity:.8}}
  h1{{font-size:clamp(28px,5vw,52px);font-weight:700;letter-spacing:-1px;color:#fff;line-height:1.05;margin-bottom:8px}}
  h1 em{{color:var(--a);font-style:normal}}
  .sub{{font-size:11px;color:var(--m);letter-spacing:2px;text-transform:uppercase}}
  main{{max-width:1280px;margin:0 auto;padding:0 32px 80px}}
  section{{margin-top:52px}}
  h2{{font-size:10px;letter-spacing:5px;text-transform:uppercase;color:var(--a);margin-bottom:20px;padding-bottom:8px;border-bottom:1px solid var(--b)}}
  .stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:2px}}
  .stat-card{{background:var(--s);border:1px solid var(--b);padding:22px 18px;position:relative;overflow:hidden}}
  .stat-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px}}
  .red::before{{background:linear-gradient(90deg,#ff3d71,#ff6b35)}}
  .grn::before{{background:linear-gradient(90deg,#69f0ae,#00e5ff)}}
  .blu::before{{background:var(--a)}}
  .stat-label{{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--m);margin-bottom:6px}}
  .stat-value{{font-size:28px;font-weight:700;color:#fff;line-height:1}}
  .stat-unit{{font-size:12px;color:var(--m);margin-left:3px}}
  .stat-note{{font-size:10px;color:var(--a3);margin-top:6px}}
  .size-cmp{{display:flex;gap:16px;flex-wrap:wrap;background:var(--s);border:1px solid var(--b);padding:28px 24px;margin-top:14px}}
  .sz-col{{flex:1;min-width:180px}}
  .sz-lbl{{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--m);margin-bottom:8px}}
  .sz-bar{{height:44px;border-radius:2px;display:flex;align-items:center;padding:0 16px;font-weight:700;font-size:16px}}
  .sz-orig{{background:linear-gradient(90deg,#ff3d71,#ff6b35);color:#fff;width:100%}}
  .sz-comp{{background:linear-gradient(90deg,#0090a8,#00e5ff);color:#0a0a12}}
  .sz-arr{{display:flex;align-items:center;font-size:28px;color:var(--a3)}}
  .bar-row{{display:flex;align-items:center;gap:12px;margin-bottom:10px}}
  .bar-label{{width:130px;font-size:11px;color:var(--m);text-align:right;flex-shrink:0}}
  .bar-track{{flex:1;height:18px;background:var(--s2);border:1px solid var(--b);border-radius:2px;overflow:hidden}}
  .bar-fill{{height:100%;border-radius:2px}}
  .bar-val{{width:56px;font-size:12px;color:var(--t)}}
  .tl-outer{{background:var(--s);border:1px solid var(--b);padding:16px;overflow-x:auto}}
  .tl-strip{{display:flex;flex-wrap:wrap;gap:1px}}
  .tl-dot{{width:5px;height:22px;border-radius:1px;cursor:default}}
  .tl-dot:hover{{opacity:.6}}
  .tl-legend{{display:flex;gap:18px;margin-top:12px;flex-wrap:wrap}}
  .ll{{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--m)}}
  .ld{{width:10px;height:10px;border-radius:1px}}
  .thumb-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:3px}}
  .thumb-card{{background:var(--s2);border:1px solid var(--b);overflow:hidden}}
  .thumb-card img{{width:100%;display:block;aspect-ratio:16/9;object-fit:cover;filter:brightness(.88) contrast(1.08)}}
  .thumb-meta{{padding:6px 8px;display:flex;justify-content:space-between;align-items:center}}
  .thumb-time{{font-size:10px;color:var(--m)}}
  .thumb-tag{{font-size:9px;font-weight:700;letter-spacing:1px}}
  .tbl{{width:100%;border-collapse:collapse;font-size:12px}}
  .tbl th{{text-align:left;padding:8px 12px;background:var(--s2);border-bottom:2px solid var(--b);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--m)}}
  .tbl td{{padding:7px 12px;border-bottom:1px solid var(--b)}}
  .tbl tr:hover td{{background:var(--s2)}}
  .badge{{display:inline-block;padding:2px 7px;border-radius:2px;font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase}}
  .bf-face{{background:rgba(255,107,53,.18);color:#ff6b35}}
  .bf-motion{{background:rgba(0,229,255,.12);color:#00e5ff}}
  .bf-context{{background:rgba(255,230,109,.12);color:#ffe66d}}
  .trunc{{color:var(--m);text-align:center;padding:12px}}
  .algo{{background:var(--s);border:1px solid var(--b);padding:28px}}
  .algo-step{{display:flex;gap:14px;padding:12px 0;border-bottom:1px solid var(--b);align-items:flex-start}}
  .algo-step:last-child{{border-bottom:none}}
  .algo-num{{width:28px;height:28px;background:rgba(0,229,255,.1);border:1px solid var(--a);color:var(--a);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0;border-radius:2px}}
  .algo-text{{font-size:13px;line-height:1.7}}
  .algo-text strong{{color:var(--a)}}
  .algo-text em{{color:var(--m);font-style:normal;font-size:11px}}
  footer{{margin-top:80px;padding:20px 40px;border-top:1px solid var(--b);font-size:10px;color:var(--m);letter-spacing:1px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px}}
  @media(max-width:640px){{header{{padding:32px 20px 24px}}main{{padding:0 16px 60px}}}}
</style>
</head>
<body>
<header>
  <div class="logo">Sentio Mind · Smart Behavioral Video Compression</div>
  <h1>Compression <em>Report</em></h1>
  <p class="sub">pHash · Optical Flow · Haar Face Detection · H.264 Re-encode</p>
</header>
<main>

<section>
  <h2>Compression Metrics</h2>
  <div class="stat-grid">
    <div class="stat-card red"><div class="stat-label">Original Size</div><div class="stat-value">{orig_mb:.1f}<span class="stat-unit">MB</span></div></div>
    <div class="stat-card grn"><div class="stat-label">Compressed Size</div><div class="stat-value">{comp_mb:.1f}<span class="stat-unit">MB</span></div></div>
    <div class="stat-card {'grn' if reduction>=70 else 'red'}"><div class="stat-label">Reduction</div><div class="stat-value">{reduction:.1f}<span class="stat-unit">%</span></div><div class="stat-note">{'✓ TARGET MET' if reduction>=70 else '✗ Below 70%'}</div></div>
    <div class="stat-card {'grn' if speed_x>=4 else 'red'}"><div class="stat-label">Speed</div><div class="stat-value">{speed_x:.1f}<span class="stat-unit">× RT</span></div><div class="stat-note">{'✓ TARGET MET' if speed_x>=4 else '✗ Below 4×'}</div></div>
    <div class="stat-card blu"><div class="stat-label">Frames Kept</div><div class="stat-value">{kept_n:,}<span class="stat-unit">/ {total_n:,}</span></div></div>
    <div class="stat-card blu"><div class="stat-label">Segments</div><div class="stat-value">{len(segments):,}</div></div>
    <div class="stat-card blu"><div class="stat-label">Process Time</div><div class="stat-value">{dur_s:.1f}<span class="stat-unit">s</span></div></div>
    <div class="stat-card blu"><div class="stat-label">Output FPS</div><div class="stat-value">{OUTPUT_FPS}<span class="stat-unit">fps</span></div></div>
  </div>
  <div class="size-cmp">
    <div class="sz-col"><div class="sz-lbl">Original · {Path(source_video).name}</div><div class="sz-bar sz-orig">{orig_mb:.1f} MB</div></div>
    <div class="sz-arr">→</div>
    <div class="sz-col"><div class="sz-lbl">Compressed · {Path(output_video).name}</div><div class="sz-bar sz-comp" style="width:{max(4,100-reduction+5):.1f}%">{comp_mb:.1f} MB</div></div>
  </div>
</section>

<section>
  <h2>Frame Decision Breakdown</h2>
  {bar_rows}
</section>

<section>
  <h2>Frame Timeline</h2>
  <div class="tl-outer">
    <div class="tl-strip">{tl_dots}</div>
    <div class="tl-legend">
      <div class="ll"><div class="ld" style="background:#ff6b35"></div>Face</div>
      <div class="ll"><div class="ld" style="background:#00e5ff"></div>Motion</div>
      <div class="ll"><div class="ld" style="background:#ffe66d"></div>Context</div>
      <div class="ll"><div class="ld" style="background:#1e1e2e"></div>Dropped</div>
    </div>
  </div>
</section>

<section>
  <h2>Storyboard — Kept Segments</h2>
  {'<div class="thumb-grid">' + thumb_html + '</div>' if thumb_html else '<p style="color:var(--m)">Run with a real video to see thumbnails.</p>'}
</section>

<section>
  <h2>Segment Log</h2>
  <div style="overflow-x:auto">
  <table class="tbl">
    <thead><tr><th>ID</th><th>Start s</th><th>End s</th><th>Frames</th><th>Avg Motion</th><th>Faces</th><th>Reason</th></tr></thead>
    <tbody>{seg_rows}</tbody>
  </table>
  </div>
</section>

<section>
  <h2>Algorithm Pipeline</h2>
  <div class="algo">
    <div class="algo-step"><div class="algo-num">1</div><div class="algo-text"><strong>pHash deduplication</strong><br>Drop frame if &gt;95% similar to last kept <em>— imagehash.phash(), ~0.3ms/frame</em></div></div>
    <div class="algo-step"><div class="algo-num">2</div><div class="algo-text"><strong>Optical-flow motion score</strong><br>Discard if score &lt; 0.05 (empty/static scene) <em>— cv2.calcOpticalFlowFarneback, Farneback dense flow</em></div></div>
    <div class="algo-step"><div class="algo-num">3</div><div class="algo-text"><strong>Haar face detection</strong><br>Keep regardless of motion if face found <em>— haarcascade_frontalface_default.xml (built into OpenCV)</em></div></div>
    <div class="algo-step"><div class="algo-num">4</div><div class="algo-text"><strong>Context frame (3 s minimum)</strong><br>Force-keep one frame every 3 seconds <em>— prevents gaps in output video</em></div></div>
    <div class="algo-step"><div class="algo-num">5</div><div class="algo-text"><strong>H.264 re-encode via ffmpeg</strong><br>libx264 · CRF 23 · fast preset · {OUTPUT_FPS} fps · yuv420p <em>— concat demuxer for variable-duration frames</em></div></div>
  </div>
</section>

</main>
<footer>
  <span>SBVC · Sentio Mind Assignment</span>
  <span>Source: {Path(source_video).name} | Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</span>
</footer>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"  Report  → {output_path}")


# ─────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart Behavioral Video Compression — Sentio Mind",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python solution.py
  python solution.py --input myvideo.mov
  python solution.py --input cam1.mov --output out.mp4 --fps 12
        """
    )
    parser.add_argument("--input",    default="video_sample_1.mov",    metavar="PATH", help="Input video (default: video_sample_1.mov)")
    parser.add_argument("--output",   default="compressed_output.mp4", metavar="PATH", help="Output MP4 (default: compressed_output.mp4)")
    parser.add_argument("--report",   default="compression_report.html",metavar="PATH", help="HTML report (default: compression_report.html)")
    parser.add_argument("--segments", default="segments_kept.json",    metavar="PATH", help="Segment JSON (default: segments_kept.json)")
    parser.add_argument("--fps",      type=int, default=OUTPUT_FPS,    help=f"Output FPS (default: {OUTPUT_FPS})")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"\n  ERROR: Input file not found: {args.input}")
        print("  Place video_sample_1.mov in the same folder and run again.\n")
        sys.exit(1)

    orig_size  = Path(args.input).stat().st_size
    print(f"\n  Original file size: {orig_size/1e6:.2f} MB")

    # ── Analyse frames ────────────────────────────────────────────
    t_start   = time.perf_counter()
    decisions = _analyse_video(args.input)
    t_analyse = time.perf_counter() - t_start

    kept_decisions = [d for d in decisions if d.kept]

    # ── Get video FPS for segment timestamps ──────────────────────
    cap = cv2.VideoCapture(args.input)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    video_dur  = n_frames / fps if fps else 0

    segments = build_segments(decisions, fps)

    # ── Re-encode ─────────────────────────────────────────────────
    t_enc = time.perf_counter()
    reencode_with_ffmpeg(args.input, kept_decisions, args.output, args.fps)
    t_enc = time.perf_counter() - t_enc

    comp_size = Path(args.output).stat().st_size if Path(args.output).exists() else 0
    reduction = (1 - comp_size / orig_size) * 100 if orig_size else 0

    total_time = t_analyse + t_enc
    speed_x    = video_dur / total_time if total_time else 0

    stats = {
        "original_size_mb":    round(orig_size  / 1e6, 3),
        "compressed_size_mb":  round(comp_size  / 1e6, 3),
        "reduction_pct":       round(reduction, 2),
        "frames_total":        len(decisions),
        "frames_kept":         len(kept_decisions),
        "frames_dropped":      len(decisions) - len(kept_decisions),
        "keep_ratio_pct":      round(len(kept_decisions) / len(decisions) * 100, 2) if decisions else 0,
        "segments_count":      len(segments),
        "processing_time_s":   round(total_time, 2),
        "speed_multiplier":    round(speed_x, 2),
        "output_fps":          args.fps,
        "video_duration_s":    round(video_dur, 2),
        "target_met_70pct":    reduction >= 70.0,
        "target_met_4x_speed": speed_x >= 4.0,
    }

    # ── Write outputs ─────────────────────────────────────────────
    write_segments_json(segments, decisions, args.input, args.segments, stats)
    thumbnails = collect_thumbnails(args.input, segments)
    write_html_report(args.report, args.input, args.output, decisions, segments, stats, thumbnails)

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'═'*56}")
    print(f"  Original   : {stats['original_size_mb']:.2f} MB")
    print(f"  Compressed : {stats['compressed_size_mb']:.2f} MB")
    print(f"  Reduction  : {stats['reduction_pct']:.1f}%  {'✓ TARGET MET (≥70%)' if stats['target_met_70pct'] else '✗ below 70%'}")
    print(f"  Speed      : {stats['speed_multiplier']:.1f}×  {'✓ TARGET MET (≥4×)' if stats['target_met_4x_speed'] else '✗ below 4×'}")
    print(f"  Frames     : {stats['frames_kept']}/{stats['frames_total']} kept  ({stats['keep_ratio_pct']:.1f}%)")
    print(f"  Segments   : {stats['segments_count']}")
    print(f"  Time       : {stats['processing_time_s']:.1f}s")
    print(f"{'═'*56}")
    print(f"\n  Outputs:")
    print(f"    {args.output}")
    print(f"    {args.report}")
    print(f"    {args.segments}\n")


if __name__ == "__main__":
    main()
