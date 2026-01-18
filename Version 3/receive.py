"""receiver-side listener that:
- receives frames via HTTP POST (/frame) (multipart form-data). meta must include frame_id
- receives detection JSON via UDP (payload must include same frame_id)
- synchronizes frame <-> detection by frame_id, then processes
- pops up an alert window
"""

import threading
import socket
import json
import time
from flask import Flask, request
import cv2
import numpy as np
import os
import math
import uuid



### -------- CONFIG ----------
LISTEN_HOST = "0.0.0.0"
receiver_HTTP_PORT = 5001           # Flask port to receive frames
receiver_DET_PORT = 8000            # UDP port to receive detection JSONs from sender
sender_CMD_PORT = 8001             # UDP port to send command to sender
receiver_ACK_PORT = 8002             # Port to receive acknowledgement from sender
sender_NETBIRD_IP = "xxx.xxx.xxx"    # sender NetBird IP


CLIP_OUTPUT_DIR = "received_clips"
CLIP_LENGTH_SEC = 5.0        # target clip length in seconds (fixed)
VIDEO_FPS = 10               # output FPS to produce smooth clip (choose 15-25)
CLIP_MAX_FRAMES = 300        # buffer limit to avoid unbounded memory (safe upper bound)
MATCH_TIMEOUT = 10.0         # seconds to keep unmatched detections/frames
CLIP_RETENTION_SEC = 600    # seconds to keep saved clips before deleting from path


#Directory to save the clips
os.makedirs(CLIP_OUTPUT_DIR, exist_ok=True)

# Per-frame matching buffers you already had (kept)
frames_buffer = {}        # frame_id -> (frame, recv_time)
detections_buffer = {}    # frame_id -> (payload_json, recv_time)
_buffer_lock = threading.Lock()

# Clip assembly buffers:
clip_lock = threading.Lock()
clip_frames = []          # list of frames in current assembling clip
clip_frame_ids = []       # parallel list of frame_ids for those frames (order kept)
clip_timestamps = []      #list of timestamps for the frames
clip_recv_times = []  # NEW: parallel list of wall-clock arrival times

clips = {}  # frame_id -> clip buffer
clips_lock = threading.Lock()

# Optional: keep recent saved clips metadata (clip_id -> info)
saved_clips = {}          # clip_ts -> { "path":..., "frame_ids":[...], "saved_at":... }

# Alert lock to avoid overlapsenderng popups
_alert_lock = threading.Lock()
alert_active = False

AUTO_RESUME_TIMEOUT = 30

INACTIVITY_TIMEOUT = CLIP_LENGTH_SEC + 1.0   # seconds of silence after which we finalize a clip
MIN_FRAMES_TO_SAVE = CLIP_MAX_FRAMES / 2     # don't save tiny clips unless forced

app = Flask(__name__)

from queue import Queue
frame_queue = Queue()


@app.route('/frame', methods=['POST'])
def receive_frame():
    """
    Expects multipart form with:
      - 'meta' form field (JSON) containing at least frame_id; optional timestamp (seconds)
      - 'frame' file (jpeg bytes)
    We enqueue a tuple (frame_id, timestamp, raw_bytes) for the worker.
    """

    try:
        if 'meta' not in request.form or 'frame' not in request.files:
            print(f"[receiver] Bad POST structure. Keys form={list(request.form.keys())}, files={list(request.files.keys())}")
            return "Missing fields", 400

        meta = json.loads(request.form['meta'])
        clip_id = meta["clip_id"]
        is_last = meta.get("is_last", False)
        if clip_id is None:
            return "meta must include frame_id", 400

         # Prefer client timestamp if provided; otherwise use server receive time (monotonic)
        #timestamp = meta.get("timestamp", time.time())
        timestamp = meta["timestamp"]
        if timestamp is None:
            # use monotonic-based timestamp to avoid clock skew issues
            timestamp = time.time()

        # read raw bytes and enqueue
        raw = request.files['frame'].read()
        frame_queue.put((clip_id, timestamp, is_last, raw))
        return "OK", 200

    except Exception as e:
        print(f"[receiver] Error in receive frame: {e}")
        return "Error", 500


@app.route('/meta', methods=['POST'])
def receive_meta():
    """
    Accept detection metadata via HTTP JSON (reliable).
    Body: JSON with at least frame_id and optionally detections list.
    """
    try:
        payload = request.get_json(force=True)
        clip_id = payload.get("clip_id")
        if clip_id is None:
            return "clip_id required", 400
        with _buffer_lock:
            detections_buffer[clip_id] = (payload, time.time())
        # quick log for diagnostics
        print(f"[receiver] Received meta via HTTP for frame_id={clip_id}")
        return "OK", 200
    except Exception as e:
        print("[receiver] /meta error:", e)
        return "Error", 500



# Backreceiver thread
def frame_worker():
    """
    Worker decodes JPEG bytes and forwards frame + metadata to add_to_clip.
    Runs in a backreceiver thread.
    """
    while True:
            clip_id, timestamp, is_last, raw = frame_queue.get()
            try:
                nsendermg = np.frombuffer(raw, np.uint8)
                frame = cv2.imdecode(nsendermg, cv2.IMREAD_COLOR)
                if frame is None:
                    print(f"[receiver] Failed to decode frame {clip_id}")
                else:# Diagnostic log — remove or reduce later
                    recv_time = time.time()  # wall-clock arrival time
                    print(f"[receiver] frame_worker: received frame_id={clip_id} ts={timestamp} is_last={is_last}")
                    add_to_clip(frame, str(clip_id), float(timestamp), recv_time, is_last)
            except Exception as e:
                print(f"[receiver] frame_worker decode error: {e}")
            finally:
                frame_queue.task_done()


def add_to_clip(frame, frame_id, clip_ts, recv_time, is_last):
    global clip_frames, clip_frame_ids, clip_timestamps, clip_recv_times

    with clip_lock:
        clip_frames.append(frame)
        clip_frame_ids.append(frame_id)
        clip_timestamps.append(float(clip_ts))
        clip_recv_times.append(float(recv_time))

        # safety cap
        if len(clip_frames) > CLIP_MAX_FRAMES:
            clip_frames.pop(0)
            clip_frame_ids.pop(0)
            clip_timestamps.pop(0)
            clip_recv_times.pop(0)

        now = time.time()
        if not is_last:
            return

        # ---- FINALIZE CLIP ONLY WHEN is_last == True ----
        frames = clip_frames.copy()
        fids   = clip_frame_ids.copy()
        ts     = clip_timestamps.copy()

        clip_frames.clear()
        clip_frame_ids.clear()
        clip_timestamps.clear()
        clip_recv_times.clear()

        threading.Thread(
            target=save_clip_and_process,
            args=(frames, fids, ts),
            daemon=True
        ).start()



def save_clip_and_process(frames, frame_ids, timestamps):
    if not frames:
        return

    triplets = sorted(zip(timestamps, frames, frame_ids), key=lambda x: x[0])

    ts = [t for t, _, _ in triplets]
    frames = [f for _, f, _ in triplets]
    frame_ids = [fid for _, _, fid in triplets]

    if len(frames) < 2:
        return

    duration = max(ts[-1] - ts[0], 0.2)
    # fps = len(frames) / duration
    # fps = max(5.0, min(15.0, fps))   # stable playback range
    fps = VIDEO_FPS   # same value used on the sender

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    clip_id = uuid.uuid4().hex[:10]
    path = os.path.join(CLIP_OUTPUT_DIR, f"clip_{clip_id}.mp4")

    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()

    print(f"[receiver] Saved clip: {path} ({len(frames)} frames @ {fps:.2f} FPS, ~{duration:.2f}s)")

    threading.Thread(
        target=process_clip,
        args=(os.path.basename(path), path, frame_ids),
        daemon=True
    ).start()


def collect_detections_for_frame_ids(frame_ids):
    """
    Return all detections whose frame_id matches any of the provided frame_ids.
    - Deduplicates detections by (class, bbox)
    - Preserves order of frame_ids
    - Returns: (detections_list, missing_frame_ids)
    """
    collected = []
    missing = []

    seen = set()   # for deduplication: store tuple(class, x1,y1,x2,y2)

    with _buffer_lock:
        for fid in frame_ids:

            entry = detections_buffer.get(fid)
            if not entry:
                missing.append(fid)
                continue
            if missing:
                print(f"[receiver] collect_detections_for_frame_ids: missing {len(missing)} detection entries for requested frames (sample first 5): {missing[:5]}")

            payload, _ts = entry
            dets = payload.get("detections", []) if isinstance(payload, dict) else []
            if not dets:
                continue

            for d in dets:
                try:
                    cls = str(d.get("class", "")).strip()
                    bbox = d.get("bbox", [])
                    # normalize bbox to ints length 4
                    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                        box = tuple([0, 0, 0, 0])
                    else:
                        box = tuple(int(x) for x in bbox[:4])
                    key = (cls.lower(), box)
                except Exception:
                    # malformed detection — skip it
                    continue

                # deduplicate
                if key not in seen:
                    seen.add(key)
                    collected.append(d)

    return collected, missing


# ---- process saved clip: run action recognition on the clip, use collected detections ---
#Run action recognition on a clip if and only if the detected class for the clip is 'person'
def process_clip(clip_name, clip_path, frame_ids_in_clip):
    print(f"[receiver] process_clip: {clip_name} | frames_count={len(frame_ids_in_clip)}")

    clip_id = frame_ids_in_clip[0]

    with _buffer_lock:
        entry = detections_buffer.get(clip_id)

    if not entry:
        print(f"[receiver] No detection metadata for clip {clip_name}")
        return

    payload, _ = entry
    detections = payload.get("detections", [])

    if not detections:
        print(f"[receiver] Detection metadata empty for clip {clip_name}")
        return

    print(f"[receiver] Collected {len(detections)} detections for clip {clip_name}")


    show_alert_window(clip_path, detections, clip_id)
    with _buffer_lock:
        detections_buffer.pop(clip_id, None)


# ----------------- Alert / popup -----------------
def show_alert_window(clip_path, event, frame_id):
    """
    Pops up a window showing the alert clip
      - Auto-resume after AUTO_RESUME_TIMEOUT if set
    """
    global alert_active
    with _alert_lock:
        if alert_active:
            print("[receiver] Another alert active — skipsenderng new popup")
            return
        alert_active = True

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"[receiver] Failed to open alert clip {clip_path}")
        with _alert_lock:
            alert_active = False
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20  # fallback

    delay = int(1000 / fps)

    win_name = f"ALERT - {frame_id}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Force window to the front and make it large
    cv2.moveWindow(win_name, 100, 100)  # position on screen (x, y)
    cv2.resizeWindow(win_name, 900, 700)  # increase size
    cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)

    start = time.time()
    print("[receiver] ALERT window opened. Press 'r' to RESUME, 'q' or ESC to dismiss (keep drone in WAIT).")

    while True:
        ret, frame = cap.read()
        if not ret:
            # loop or pause at end if desired
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            print("[receiver] Alert window closed manually (X button).")
            break

        if time.time() - start > AUTO_RESUME_TIMEOUT:
            print("[receiver] Auto-resume timeout reached. Closing alert.")
            break

        cv2.putText(frame, f"CONCERNING EVENT DETECTED: {event}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imshow(win_name, frame)

        key = cv2.waitKey(delay) & 0xFF  # ~33ms per frame = ~30fps

        if key != 255:  # 255 means no key pressed
            print(f"[DEBUG] Key pressed: {key}")

        if key == ord('q') or key == 27:
            print("[receiver] Alert dismissed by operator (no resume).")
            break

    cap.release()
    cv2.destroyWindow(win_name)

    with _alert_lock:
        alert_active = False


# CLEANUP
def cleanup_buffers():
    while True:
        now = time.time()
        with _buffer_lock:
            stale_frames = [fid for fid, (_, t) in frames_buffer.items() if now - t > 10]
            stale_dets = [fid for fid, (_, t) in detections_buffer.items() if now - t > 10]
            for fid in stale_frames:
                frames_buffer.pop(fid, None)
            for fid in stale_dets:
                detections_buffer.pop(fid, None)
        time.sleep(2.0)


def cleanup_old_clips():
    while True:
        now = time.time()
        for clip_name, meta in list(saved_clips.items()):
            if now - meta["saved_at"] > CLIP_RETENTION_SEC:
                try:
                    os.remove(meta["path"])
                    print(f"[receiver] Deleted old clip: {clip_name}")
                except Exception as e:
                    print(f"[receiver] Error deleting {clip_name}: {e}")
                saved_clips.pop(clip_name, None)
        time.sleep(60)

        
# ----------------- main -----------------
if __name__ == "__main__":
    #Thread to delete old clips after a while
    threading.Thread(target=cleanup_old_clips, daemon=True).start()

    # start cleanup thread
    threading.Thread(target=cleanup_buffers, daemon=True).start()

    # start the frame worker thread (NEW)
    threading.Thread(target=frame_worker, daemon=True).start()

    # start Flask server (frame receiver) in another thread so we keep main free
    threading.Thread(target=lambda: app.run(host=LISTEN_HOST, port=receiver_HTTP_PORT, threaded=True, use_reloader=False), daemon=True).start()

    print("[receiver] Listener ready. HTTP port:", receiver_HTTP_PORT, " UDP det port:", receiver_DET_PORT)
    # main thread can just wait forever
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Exiting receiver listener.")