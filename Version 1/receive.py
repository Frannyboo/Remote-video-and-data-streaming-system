"""RECEIVER-side listener that:
- receives frames via HTTP POST (/frame) (multipart form-data). meta must include frame_id
- receives detection JSON via UDP (payload must include same frame_id)
- synchronizes frame <-> detection by frame_id, then processes
"""

import threading
import socket
import json
import time
from flask import Flask, request, Response
import cv2
import numpy as np

### -------- CONFIG ----------
LISTEN_HOST = "0.0.0.0"
RECEIVER_HTTP_PORT = 5001           # Flask port to receive frames
RECEIVER_DET_PORT = 8000            # UDP port to receive detection JSONs from SENDER
SENDER_CMD_PORT = 8001             # UDP port on SENDER where it listens for 'wait'/'resume
RECEIVER_ACK_PORT = 8002             # Port SENDER sends acknowledgments
SENDER_NETBIRD_IP = "xxx.xx.xx.xx"    # SENDER NetBird IP

# critical classes that should trigger operator attention (non-person or general)
CRITICAL_CLASSES = {"fire_smoke"}

# how long to keep unmatched items (seconds)
MATCH_TIMEOUT = 10.0

# whether to auto-resume if operator doesn't press resume (None = never auto-resume)
AUTO_RESUME_TIMEOUT = None  # e.g., 60  # seconds

# UDP send socket for commands to SENDER
_cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_cmd_sock.bind(("", RECEIVER_ACK_PORT))  # Bind so SENDER can send ACKs back

# thread-safe buffers
_buffer_lock = threading.Lock()
frames_buffer = {}      # frame_id -> (frame, recv_time)
detections_buffer = {}  # frame_id -> (payload, recv_time)

# flag to avoid multiple simultaneous popups
_alert_lock = threading.Lock()
alert_active = False

app = Flask(__name__)

def send_ack(msg, gcs_addr):
    """Send acknowledgment back to the GCS."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as ack_sock:
            ack_sock.sendto(msg.encode(), (gcs_addr[0], RECEIVER_ACK_PORT))
        print(f"[ACK] Sent to GCS: {msg}")
    except Exception as e:
        print(f"[ACK] Failed to send ACK: {e}")

def is_critical_detection(detections):
    """Return True if detections list has any critical class (non-person)"""
    for d in detections:
        cls = d.get("class", "").lower()
        if cls in CRITICAL_CLASSES:
            return cls, True
    return None, False

@app.route('/frame', methods=['POST'])
def receive_frame():
    """
    Expects:
      - multipart form with 'frame' file (jpeg)
      - form field 'meta' containing JSON with at least 'frame_id' and 'timestamp' and optionally detections
    """
    try:
        meta_raw = request.form.get('meta', None)
        if meta_raw is None:
            return "Missing meta", 400
        meta = json.loads(meta_raw)
        frame_id = meta.get("frame_id") or meta.get("timestamp")
        if frame_id is None:
            return "meta must include frame_id or timestamp", 400

        file = request.files.get('frame', None)
        if file is None:
            return "Missing frame file", 400
        img_bytes = file.read()
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return "Bad image", 400

        # store frame in buffer
        with _buffer_lock:
            frames_buffer[frame_id] = (frame, time.time())

        # Try to match right away
        try_match(frame_id)

        return "OK", 200

    except Exception as e:
        print("[RECEIVER] /frame error:", e)
        return "Error", 500


# ----------------- UDP detection listener -----------------
def detection_udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', RECEIVER_DET_PORT))
    print(f"[RECEIVER] Listening for detection JSONs on UDP port {RECEIVER_DET_PORT}")
    while True:
        data, addr = sock.recvfrom(65536)
        try:
            payload = json.loads(data.decode())
        except Exception as e:
            print("[RECEIVER] Invalid JSON from", addr, e)
            continue

        frame_id = payload.get("frame_id") or payload.get("timestamp")
        if frame_id is None:
            print("[RECEIVER] detection JSON missing frame_id/timestamp, ignoring")
            continue

        with _buffer_lock:
            detections_buffer[frame_id] = (payload, time.time())

        try_match(frame_id)

# ----------------- matching and processing -----------------
def try_match(frame_id):
    """
    If both frame and detection exist for frame_id, process them together.
    """
    # must be called with no lock held, will acquire as needed
    with _buffer_lock:
        frame_entry = frames_buffer.get(frame_id)
        det_entry = detections_buffer.get(frame_id)

    if (frame_entry is not None) and (det_entry is not None):
        frame, _ = frame_entry
        det_payload, _ = det_entry

        # remove matched
        with _buffer_lock:
            frames_buffer.pop(frame_id, None)
            detections_buffer.pop(frame_id, None)

        # process in a worker to avoid blocking listener
        threading.Thread(target=process_matched, args=(frame_id, frame, det_payload), daemon=True).start()

def process_matched(frame_id, frame, det_payload):
    """
    Core processing: check detections, run action recognition if needed,
    send WAIT/RESUME to SENDER, and pop up alert window for operator interaction.
    """
    global alert_active
    detections = det_payload.get("detections", [])
    print(f"[RECEIVER] Matched frame_id={frame_id} with {len(detections)} detections")

    # if nothing interesting, ignore
    if not detections:
        return
    
    

    # If any person detected -> show alert
    for d in detections:
        if d.get("class","").lower() != "":
            show_alert_window(frame, d, frame_id)
    


# ----------------- Alert / popup -----------------
def show_alert_window(frame, event, frame_id):
    """
    Pops up a window showing the frame and waits for operator input:
      - Press 'r' to send RESUME (and close)
      - Press 'q' or ESC to close but keep in WAIT (no resume)
    This blocks this thread only, not the entire app.
    """
    global alert_active
    with _alert_lock:
        if alert_active:
            # if another alert already active, we just display a thumbnail update
            print("[RECEIVER] Another alert active — skipSENDERng new popup")
            return
        alert_active = True

    win_name = f"ALERT - {frame_id}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, frame)
    cv2.putText(frame, event, (620,10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    print("[RECEIVER] ALERT window opened. Press 'r' to RESUME, 'q' or ESC to dismiss (keep drone in WAIT).")

    # start = time.time()
    while True:
        key = cv2.waitKey(100) & 0xFF
        # auto-resume logic (optional)
        # if AUTO_RESUME_TIMEOUT is not None and (time.time() - start) > AUTO_RESUME_TIMEOUT:
        #     print("[RECEIVER] AUTO_RESUME timeout reached. Sending RESUME.")
        #     send_command_to_SENDER("RESUME")
        #     break

        # if key == ord('r'):   # resume
        #     print("[RECEIVER] Operator requested RESUME")
        #     send_command_to_SENDER("RESUME")
        #     break
        if key == ord('q') or key == 27:  # q or ESC to close without resuming
            print("[RECEIVER] Alert dismissed by operator (no resume).")
            break

    cv2.destroyWindow(win_name)
    with _alert_lock:
        alert_active = False

# ----------------- cleanup thread -----------------
def cleanup_buffers():
    """Periodically remove stale frames/detections older than MATCH_TIMEOUT"""
    while True:
        now = time.time()
        with _buffer_lock:
            stale_frames = [fid for fid, (_, t) in frames_buffer.items() if now - t > MATCH_TIMEOUT]
            stale_dets = [fid for fid, (_, t) in detections_buffer.items() if now - t > MATCH_TIMEOUT]
            for fid in stale_frames:
                frames_buffer.pop(fid, None)
            for fid in stale_dets:
                detections_buffer.pop(fid, None)
        time.sleep(2.0)

# ----------------- main -----------------
if __name__ == "__main__":
    # start UDP detection listener
    t = threading.Thread(target=detection_udp_listener, daemon=True)
    t.start()

    #Thread to listen for acknowledgement of command sent to SENDER
    threading.Thread(target=send_ack, daemon=True).start()

    # start cleanup thread
    threading.Thread(target=cleanup_buffers, daemon=True).start()

    # start Flask server (frame receiver) in another thread so we keep main free
    threading.Thread(target=lambda: app.run(host=LISTEN_HOST, port=RECEIVER_HTTP_PORT, threaded=True, use_reloader=False), daemon=True).start()

    print("[RECEIVER] Listener ready. HTTP port:", RECEIVER_HTTP_PORT, " UDP det port:", RECEIVER_DET_PORT)
    # main thread can just wait forever
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Exiting RECEIVER listener.")