import os
import json
import time
import socket
import logging
import threading
from flask import Flask, request, jsonify

# ---------------- CONFIG ----------------
RECEIVER_HTTP_PORT = 5001     # must match sender receiver_HTTP_PORT
RECEIVER_DET_PORT = 8000      # must match sender receiver_DET_PORT

SAVE_DIR = "receiver_data"
CLIPS_DIR = os.path.join(SAVE_DIR, "clips")
DETS_DIR = os.path.join(SAVE_DIR, "detections")

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("receiver")

# ---------------- DIRS ----------------
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(DETS_DIR, exist_ok=True)

# ---------------- FLASK APP ----------------
app = Flask(__name__)

# optional: store last meta seen via UDP
last_udp_payload = None


@app.route("/clip", methods=["POST"])
def receive_clip():
    """
    Matches sender:
      files = {'video': ('clip.avi', open(tmpfile.name, 'rb'), 'video/x-msvideo')}
      data  = {'meta': json.dumps(payload)}
    """
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video uploaded"}), 400

        meta_raw = request.form.get("meta", "{}")
        meta = json.loads(meta_raw)

        frame_id = meta.get("frame_id", str(time.time()))
        ts = meta.get("timestamp", time.time())

        # Save clip
        filename = f"clip_{frame_id}_{int(ts)}.avi"
        clip_path = os.path.join(CLIPS_DIR, filename)

        request.files["video"].save(clip_path)

        logger.info(f"[HTTP] Clip received & saved: {clip_path}")
        logger.info(f"[HTTP] Clip meta: {meta}")

        # Also save meta beside the clip (very useful for debugging)
        meta_path = os.path.join(CLIPS_DIR, f"clip_{frame_id}_{int(ts)}.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        logger.exception("[HTTP] Error receiving clip")
        return jsonify({"error": str(e)}), 500


# ---------------- UDP DETECTIONS SERVER ----------------
def udp_detection_server():
    global last_udp_payload

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", RECEIVER_DET_PORT))

    logger.info(f"[UDP] Listening for detections on port {RECEIVER_DET_PORT}")

    while True:
        try:
            data, addr = sock.recvfrom(65535)
            payload = json.loads(data.decode())

            last_udp_payload = payload

            frame_id = payload.get("frame_id")
            ts = payload.get("timestamp", time.time())
            detections = payload.get("detections", [])

            logger.info(
                f"[UDP] Detections received from {addr} | frame_id={frame_id} | count={len(detections)}"
            )

            # Save detection JSON
            det_path = os.path.join(DETS_DIR, f"detections_{frame_id}_{int(ts)}.json")
            with open(det_path, "w") as f:
                json.dump(payload, f, indent=2)

        except Exception:
            logger.exception("[UDP] Error receiving detection packet")


# ---------------- MAIN ----------------
def main():
    # start UDP server thread
    threading.Thread(target=udp_detection_server, daemon=True).start()

    # start Flask server
    logger.info(f"[HTTP] Starting Flask server on 0.0.0.0:{RECEIVER_HTTP_PORT}")
    app.run(host="0.0.0.0", port=RECEIVER_HTTP_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
