#receiverstation code

import cv2
import json
import requests
import logging
import time

logger = logging.getLogger(__name__)  # Get a logger for this module


import socket
import threading
import os


### ------------ CONFIG ------------
receiver_NETBIRD_IP = "xxx.xx.xx.xx"         # <-- set to your receiver NetBird IP
receiver_HTTP_PORT = 5001                 # Flask server on receiver to receive frame posts
receiver_DET_PORT = 8000                  # receiver UDP port to receive detection JSONs
sender_CMD_PORT = 8001                   # sender UDP port to receive commands from receiver (wait/resume)
receiver_ACK_PORT = 8002                  # Port receiver listens on for acknowledgments
JPEG_QUALITY = 80

VIDEO_FPS = 20
VIDEO_DURATION = 10  # seconds


# UDP socket to send detections
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_to_receiver(detections, get_frame_func):
    """
    Sends detection metadata via HTTP and streams camera frames
    for VIDEO_DURATION seconds with explicit end-of-clip signaling.
    """

    try:
        # -----------------------------
        # 1. Create a UNIQUE CLIP ID
        # -----------------------------
        clip_id = str(time.time())  # unique per detection event

        # -----------------------------
        # 2. Send METADATA (once)
        # -----------------------------
        meta_payload = {
            "clip_id": clip_id,
            "detections": detections,
            "video_duration": VIDEO_DURATION,
            "video_fps": VIDEO_FPS
        }

        meta_url = f"http://{receiver_NETBIRD_IP}:{receiver_HTTP_PORT}/meta"

        for attempt in range(3):
            try:
                r = requests.post(meta_url, json=meta_payload, timeout=8)
                if r.status_code == 200:
                    logging.info("[PI] Metadata sent to receiver")
                    break
            except Exception as e:
                logging.warning(f"[PI] Meta POST attempt {attempt+1} failed: {e}")
            time.sleep(1.0 + attempt)
        else:
            logging.warning("[PI] Metadata POST failed — continuing anyway")

        # -----------------------------
        # 3. Stream FRAMES
        # -----------------------------
        frame_interval = 1.0 / VIDEO_FPS
        total_frames = int(VIDEO_DURATION * VIDEO_FPS)

        clip_start = time.monotonic()
        next_frame_time = clip_start
        sent_frames = 0

        for frame_idx in range(total_frames):
            now = time.monotonic()

            # Pace sending (stable FPS)
            if now < next_frame_time:
                time.sleep(next_frame_time - now)

            frame = get_frame_func()
            if frame is None:
                next_frame_time += frame_interval
                continue

            # Encode frame
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            ok, jpeg = cv2.imencode(".jpg", frame, encode_param)
            if not ok:
                logging.warning("[PI] JPEG encode failed")
                next_frame_time += frame_interval
                continue

            # Clip-relative timestamp
            clip_ts = frame_idx * frame_interval

            # -----------------------------
            # 4. FRAME METADATA
            # -----------------------------
            frame_meta = {
                "clip_id": clip_id,
                "timestamp": clip_ts,
                "is_last": (frame_idx == total_frames - 1)
            }

            files = {
                "frame": ("frame.jpg", jpeg.tobytes(), "image/jpeg")
            }
            data = {
                "meta": json.dumps(frame_meta)
            }

            # Send frame (retry)
            success = False
            for attempt in range(3):
                try:
                    r = requests.post(
                        f"http://{receiver_NETBIRD_IP}:{receiver_HTTP_PORT}/frame",
                        files=files,
                        data=data,
                        timeout=15
                    )
                    if r.status_code == 200:
                        sent_frames += 1
                        success = True
                        break
                except Exception as e:
                    logging.warning(f"[PI] Frame POST attempt {attempt+1} failed: {e}")
                time.sleep(0.3 + attempt * 0.5)

            if not success:
                logging.warning("[PI] Dropped frame (network busy)")

            next_frame_time += frame_interval

        logging.info(
            f"[PI] Finished streaming clip {clip_id} "
            f"({sent_frames}/{total_frames} frames)"
        )

    except Exception as e:
        logging.error(f"[PI] send_to_receiver error: {e}")

def make_get_frame_func(video_source=0):
    """
    Returns a get_frame_func() that reads frames from OpenCV VideoCapture.

    video_source:
        0 -> webcam
        "video.mp4" -> video file
        "rtsp://..." -> rtsp stream
    """
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    def get_frame():
        ret, frame = cap.read()
        if not ret:
            return None
        return frame

    return get_frame, cap


# create the get_frame_func
get_frame_func, cap = make_get_frame_func(0)  # 0 = webcam

# example detections
detections = [{"class": "person", "conf": 0.95, "bbox": [10, 20, 200, 300]}]

# call your sender function
send_to_receiver(detections, get_frame_func)

# release camera when done
cap.release()
