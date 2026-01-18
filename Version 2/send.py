#Sender code

import cv2
import json
import requests
import logging
import time
import socket
import threading
import os
import tempfile


logger = logging.getLogger(__name__)  # Get a logger for this module


### ------------ CONFIG ------------
receiver_NETBIRD_IP = "xxx.xx.xx.xx"         # <-- set to your receiver NetBird IP
receiver_HTTP_PORT = 5001                 # Flask server on receiver to receive frame posts
receiver_DET_PORT = 8000                  # receiver UDP port to receive detection JSONs
PI_CMD_PORT = 8001                   # Pi UDP port to receive commands from receiver (wait/resume)
receiver_ACK_PORT = 8002                  # Port receiver listens on for acknowledgments
JPEG_QUALITY = 80

VIDEO_FPS = 20
VIDEO_DURATION = 10  # seconds
VIDEO_FILENAME = "detected_clip.avi"
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'XVID')

# UDP socket to send detections
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# UDP socket to listen for commands
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind(('', PI_CMD_PORT))


def send_to_receiver(detections, get_frame_func):
    """Record ~10 s of video and send to receiver."""
    logging.info("[PI] Recording 10s clip due to detection...")

    frame_height, frame_width = 640, 640
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
    out = cv2.VideoWriter(tmpfile.name, VIDEO_CODEC, VIDEO_FPS, (frame_width, frame_height))

    start_time = time.time()
    while time.time() - start_time < VIDEO_DURATION:
        frame = cv2.resize(get_frame_func(), (640, 640))
        out.write(frame)
        time.sleep(1 / VIDEO_FPS)

    out.release()
    logging.info(f"[PI] Clip saved to {tmpfile.name}")

    # Send detections metadata via UDP
    frame_id = str(time.time())
    payload = {
        "frame_id": frame_id,
        "timestamp": time.time(),
        "detections": detections
    }
    udp_sock.sendto(json.dumps(payload).encode(), (receiver_NETBIRD_IP, receiver_DET_PORT))

    # Send video clip via HTTP POST
    files = {'video': ('clip.avi', open(tmpfile.name, 'rb'), 'video/x-msvideo')}
    data = {"meta": json.dumps(payload)}
    url = f"http://{receiver_NETBIRD_IP}:{receiver_HTTP_PORT}/clip"
    try:
        r = requests.post(url, files=files, data=data, timeout=20)
        if r.status_code == 200:
            logging.info("[PI] Video clip sent successfully.")
        else:
            logging.warning(f"[PI] Clip POST returned {r.status_code}")
    except Exception as e:
        logging.error(f"[PI] Error sending video clip: {e}")
    finally:
        files['video'][1].close()
        os.remove(tmpfile.name)


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
