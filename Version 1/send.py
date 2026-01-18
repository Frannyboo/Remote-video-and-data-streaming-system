#Groundstation code

import cv2 as cv
from ultralytics import YOLO
import math
import json
import requests
import logging

import socket
import threading
import time

### ------------ CONFIG ------------
RECEIVER_NETBIRD_IP = "xx.xx.xx.xx"         # <-- set to your RECEIVER NetBird IP
RECEIVER_HTTP_PORT = 5001                 # Flask server on RECEIVER to receive frame posts
RECEIVER_DET_PORT = 8000                  # RECEIVER UDP port to receive detection JSONs
SENDER_CMD_PORT = 8001                   # SENDER UDP port to receive commands from RECEIVER (wait/resume)
RECEIVER_ACK_PORT = 8002                  # Port RECEIVER listens on for acknowledgments
JPEG_QUALITY = 80

# UDP socket to send detections
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# UDP socket to listen for commands
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind(('', SENDER_CMD_PORT))

# Load YOLO model
model = YOLO("yolov8n.pt")  # change this when you move to RPi
class_names = model.model.names

def detect_and_track(frame):
    results = model(frame, conf=0.5, imgsz=640)
    detections = []

    if len(results) > 0:
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = class_names[cls]

                detections.append({
                    "class": current_class,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2]
                })

        if len(detections) > 0:
            send_to_receiver(detections, frame)
        else:
            print("[PI] No detections found in frame results.")

    else:
        print("[PI] No detections this frame.")

    return detections


def detection_thread_func():
    video_path = "..." #add a demo video

    while True:  # Loop video indefinitely
        video = cv.VideoCapture(video_path)
        if not video.isOpened():
            print(f"[ERROR] Could not open video at {video_path}")
            break

        print("[INFO] Starting video loop...")
        prev_time = 0
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = video.read()
            if not ret:
                print("[INFO] Video ended. Restarting...")
                break

            frame = cv.resize(frame, (640, 640))

            # Start timer for FPS
            t0 = time.time()

            detections = detect_and_track(frame)
            #send_to_ground(detections, frame)
            time.sleep(0.05)

            # Draw boxes and labels
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['class']} {det['conf']:.2f}"
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, label, (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - t0)
            frame_count += 1

            # Print every few frames
            if frame_count % 10 == 0:
                print(f"[INFO] Current FPS: {fps:.2f}")

            # Draw FPS on frame
            cv.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Show frame
            cv.imshow("Detection Speed Test", frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Exiting detection loop.")
                video.release()
                cv.destroyAllWindows()
                return

        video.release()

def send_to_receiver(detections, frame):
    """
    Sends detection JSON via UDP and frame via HTTP POST to the RECEIVER.
    - detections: list of dicts, each with {class, conf, bbox}
    - frame: np.array (BGR) from OpenCV
    """
    try:
        # 1) Send detections via UDP
        frame_id = str(time.time())
        payload = {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "detections": detections
        }
        udp_sock.sendto(json.dumps(payload).encode(), (RECEIVER_NETBIRD_IP, RECEIVER_DET_PORT))

        # 2) Encode frame to JPEG
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        success, jpeg = cv.imencode('.jpg', frame, encode_param)
        if not success:
            logging.info("[SENDER] JPEG encode failed")
            return

        # 3) Send frame via HTTP POST
        files = {'frame': ('frame.jpg', jpeg.tobytes(), 'image/jpeg')}
        data = {"meta": json.dumps(payload)}
        url = f"http://{RECEIVER_NETBIRD_IP}:{RECEIVER_HTTP_PORT}/frame"
        r = requests.post(url, files=files, data=data, timeout=5)

        if r.status_code != 200:
            logging.info(f"[SENDER] Frame POST returned {r.status_code}")
        else:
            logging.info("[SENDER] Frame + detections sent successfully")

    except Exception as e:
        logging.error("[SENDER] Error sending detection/frame:", e)

def listen_for_ack():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", RECEIVER_ACK_PORT))
        print(f"[RECEIVER] Listening for ACKs on port {RECEIVER_ACK_PORT}...")
        while True:
            data, addr = s.recvfrom(1024)
            print(f"[RECEIVER] {data.decode()} from {addr}")


if __name__ == "__main__":
    detection_thread_func()