# Remote-video-and-data-streaming-system
This project demonstrates a reliable system for transmitting detection data and live video frames between devices connected to different networks (such as mobile data or Wi-Fi) using NetBird for secure peer-to-peer connectivity.  
The system handles network instability, real-time data synchronization, and asynchronous frame transmission using Python.

**Overview**
The system consists of:
- Sender Node – captures frames and sends both JSON-encoded detection data (via UDP) and video frames (via HTTP POST).
- Receiver Node – built with Flask, receives data asynchronously and processes or displays it in real time.
- NetBird VPN – ensures both nodes can communicate securely even when located on separate private networks.

## Technologies Used
- Python
- Flask – web server for receiving video/data
- Requests – HTTP communication
- Sockets (UDP) – real-time detection data
- OpenCV – frame capture and encoding
- Threading – parallel video/data handling
- JSON – lightweight data serialization
- NetBird – secure peer-to-peer VPN for cross-network communication

## 🧱 System Architecture
[Sender Device]
│
├── UDP (Detection Data)
├── HTTP POST (Video Frames)
│
▼
[Receiver Device]
│
▼
Flask Server

> Both devices are connected via NetBird, allowing secure and seamless communication even when on separate networks (e.g., mobile data ↔ Wi-Fi).

## 🧩 Project Versions

| Version | Key Update |
|----|----------------------------------------------------------------------------------------|
| v1 | Basic setup of sending frames over the network |
| v2 | Enabled video saving on sending device to be sent once over the network as opposed to frames |
| v3 | Reduced computational weight on sending device |

## 🧠 Key Challenges & Fixes
1. V1 suffered from inability of the receiving device to view all the frames together so as to create the video
2. V2 had high computational cost for CPU device
3. V3 solved the issues of V1 and V2

## 🔐 About NetBird Integration

NetBird was configured on both devices to create a private mesh VPN for communication.  
This ensures:
- Encrypted traffic over any network (mobile, Wi-Fi, or LAN)
- Static peer-to-peer IPs usable by both UDP and HTTP
- Minimal latency compared to public relay servers

Example setup steps:
1. Install NetBird on both devices  
2. Join both devices to the same NetBird network  
3. Use the assigned internal IPs for communication:
