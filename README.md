
# Intelligent Face Tracker

This project is an AI-based face detection and tracking system to count unique visitors from a video stream.

## 🎯 Features

- 🔍 Real-time face detection using YOLOv8
- 🧠 Face embedding using InsightFace (ArcFace model)
- 🧭 Tracking with DeepSort/ByteTrack
- 📌 Automatic face registration with unique ID
- 📝 Logs entry & exit with cropped face, timestamp, and event type
- 💾 Stores everything in local folders and a SQLite database

## ⚙️ Tech Stack

| Module           | Tech                              |
|------------------|------------------------------------|
| Face Detection   | YOLOv8                             |
| Face Recognition | InsightFace (ArcFace)              |
| Tracking         | DeepSort / ByteTrack               |
| Backend          | Python                             |
| DB & Storage     | SQLite, Filesystem                 |
| Configuration    | JSON (`config.json`)               |

## 🛠️ Setup Instructions

```bash
git clone https://github.com/YOUR_USERNAME/intelligent-face-tracker.git
cd intelligent-face-tracker
python -m venv venv
venv\Scripts\activate       # On Windows
pip install -r requirements.txt
````

## 🧾 Sample `config.json`

```json
{
  "frame_skip": 5,
  "detection_threshold": 0.5,
  "embedding_model": "arcface",
  "face_size": [112, 112],
  "db_path": "data/database.sqlite3",
  "log_path": "logs/events.log"
}
```

## 📂 Project Structure

```
intelligent-face-tracker/
│
├── config/
│   └── config.json
├── data/
│   ├── embeddings/
│   ├── database.sqlite3
│   └── registered_faces/
├── logs/
│   ├── entries/
│   ├── exits/
│   └── events.log
├── models/
│   └── yolov8.pt
├── src/
│   ├── detector.py
│   ├── recognizer.py
│   ├── tracker.py
│   ├── logger.py
│   ├── db.py
│   ├── main.py
│   └── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🎥 Demo Video

📽️ \[Add Loom/YouTube Link Here]

---

This project is a part of a hackathon run by [https://katomaran.com](https://katomaran.com)

```

