import cv2
import numpy as np
import os
import logging
import sqlite3
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ==========================
# Configuration
# ==========================
SIMILARITY_THRESHOLD = 0.6
EXIT_FRAME_THRESHOLD = 15

# ==========================
# Setup Logging and Database
# ==========================
os.makedirs("logs/entries", exist_ok=True)

# Setup text log file
logging.basicConfig(
    filename="logs/events.log",
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
)

# Connect to SQLite database
db_path = os.path.join("logs", "face_data.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Ensure the table exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id TEXT,
        timestamp TEXT,
        image_path TEXT,
        event_type TEXT
    )
''')
conn.commit()

def log_event(event_type, face_id, image_path=None):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"{event_type.upper()} - ID {face_id} at {time_now}"
    logging.info(log_msg)
    print(f"[LOG] {log_msg}")

    # Insert into database (store face_id as plain number string)
    cursor.execute('''
        INSERT INTO faces (id, timestamp, image_path, event_type)
        VALUES (?, ?, ?, ?)
    ''', (str(face_id), time_now, image_path, event_type.upper()))
    conn.commit()

# ==========================
# Load Models
# ==========================
detector = YOLO("yolov8n-face.pt")
recognizer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
recognizer.prepare(ctx_id=0)

# ==========================
# Initialize Variables
# ==========================
face_db = {}
next_id = 0
active_faces = {}
frame_count = 0

# ==========================
# Open Video
# ==========================
cap = cv2.VideoCapture("input.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_ids = set()

    results = detector(frame)[0]
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_img = frame[y1:y2, x1:x2]

            faces = recognizer.get(face_img)
            if not faces:
                continue

            embedding = faces[0].embedding
            matched = False

            for face_id, saved_emb in face_db.items():
                sim = np.dot(embedding, saved_emb) / (np.linalg.norm(embedding) * np.linalg.norm(saved_emb))
                if sim > SIMILARITY_THRESHOLD:
                    matched = True
                    label = f"ID {face_id}"
                    current_ids.add(face_id)
                    active_faces[face_id] = frame_count
                    break

            if not matched:
                face_id = next_id
                face_db[face_id] = embedding
                current_ids.add(face_id)
                active_faces[face_id] = frame_count
                next_id += 1
                label = f"ID {face_id}"

                # Save cropped image in logs/entries/YYYY-MM-DD/
                now = datetime.now()
                date_folder = now.strftime("%Y-%m-%d")
                time_stamp = now.strftime("%H-%M-%S")
                entry_folder = os.path.join("logs", "entries", date_folder)
                os.makedirs(entry_folder, exist_ok=True)
                face_filename = os.path.join(entry_folder, f"ID_{face_id}_{time_stamp}.jpg")
                cv2.imwrite(face_filename, face_img)

                # Log Entry
                log_event("entry", face_id, face_filename)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 🚪 Handle Exits
    for face_id in list(active_faces.keys()):
        if face_id not in current_ids:
            last_seen = active_faces[face_id]
            if frame_count - last_seen > EXIT_FRAME_THRESHOLD:
                log_event("exit", face_id)
                del active_faces[face_id]

    cv2.imshow("Face Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================
# Cleanup
# ==========================
cap.release()
cv2.destroyAllWindows()
conn.close()
print("✅ Done. All events logged.")
