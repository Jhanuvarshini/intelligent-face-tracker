import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# Load models
detector = YOLO("yolov8n-face.pt")
recognizer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
recognizer.prepare(ctx_id=0)

face_db = {}
next_id = 0
SIMILARITY_THRESHOLD = 0.6

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Use webcam (0 = default laptop camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = detector(frame)[0]
    if results.boxes is None:
        continue

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_img = frame[y1:y2, x1:x2]

        faces = recognizer.get(face_img)
        if len(faces) == 0:
            continue

        embedding = faces[0].embedding
        matched = False

        for face_id, saved_emb in face_db.items():
            sim = cosine_similarity(embedding, saved_emb)
            if sim > SIMILARITY_THRESHOLD:
                matched = True
                label = f"ID {face_id}"
                break

        if not matched:
            face_id = next_id
            face_db[face_id] = embedding
            next_id += 1
            label = f"ID {face_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    cv2.imshow("Real-Time Face Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
