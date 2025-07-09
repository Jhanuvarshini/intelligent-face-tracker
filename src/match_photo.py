import cv2
import numpy as np
import sqlite3
from insightface.app import FaceAnalysis

# Load embedding model
recognizer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
recognizer.prepare(ctx_id=0)

# Load query image
img_path = "query.jpg"
img = cv2.imread(img_path)
if img is None:
    print("❌ Could not load query.jpg")
    exit()

faces = recognizer.get(img)
if len(faces) == 0:
    print("❌ No face found in query image.")
    exit()

query_embedding = faces[0].embedding

# Load saved face embeddings from database
conn = sqlite3.connect("logs/face_data.db")
cursor = conn.cursor()
cursor.execute("SELECT id, image_path FROM faces")
rows = cursor.fetchall()
conn.close()

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

matched = False
SIMILARITY_THRESHOLD = 0.6

for face_id, image_path in rows:
    db_img = cv2.imread(image_path)
    db_faces = recognizer.get(db_img)
    if len(db_faces) == 0:
        continue
    db_embedding = db_faces[0].embedding
    sim = cosine_similarity(query_embedding, db_embedding)

    if sim > SIMILARITY_THRESHOLD:
        print(f"✅ Match found: Same person as ID {face_id} (Similarity: {sim:.4f})")
        matched = True
        break

if not matched:
    print("❌ No match found in the database.")
