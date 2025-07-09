import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Initialize face analysis model
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Function to get face embedding
def get_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if len(faces) == 0:
        print(f"❌ No face found in {image_path}")
        return None
    return faces[0].embedding

# Compare two images: sample.jpg vs sample1.jpg
embedding1 = get_embedding("sample.jpg")
embedding2 = get_embedding("sample1.jpg")

if embedding1 is not None and embedding2 is not None:
    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    print(f"\n🔍 Cosine Similarity: {similarity:.4f}")
    
    if similarity > 0.6:
        print("✅ Same person")
    else:
        print("❌ Different person")
