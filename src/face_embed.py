import cv2
from insightface.app import FaceAnalysis

# Initialize the face analysis app with models
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Load the image
img_path = "sample.jpg"
img = cv2.imread(img_path)

# Run face detection and get embeddings
faces = app.get(img)

if len(faces) > 0:
    for i, face in enumerate(faces):
        print(f"\n🧠 Face {i+1} Embedding Vector:")
        print(face.embedding)  # 512-dimensional face vector
        # Draw bounding box for visualization
        box = face.bbox.astype(int)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
else:
    print("❌ No face detected.")

# Show image with bounding box
cv2.imshow("Face Embedding Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
