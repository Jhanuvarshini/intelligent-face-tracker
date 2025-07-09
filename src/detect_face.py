import cv2
from ultralytics import YOLO

# Load the YOLOv8 face model
model = YOLO("yolov8n-face.pt")

# Load image
img_path = "sample.jpg"
img = cv2.imread(img_path)

# Run detection
results = model(img)[0]

# Check if any detections
if results.boxes is not None and len(results.boxes) > 0:
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
else:
    print("❌ No faces detected.")

# Show image
cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
