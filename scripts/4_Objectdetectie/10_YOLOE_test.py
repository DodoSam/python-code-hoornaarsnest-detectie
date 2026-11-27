
import cv2
from ultralytics import YOLOE

# Load pretrained YOLOE
model = YOLOE("yoloe-26s-seg.pt")

# Objects to look for
model.set_classes([
    "hornet nest",
    "wasp nest",
    "paper wasp nest",
    "hornet"
])

# Open webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError("Cannot open webcam!")

print("YOLOE started. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        break

    # Detect nests in the frame
    results = model.predict(
        frame,
        conf=0.10,
        imgsz=640,
        verbose=False
    )

    # Draw bounding boxes
    annotated = results[0].plot(boxes=True, masks=False)

    cv2.imshow("YOLOE - Hornet Nest Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
