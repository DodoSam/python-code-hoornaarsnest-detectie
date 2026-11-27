
import cv2
from ultralytics import YOLO

MODEL_PATH = (
    r"D:\AI_Lab\Hornet_YOLO\Training"
    r"\hornet_detector_v1\weights\best.pt"
)

model = YOLO(MODEL_PATH)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError("Cannot open webcam.")

print("Hornet nest detector started. Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            break

        results = model.predict(
            frame,
            conf=0.5,
            imgsz=960,
            device=0,
            verbose=False
        )

        annotated = results[0].plot()

        cv2.imshow("Custom Hornet Nest Detector", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
