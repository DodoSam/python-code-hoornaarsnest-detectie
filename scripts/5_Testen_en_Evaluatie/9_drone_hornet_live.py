"""Live Vespa velutina nest-presence screening from a drone video stream.

Compatible inputs: webcam index (0), RTSP/HTTP/UDP URL supported by OpenCV,
video file, or a USB capture device. The drone/controller must expose its video
feed to the computer; this program does not connect to or control a drone itself.

Install: python -m pip install torch torchvision opencv-python pillow numpy
Run: python drone_hornet_live.py --source 0
     python drone_hornet_live.py --source "rtsp://YOUR_STREAM_ADDRESS"
     python drone_hornet_live.py --source "flight_sample.mp4"
Keys: Q quit; S save current frame; C toggle slower multi-crop screening.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

DEFAULT_WEIGHTS = Path(r"D:\AI_Lab\Hornet_Manual_Verified\trained_model_v3\hornet_nest_resnet18_v3.pt")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default='0', help='OpenCV camera index (0), video path, RTSP/HTTP/UDP URL, or capture device')
    parser.add_argument('--model', type=Path, default=DEFAULT_WEIGHTS, help='Path to trained v3 .pt checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5, help='Nest-likelihood display threshold (0-1; needs live calibration)')
    parser.add_argument('--interval', type=float, default=0.7, help='Minimum seconds between model updates')
    parser.add_argument('--multicrop', action='store_true', help='Start with whole-image plus 5 overlapping crops (slower)')
    parser.add_argument('--output', type=Path, default=Path('drone_detection_results'), help='Folder for saved snapshots and log')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be between 0 and 1')
    if args.interval <= 0:
        parser.error('--interval must be positive')
    return args


def load_model(checkpoint, device):
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Model checkpoint not found: {checkpoint}')
    model = models.resnet18(weights=None)  # Do NOT download ImageNet weights again.
    model.fc = nn.Linear(model.fc.in_features, 2)
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    return model


PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),  # Exact v3 evaluation transform
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def make_views(rgb, multicrop):
    image = Image.fromarray(rgb)
    views = [image]
    if multicrop:
        w, h = image.size
        cw, ch = max(1, int(w * 0.6)), max(1, int(h * 0.6))
        positions = [(0, 0), (w-cw, 0), (0, h-ch), (w-cw, h-ch), ((w-cw)//2, (h-ch)//2)]
        views.extend(image.crop((x, y, x+cw, y+ch)) for x, y in positions)
    return views


def classify(model, frame, device, multicrop):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    batch = torch.stack([PREPROCESS(view) for view in make_views(rgb, multicrop)]).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy()
    whole = float(probabilities[0])
    strongest = float(probabilities.max())
    return whole, strongest


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading model on {device}: {args.model}')
    model = load_model(args.model, device)

    source = int(args.source) if args.source.isdecimal() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video source: {args.source}. Check your drone stream/capture connection.')
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Advisory: backend may ignore this

    multicrop = args.multicrop
    score = None
    whole_score = None
    last_inference = -1e9
    log_path = args.output / 'snapshots.csv'
    print('Video opened. Q = quit, S = save snapshot, C = toggle multi-crop.')
    print('These scores are NOT validated probabilities on drone footage.')
    print('No flight commands are sent to the drone.')
    failures = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures >= 10:
                    print('Video ended or stream lost. Exiting.')
                    break
                time.sleep(0.1)
                continue
            failures = 0
            now = time.monotonic()
            if now - last_inference >= args.interval:
                whole_score, score = classify(model, frame, device, multicrop)
                last_inference = time.monotonic()

            show = frame.copy()
            if score is None:
                label = 'Analyzing...'
                color = (255, 255, 255)
            elif score >= args.threshold:
                label = f'POSSIBLE NEST | score {score:.2f}'
                color = (0, 180, 255)
            else:
                label = f'No nest indicated | score {score:.2f}'
                color = (100, 230, 100)

            h, w = show.shape[:2]
            overlay = show.copy()
            cv2.rectangle(overlay, (0, 0), (w, 88), (0, 0, 0), -1)
            show = cv2.addWeighted(overlay, 0.6, show, 0.4, 0)
            cv2.putText(show, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
            mode = 'whole + 5 crops (exploratory)' if multicrop else 'whole image'
            cv2.putText(show, f'{mode}  |  Q quit  S snapshot  C crops', (14, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.imshow('Hornet nest drone screening - classification only', show)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord('c'):
                multicrop = not multicrop
                last_inference = -1e9
                print('Multi-crop mode:', 'ON' if multicrop else 'OFF')
            if key == ord('s'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                image_file = args.output / f'frame_{timestamp}.jpg'
                cv2.imwrite(str(image_file), frame)  # Store unannotated original frame
                write_header = not log_path.exists()
                with log_path.open('a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(['timestamp', 'filename', 'whole_score', 'max_score', 'multicrop', 'threshold'])
                    writer.writerow([timestamp, image_file.name, whole_score, score, multicrop, args.threshold])
                print('Saved:', image_file)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
