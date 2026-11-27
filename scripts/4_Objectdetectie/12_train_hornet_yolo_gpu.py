
from pathlib import Path

import torch
from ultralytics import YOLO


# ==========================================
# HORNET NEST DETECTION - GPU TRAINING
# ==========================================

def main():

    # 1. Dataset location
    dataset_folder = Path(
        r"D:\AI_Lab\Hornet_YOLO\Hornet_YOLO_ready"
    )#D:\AI_Lab\Hornet_YOLO\Hornet_YOLO_ready

    data_yaml = dataset_folder / "data.yaml"

    # 2. Training output folder
    output_folder = Path(
        r"D:\AI_Lab\Hornet_YOLO\Training"
    )

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Cannot find dataset YAML: {data_yaml}"
        )

    # 3. Check NVIDIA CUDA
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check your "
            "PyTorch GPU installation."
        )

    print("=" * 50)
    print("HORNET NEST - YOLO GPU TRAINING")
    print("=" * 50)
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print("Dataset:", data_yaml)
    print("Output:", output_folder)

    # 4. Load pretrained YOLO detection model
    model = YOLO("yolo26s.pt")

    # 5. Train the model
    model.train(
        data=str(data_yaml),

        # Training duration
        epochs=150,
        patience=35,

        # Image resolution
        imgsz=960,

        # RTX 4060 8 GB
        batch=4,
        device=0,

        # Windows DataLoader
        workers=2,

        # Optimization
        optimizer="auto",
        pretrained=True,

        # Augmentation
        degrees=10.0,
        translate=0.10,
        scale=0.30,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,

        # Output settings
        project=str(output_folder),
        name="hornet_detector_v1",
        exist_ok=False,

        save=True,
        save_period=10,
        plots=True,

        # Reproducibility
        seed=42,

        verbose=True
    )

    print("\nTraining completed!")
    print("Your model and results are in:", output_folder)


if __name__ == "__main__":
    main()
