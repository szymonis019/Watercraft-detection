import os
import torch
from ultralytics import YOLO

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
YOLO_MODEL_PATH = "yolo_ship_project/nano_training_run/weights/best.pt"
TEST_IMAGES_DIR = "C:/Users/16965/PycharmProjects/Watercraft-detection/datasets_coco/test/images"
OUTPUT_DIR = "yolo_ship_project/nano_test_results"

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Wczytanie wytrenowanego modelu
    model = YOLO(YOLO_MODEL_PATH)

    # Wykonanie predykcji na zbiorze testowym
    results = model.predict(
        source=TEST_IMAGES_DIR,
        imgsz=320,
        device=device,
        conf=0.25,
        save=True,
        save_txt=True,
        project=OUTPUT_DIR,
        exist_ok=True
    )

    print(f"Prediction finished. Results saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
