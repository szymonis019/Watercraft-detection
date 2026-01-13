import os
import gc
import torch
from ultralytics import YOLO
import time


def main():
    gc.collect()
    torch.cuda.empty_cache()

    if not torch.cuda.is_available():
        return

    model = YOLO("yolov10m.pt")

    start_time = time.time()

    model.train(
        data="D:/Watercraft-detection-main/Test_5/dataset_yolo/data.yaml",
        epochs=50,
        imgsz=320,
        batch=16,
        workers=0,
        device=0,
        project="D:/Watercraft-detection-main/Test_5/Results",
        name="yolov10_training_success",
        exist_ok=True,
        verbose=True,
        patience=15,
        cache=False,
        amp=True,
        optimizer='auto',
        cos_lr=True,
        warmup_epochs=3
    )

    elapsed = time.time() - start_time
    print(f"\nTrening zakończony w: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()