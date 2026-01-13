import os
import gc
import torch
from ultralytics import RTDETR
import time


def main():
    gc.collect()
    torch.cuda.empty_cache()

    if not torch.cuda.is_available():
        return

    model = RTDETR("rtdetr-l.pt")

    start_time = time.time()

    model.train(
        data="D:/Watercraft-detection-main/Test_6/dataset_yolo/data.yaml",
        epochs=50,
        imgsz=160,
        batch=16,
        workers=0,
        device=0,
        project="D:/Watercraft-detection-main/Test_6/Results",
        name="rtdetr_transformer_fast",
        exist_ok=True,
        verbose=True,
        patience=15,
        cache=False,
        amp=True
    )

    elapsed = time.time() - start_time
    print(f"Done: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()