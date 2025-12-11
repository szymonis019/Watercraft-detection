import os
import gc
import torch
from ultralytics import YOLO
import time

# ---------------------------------------------------
# CONFIG 
# ---------------------------------------------------
IMG_SIZE = 320
EPOCHS = 50 
BATCH = -1 
WORKERS = 8 

YOLO_MODEL = "yolov8m.pt" 
PROJECT = "yolo_ship_project"
EXP_NAME = "training_run_optimized"

# ---------------------------------------------------
# YAML
# ---------------------------------------------------
BASE_DIR = "/home/mysiek44/Documents/sources/szymon/datasets_coco/"
DATA_YAML = os.path.join(BASE_DIR, "data_filtered.yaml")

# ---------------------------------------------------
# MAIN 
# ---------------------------------------------------
def main():
    gc.collect()
    torch.cuda.empty_cache()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("UWAGA: TTrening na CPU")

    # Wczytanie modelu
    model = YOLO(YOLO_MODEL)

    start_time = time.time()

    # Trening
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=device,
        project=PROJECT,
        name=EXP_NAME,
        exist_ok=True,
        verbose=True,
        
        # OPTYMALIZACJE
        patience=15,
        cache=True,
        amp=True,
        
        # PARAMETRY UCZENIA
        optimizer='auto',
        cos_lr=True,
        warmup_epochs=3,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} min")

if __name__ == "__main__":
    main()
