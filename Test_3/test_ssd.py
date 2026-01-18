import os
import torch
import cv2
import yaml
import glob
import numpy as np
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDHead
from torchvision.transforms import functional as F

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
BASE_DIR = "C:/Users/16965/PycharmProjects/Watercraft-detection/datasets_coco"
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test/images")
YAML_PATH = os.path.join(BASE_DIR, "data_filtered.yaml")
RESULTS_DIR = "results_ssd_final"
OUTPUT_DIR = "results_ssd_visualized_test"

CONFIDENCE_THRESHOLD = 0.35
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. FUNKCJA ZNAJDUJĄCA OSTATNI MODEL
def get_latest_model_path(results_dir):
    search_path = os.path.join(results_dir, "run_*")
    runs = glob.glob(search_path)

    if not runs:
        raise FileNotFoundError(f"Nie znaleziono żadnych folderów treningowych w {results_dir}")

    latest_run = max(runs, key=os.path.getmtime)
    model_path = os.path.join(latest_run, "ssd_final.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Folder {latest_run} istnieje, ale brak w nim pliku ssd_final.pth")

    print(f"[INFO] Znaleziono najnowszy model: {model_path}")
    return model_path


# 2. DEFINICJA MODELU (Musi być taka sama jak w treningu)
def get_ssd_model(num_classes):
    model = ssd300_vgg16(weights=None, weights_backbone=None)

    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head = SSDHead(in_channels, num_anchors, num_classes + 1)

    return model


# MAIN
def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(YAML_PATH):
        print(f"Błąd: Nie znaleziono pliku {YAML_PATH}")
        return

    with open(YAML_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    class_names = ["background"] + cfg["names"]
    num_classes = len(cfg["names"])
    print(f"Liczba klas (bez tła): {num_classes}")

    try:
        model_path = get_latest_model_path(RESULTS_DIR)
    except FileNotFoundError as e:
        print(e)
        return

    model = get_ssd_model(num_classes)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    test_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"Znaleziono {len(test_files)} obrazów w folderze testowym.")

    if not test_files:
        print("Brak zdjęć do testowania.")
        return

    print("Rozpoczynam inferencję...")

    with torch.no_grad():
        for img_name in test_files:
            img_path = os.path.join(TEST_IMAGES_DIR, img_name)

            # Wczytanie oryginału (BGR)
            orig_img = cv2.imread(img_path)
            if orig_img is None: continue

            # Konwersja do RGB
            img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

            # Zamiana na Tensor (Wartości 0-1)
            img_tensor = F.to_tensor(img_rgb).to(DEVICE)

            preds = model([img_tensor])[0]

            # Wizualizacja
            out_img = orig_img.copy()
            found_any = False

            boxes = preds['boxes'].cpu().numpy()
            scores = preds['scores'].cpu().numpy()
            labels = preds['labels'].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score < CONFIDENCE_THRESHOLD:
                    continue

                found_any = True

                # Koordynaty (int)
                x1, y1, x2, y2 = box.astype(int)

                # Nazwa klasy
                idx = int(label)
                if 0 <= idx < len(class_names):
                    cls_name = class_names[idx]
                else:
                    cls_name = f"Class {idx}"

                text = f"{cls_name}: {score:.2f}"

                # Rysowanie ramki
                cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Tło dla tekstu
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out_img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(out_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Zapisz
            save_path = os.path.join(OUTPUT_DIR, img_name)
            cv2.imwrite(save_path, out_img)

            if found_any:
                print(f" -> {img_name}: WYKRYTO OBIEKTY")

    print(f"\nWyniki sprawdź w folderze: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()