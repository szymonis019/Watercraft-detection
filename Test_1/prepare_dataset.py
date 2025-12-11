import os
import json
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import yaml

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
ENABLED_UNIVERSES = [1,2,3,4,5,7] 
SPLITS = ["train", "valid", "test"]
CLASSES_TO_SKIP = {"Human", "Beacon", "Bridge", "Buoy", "Ship"}

notebook_dir = os.getcwd()
BASE_DIR = "/home/mysiek44/Documents/sources/szymon/datasets_coco"

OUT_DIR = BASE_DIR

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def load_annotations():
    rows = []
    categories = {}

    for u in ENABLED_UNIVERSES:
        for split in SPLITS:
            ann_path = os.path.join(BASE_DIR, f"universe_{u}", split, "_annotations_unified.coco.json")
            if not os.path.exists(ann_path):
                continue

            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # kategorie
            for c in data.get("categories", []):
                categories[c["id"]] = c["name"]

            # mapowanie obrazów
            imgs = {i["id"]: i for i in data.get('images', [])}

            # bboxy
            for ann in data.get("annotations", []):
                img_info = imgs.get(ann["image_id"])
                if not img_info:
                    continue

                rows.append({
                    "image_path": os.path.join(BASE_DIR, f"universe_{u}", split, img_info["file_name"]),
                    "dataset": split,
                    "x": ann["bbox"][0],
                    "y": ann["bbox"][1],
                    "width": ann["bbox"][2],
                    "height": ann["bbox"][3],
                    "class_id": ann["category_id"],
                })

    return pd.DataFrame(rows), categories


def add_image_dims(df):
    df = df.copy()
    df["w_img"] = 0
    df["h_img"] = 0

    for p in df["image_path"].unique():
        try:
            with Image.open(p) as im:
                w, h = im.size
        except:
            continue

        df.loc[df.image_path == p, ["w_img", "h_img"]] = (w, h)

    return df[df.w_img > 0]


def create_yolo_dirs():
    for s in SPLITS:
        Path(os.path.join(OUT_DIR, s, "images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(OUT_DIR, s, "labels")).mkdir(parents=True, exist_ok=True)


def clear_yolo_dirs():
    for s in SPLITS:
        for f in Path(os.path.join(OUT_DIR, s, "images")).glob("*.*"):
            f.unlink()
        for f in Path(os.path.join(OUT_DIR, s, "labels")).glob("*.txt"):
            f.unlink()


def save_labels(df, class_to_id):
    for _, r in df.iterrows():
        if r["class_name"] not in class_to_id:
            continue

        split = r["dataset"]
        dst_img = os.path.join(OUT_DIR, split, "images", os.path.basename(r["image_path"]))
        shutil.copy(r["image_path"], dst_img)

        # YOLO bbox
        xc = (r["x"] + r["width"] / 2) / r["w_img"]
        yc = (r["y"] + r["height"] / 2) / r["h_img"]
        w = r["width"] / r["w_img"]
        h = r["height"] / r["h_img"]

        label_path = os.path.join(OUT_DIR, split, "labels", Path(r["image_path"]).stem + ".txt")

        with open(label_path, "a") as f:
            f.write(f"{class_to_id[r['class_name']]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def write_yaml(class_names):
    data = {
        "train": os.path.join(OUT_DIR, "train/images"),
        "val": os.path.join(OUT_DIR, "valid/images"),
        "test": os.path.join(OUT_DIR, "test/images"),
        "nc": len(class_names),
        "names": class_names
    }

    with open(os.path.join(OUT_DIR, "data_filtered.yaml"), "w") as f:
        yaml.dump(data, f)

    print("\nUtworzono data_filtered.yaml")

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    print("Ładowanie adnotacji COCO…")
    df, categories = load_annotations()

    print("Usuwam klasy:", CLASSES_TO_SKIP)
    df["class_name"] = df["class_id"].map(categories)
    df = df[~df["class_name"].isin(CLASSES_TO_SKIP)]

    print("Wczytywanie wymiarów obrazów…")
    df = add_image_dims(df)

    print("Przygotowanie folderów YOLO…")
    create_yolo_dirs()
    clear_yolo_dirs()

    # klasy YOLO
    class_names = sorted(df["class_name"].unique())
    class_to_id = {c: i for i, c in enumerate(class_names)}

    print("Zapisywanie etykiet i kopiowanie obrazów…")
    save_labels(df, class_to_id)

    print("Tworzenie pliku YAML…")
    write_yaml(class_names)

    print("\n=== Dataset gotowy ===")


if __name__ == "__main__":
    main()
