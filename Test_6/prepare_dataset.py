import os
import json
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import yaml

ENABLED_UNIVERSES = [1, 2, 3, 4, 5, 7]
SPLITS = ["train", "valid", "test"]
CLASSES_TO_SKIP = {"Human", "Beacon", "Bridge", "Buoy", "Ship"}

BASE_DIR = "D:/Watercraft-detection-main/datasets_coco"
OUT_DIR = "D:/Watercraft-detection-main/Test_6/dataset_yolo"

def load_annotations():
    rows = []
    categories = {}
    for u in ENABLED_UNIVERSES:
        for split in SPLITS:
            u_dir = os.path.join(BASE_DIR, f"universe_{u}", split)
            ann_path = os.path.join(u_dir, "_annotations.coco.json")

            if not os.path.exists(ann_path):
                continue

            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for c in data.get("categories", []):
                categories[c["id"]] = c["name"]

            imgs = {i["id"]: i for i in data.get('images', [])}

            for ann in data.get("annotations", []):
                img_info = imgs.get(ann["image_id"])
                if not img_info:
                    continue

                rows.append({
                    "image_path": os.path.join(u_dir, img_info["file_name"]),
                    "dataset": split if split != "valid" else "val",
                    "x": ann["bbox"][0],
                    "y": ann["bbox"][1],
                    "width": ann["bbox"][2],
                    "height": ann["bbox"][3],
                    "class_id": ann["category_id"],
                })

    if not rows:
        return pd.DataFrame(columns=["image_path", "dataset", "x", "y", "width", "height", "class_id"]), categories

    return pd.DataFrame(rows), categories

def add_image_dims(df):
    if df.empty: return df
    df = df.copy()
    df["w_img"] = 0
    df["h_img"] = 0
    for p in df["image_path"].unique():
        try:
            with Image.open(p) as im:
                w, h = im.size
            df.loc[df.image_path == p, ["w_img", "h_img"]] = (w, h)
        except:
            continue
    return df[df.w_img > 0]

def create_yolo_dirs():
    for s in ["train", "val", "test"]:
        Path(os.path.join(OUT_DIR, s, "images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(OUT_DIR, s, "labels")).mkdir(parents=True, exist_ok=True)

def clear_yolo_dirs():
    for s in ["train", "val", "test"]:
        for f in Path(os.path.join(OUT_DIR, s, "images")).glob("*.*"): f.unlink()
        for f in Path(os.path.join(OUT_DIR, s, "labels")).glob("*.txt"): f.unlink()

def save_labels(df, class_to_id):
    for _, r in df.iterrows():
        if r["class_name"] not in class_to_id:
            continue
        split = r["dataset"]
        dst_img = os.path.join(OUT_DIR, split, "images", os.path.basename(r["image_path"]))
        if not os.path.exists(dst_img):
            shutil.copy2(r["image_path"], dst_img)

        xc = (r["x"] + r["width"] / 2) / r["w_img"]
        yc = (r["y"] + r["height"] / 2) / r["h_img"]
        w = r["width"] / r["w_img"]
        h = r["height"] / r["h_img"]

        label_path = os.path.join(OUT_DIR, split, "labels", Path(r["image_path"]).stem + ".txt")
        with open(label_path, "a") as f:
            f.write(f"{class_to_id[r['class_name']]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def write_yaml(class_names):
    data = {
        "path": os.path.abspath(OUT_DIR),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names
    }
    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        yaml.dump(data, f)

def main():
    df, categories = load_annotations()
    if df.empty:
        return

    df["class_name"] = df["class_id"].map(categories)
    df = df[~df["class_name"].isin(CLASSES_TO_SKIP)]
    df = add_image_dims(df)

    create_yolo_dirs()
    clear_yolo_dirs()

    class_names = sorted(df["class_name"].unique())
    class_to_id = {c: i for i, c in enumerate(class_names)}

    save_labels(df, class_to_id)
    write_yaml(class_names)

if __name__ == "__main__":
    main()