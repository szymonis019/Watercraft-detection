import os
import gc
import time
import json
import torch
import cv2
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.cuda.amp import GradScaler
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ---------------------------------------------------
# CONFIG 
# ---------------------------------------------------
BATCH = 4       
EPOCHS = 20
WORKERS = 8    
ENABLED_UNIVERSES = [1, 2, 3, 4, 5, 7] 
PROJECT = "training_output"
EXP_NAME = "faster_rcnn_resnet50_v2"
CLASSES_TO_SKIP = {"Human", "Beacon", "Bridge", "Buoy", "Ship"}

TRAIN_LIMIT = "all"
VAL_LIMIT = "all"
USE_AMP = True 

# ---------------------------------------------------
# PATH
# ---------------------------------------------------
if '__file__' not in locals():
    CURRENT_DIR = os.getcwd()
else:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
BASE_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "datasets_coco")
OUTPUT_DIR = os.path.join(CURRENT_DIR, PROJECT, EXP_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# Deivice 
# ---------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(f"Wykryto GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Brak GPU. Trening na CPU.")
        return torch.device("cpu")

# ---------------------------------------------------
# HELPERS 
# ---------------------------------------------------
def load_and_filter_coco_data(json_path, image_root):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_cats = data.get('categories', [])
    valid_cats = [c for c in original_cats if c['name'] not in CLASSES_TO_SKIP]

    if not valid_cats:
        return [], []

    cat_id_map = {c['id']: new_id for new_id, c in enumerate(valid_cats)}
    valid_class_names = [c['name'] for c in valid_cats]

    images_info = {img['id']: img for img in data.get('images', [])}
    annotations = data.get('annotations', [])

    img_to_anns = {}
    
    for ann in annotations:
        if ann['category_id'] in cat_id_map:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            ann_copy = ann.copy()
            ann_copy['new_category_id'] = cat_id_map[ann['category_id']]
            img_to_anns[img_id].append(ann_copy)

    dataset_dicts = []
    for img_id, anns in img_to_anns.items():
        img_data = images_info.get(img_id)
        if img_data:
            record = {
                "file_path": os.path.join(image_root, img_data["file_name"]),
                "height": img_data["height"],
                "width": img_data["width"],
                "annotations": anns,
                "image_id": img_id
            }
            dataset_dicts.append(record)

    return dataset_dicts, valid_class_names

def collect_all_datasets():
    all_data = []
    final_classes = []
    print(f"Zbieranie danych (Skip: {CLASSES_TO_SKIP})")
    for u in ENABLED_UNIVERSES:
        u_path = os.path.join(BASE_DIR, f"universe_{u}")
        splits = [
            (os.path.join(u_path, "train", "_annotations_unified.coco.json"), os.path.join(u_path, "train")),
            (os.path.join(u_path, "valid", "_annotations_unified.coco.json"), os.path.join(u_path, "valid"))
        ]
        for json_p, img_p in splits:
            if os.path.exists(json_p):
                data, classes = load_and_filter_coco_data(json_p, img_p)
                all_data.extend(data)
                if not final_classes and classes:
                    final_classes = classes
    print(f"Łącznie zebrano: {len(all_data)} obrazów.")
    return all_data, final_classes

def split_dataset(full_data, train_lim, val_lim):
    random.shuffle(full_data)
    t_len = len(full_data) if train_lim == "all" else min(int(train_lim), len(full_data))
    train_subset = full_data[:t_len]
    remaining = full_data[t_len:]
    v_len = len(remaining) if val_lim == "all" else min(int(val_lim), len(remaining))
    val_subset = remaining[:v_len]
    print(f"Dataset split -> Train: {len(train_subset)} | Valid: {len(val_subset)}")
    return train_subset, val_subset

# ---------------------------------------------------
# DATASET 
# ---------------------------------------------------
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class ShipDataset(Dataset):
    def __init__(self, data_list, transforms=None):
        self.data = data_list
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = cv2.imread(item["file_path"])
        if img is None:
            img = np.zeros((item["height"], item["width"], 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = []
        labels = []
        for ann in item["annotations"]:
            x, y, w, h = ann["bbox"]
            if w > 1 and h > 1: 
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['new_category_id'] + 1)
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([item["image_id"]])
        if len(target["boxes"]) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
        if self.transforms:
            img = self.transforms(img)
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ---------------------------------------------------
# MODEL 
# ---------------------------------------------------
def get_model(num_classes):
    print("Ładowanie Faster R-CNN ResNet50 V2...")
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    model.train()
    total_loss = 0
    steps = 0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        steps += 1
    return total_loss / max(steps, 1)

# mAP ---
@torch.no_grad()
def evaluate_map(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        preds = model(images)
        metric.update(preds, targets)
        
    result = metric.compute()
    return result

# ---------------------------------------------------
# MAIN 
# ---------------------------------------------------
def main():
    gc.collect()
    torch.cuda.empty_cache()
    device = get_device()

    full_data_raw, class_names = collect_all_datasets()
    if not full_data_raw: return
    print(f"Klasy: {class_names}")
    num_classes = len(class_names) + 1 

    train_data, val_data = split_dataset(full_data_raw, TRAIN_LIMIT, VAL_LIMIT)
    
    train_dataset = ShipDataset(train_data, transforms=get_transform(train=True))
    val_dataset = ShipDataset(val_data, transforms=get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, 
                              num_workers=WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, 
                            num_workers=WORKERS, collate_fn=collate_fn)

    model = get_model(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    scaler = GradScaler() if USE_AMP else None

    print(f"\n=== START TRENINGU: {EPOCHS} epok, Batch {BATCH}, AMP={USE_AMP} ===")
    start_time = time.time()

    total_train_images = len(train_data)
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # Trenowanie
        model.train()
        epoch_loss = 0
        print(f"\nEpoka {epoch+1}/{EPOCHS} [LR: {optimizer.param_groups[0]['lr']:.6f}]")
        
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            if USE_AMP:
                with torch.amp.autocast('cuda'):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                scaler.scale(losses).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            epoch_loss += losses.item()
            
            processed_imgs = min((i + 1) * BATCH, total_train_images)
            pct = (processed_imgs / total_train_images) * 100
                
            print(f"  Iter {i}/{len(train_loader)} | Imgs: {processed_imgs}/{total_train_images} ({pct:.1f}%) | Loss: {losses.item():.4f}")

        lr_scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        print(f" >> Train Loss: {avg_train_loss:.4f}")
        
        # Strata i mAP
        avg_val_loss = float('nan')
        if len(val_data) > 0:
            # Obliczanie straty (Validation Loss)
            avg_val_loss = evaluate_loss(model, val_loader, device)
            print(f" >> Valid Loss: {avg_val_loss:.4f}")
            
            # mAP na zbiorze walidacyjnym 
            print(" >> Obliczanie mAP (to może chwilę potrwać)...")
            map_stats = evaluate_map(model, val_loader, device)
            print(f" >> mAP@50:    {map_stats['map_50']:.4f}")
            print(f" >> mAP@50-95: {map_stats['map']:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(OUTPUT_DIR, "model_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"   [Zapisano BEST]: Nowa najniższa strata: {best_val_loss:.4f}")

        last_path = os.path.join(OUTPUT_DIR, f"model_last.pth")
        torch.save(model.state_dict(), last_path)
        
        if (epoch+1) % 5 == 0:
             epoch_path = os.path.join(OUTPUT_DIR, f"model_ep{epoch+1}.pth")
             torch.save(model.state_dict(), epoch_path)
             print(f"   [Zapisano CHECKPOINT]: {epoch_path}")

    elapsed = (time.time() - start_time) / 60
    print(f"\nTrening zakończony w {elapsed:.1f} min.")
    print(f"Wagi dostępne w: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()