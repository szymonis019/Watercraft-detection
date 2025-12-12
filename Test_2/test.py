import os
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from sklearn.metrics import roc_auc_score
import json

# ---------------------------------------------------
# KONFIGURACJA TESTU
# ---------------------------------------------------
MODELS_TO_TEST = ["model_ep5.pth", "model_ep10.pth", "model_ep15.pth", "model_last.pth"]
EXP_DIR = "training_output"
TEST_LIMIT = "all"

BATCH_SIZE = 4
WORKERS = 2  
CONF_THRESHOLD = 0.5 # Próg pewności do akceptacji detekcji
IOU_THRESHOLD = 0.5  # Próg nakładania się ramek

ENABLED_UNIVERSES = [1, 2, 3, 4, 5, 7] 

CLASSES_TO_SKIP_LOADING = {"Human", "Beacon", "Bridge", "Buoy"} 
CLASSES_TO_HIDE = {"Ship", "Shp", "Unknown Object"}

# ---------------------------------------------------
# ŚCIEŻKI
# ---------------------------------------------------
if '__file__' not in locals():
    CURRENT_DIR = os.getcwd()
else:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
BASE_DIR = os.path.join(PROJECT_ROOT, "datasets_coco")

print(f"Lokalizacja skryptu: {CURRENT_DIR}")
print(f"Szukam datasetu w:   {BASE_DIR}")
print(f"Szukam modeli w:     {os.path.join(CURRENT_DIR, EXP_DIR)}")

# ---------------------------------------------------
# FUNKCJE POMOCNICZE (DATASET & MODEL)
# ---------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_filter_coco_data(json_path, image_root):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model_classes = collect_train_classes()
    class_name_to_id = {name: i for i, name in enumerate(model_classes)}
    json_cat_id_to_name = {c['id']: c['name'] for c in data.get('categories', [])}
    images_info = {img['id']: img for img in data.get('images', [])}
    annotations = data.get('annotations', [])

    img_to_anns = {}
    for ann in annotations:
        cat_id_json = ann['category_id']
        if cat_id_json not in json_cat_id_to_name: continue
        class_name = json_cat_id_to_name[cat_id_json]
        if class_name in CLASSES_TO_SKIP_LOADING: continue
        if class_name not in class_name_to_id: continue

        img_id = ann['image_id']
        if img_id not in img_to_anns: img_to_anns[img_id] = []
        ann_copy = ann.copy()
        ann_copy['new_category_id'] = class_name_to_id[class_name]
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
    return dataset_dicts

def collect_train_classes():
    classes = [
        'Aircraft carrier', 'Boat', 'Destroyer', 'Fishing boat', 
        'General cargo ship', 'Landing ship', 'Passenger ship', 
        'Sail boat', 'Ship', 'Shp', 'Speedboat', 'Submarine', 'Unknown Object'
    ]
    return sorted(classes)

def collect_test_data():
    all_data = []
    print(f"Zbieranie danych")
    for u in ENABLED_UNIVERSES:
        u_path = os.path.join(BASE_DIR, f"universe_{u}")
        splits = [(os.path.join(u_path, "test", "_annotations_unified.coco.json"), os.path.join(u_path, "test"))]
        for json_p, img_p in splits:
            if os.path.exists(json_p):
                data = load_and_filter_coco_data(json_p, img_p)
                all_data.extend(data)
    print(f"Zebrano {len(all_data)} zdjęć testowych (Total).")
    return all_data

class ShipDataset(Dataset):
    def __init__(self, data_list, transforms=None):
        self.data = data_list
        self.transforms = transforms
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img = cv2.imread(item["file_path"])
        if img is None: img = np.zeros((item["height"], item["width"], 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = []; labels = []
        for ann in item["annotations"]:
            x, y, w, h = ann["bbox"]
            if w > 1 and h > 1: 
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['new_category_id'] + 1)
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([item["image_id"]])
        if self.transforms: img = self.transforms(img)
        return img, target

def get_transform(): return T.Compose([T.ToTensor()])
def collate_fn(batch): return tuple(zip(*batch))

def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ---------------------------------------------------
# LOGIKA OBLICZANIA METRYK
# ---------------------------------------------------
def calculate_batch_stats(preds, targets, iou_thresh=0.5):
    tp, fp, fn = 0, 0, 0
    y_true = []
    y_scores = []

    for pred, target in zip(preds, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        target_boxes = target['boxes']

        if len(target_boxes) == 0:
            fp += len(pred_boxes)
            y_true.extend([0] * len(pred_boxes))
            y_scores.extend(pred_scores.cpu().tolist())
            continue

        if len(pred_boxes) == 0:
            fn += len(target_boxes)
            continue

        iou_matrix = box_iou(pred_boxes, target_boxes)

        matched_gt_indices = set()
        
        sorted_indices = torch.argsort(pred_scores, descending=True)
        
        for p_idx in sorted_indices:
            p_idx = p_idx.item()
            max_iou, gt_idx = torch.max(iou_matrix[p_idx], dim=0)
            max_iou = max_iou.item()
            gt_idx = gt_idx.item()

            if max_iou >= iou_thresh and gt_idx not in matched_gt_indices:
                tp += 1
                matched_gt_indices.add(gt_idx)
                y_true.append(1)
                y_scores.append(pred_scores[p_idx].item())
            else:
                fp += 1
                y_true.append(0)
                y_scores.append(pred_scores[p_idx].item())

        fn += len(target_boxes) - len(matched_gt_indices)

    return tp, fp, fn, y_true, y_scores

def filter_results(outputs, class_names):
    filtered_outputs = []
    hide_ids = [i + 1 for i, name in enumerate(class_names) if name in CLASSES_TO_HIDE]
    for out in outputs:
        keep = [i for i, label in enumerate(out['labels']) if label.item() not in hide_ids]
        if not keep:
            filtered_outputs.append({
                'boxes': torch.zeros((0, 4), device=out['boxes'].device),
                'scores': torch.tensor([], device=out['scores'].device) if 'scores' in out else None,
                'labels': torch.tensor([], device=out['labels'].device, dtype=torch.int64)
            })
        else:
            keep = torch.as_tensor(keep, device=out['boxes'].device)
            new_dict = {'boxes': out['boxes'][keep], 'labels': out['labels'][keep]}
            if 'scores' in out: new_dict['scores'] = out['scores'][keep]
            filtered_outputs.append(new_dict)
    return filtered_outputs

def visualize_predictions(image_tensor, prediction, target, class_names, conf_thresh=0.5):
    image = image_tensor.permute(1, 2, 0).cpu().numpy().copy()
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hide_set = set(CLASSES_TO_HIDE)

    # Predykcje (Zielone)
    for box, score, label in zip(prediction['boxes'].cpu().numpy(), prediction['scores'].cpu().numpy(), prediction['labels'].cpu().numpy()):
        if score >= conf_thresh:
            idx = label - 1
            class_name = class_names[idx] if 0 <= idx < len(class_names) else str(label)
            if class_name in hide_set: continue
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{class_name} {score:.2f}", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    device = get_device()
    print(f"Device: {device}")

    class_names = collect_train_classes()
    test_data = collect_test_data()

    if not test_data:
        print("BŁĄD: Brak danych testowych!")
        return
# ---------------------------------------------------
# LIMIT
# ---------------------------------------------------
    if TEST_LIMIT != "all":
        try:
            limit = int(TEST_LIMIT)
            if limit < len(test_data):
                print(f"\n[INFO] Limit: {limit} losowych zdjęć.")
                random.seed(42)
                random.shuffle(test_data)
                test_data = test_data[:limit]
        except: pass

    num_classes = len(class_names) + 1 
    test_dataset = ShipDataset(test_data, transforms=get_transform())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=WORKERS, collate_fn=collate_fn)

    results_table = []
    metric_map = MeanAveragePrecision(iou_type="bbox", class_metrics=False)

    for model_name in MODELS_TO_TEST:
        model_path = os.path.join(CURRENT_DIR, EXP_DIR, model_name)
        if not os.path.exists(model_path):
            print(f"Brak pliku: {model_path}"); continue

        print(f"\n--- Testowanie: {model_name} ---")
        model = get_model(num_classes)
        
        try:
            map_loc = device if device.type == 'cpu' else None
            model.load_state_dict(torch.load(model_path, map_location=map_loc))
        except RuntimeError as e:
            print(f"BŁĄD WAG: {e}"); continue

        model.to(device)
        model.eval()
        metric_map.reset()
        total_tp, total_fp, total_fn = 0, 0, 0
        all_y_true = []
        all_y_scores = []

        vis_images = []
        total_batches = len(test_loader)

        with torch.no_grad():
            for i, (images, targets) in enumerate(test_loader):
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                preds = model(images)
                
                # Filtrowanie klas
                filtered_preds = filter_results(preds, class_names)
                filtered_targets = filter_results(targets, class_names)
                
                # Aktualizacja mAP
                metric_map.update(filtered_preds, filtered_targets)

                thresh_preds = []
                for p in filtered_preds:
                    keep = p['scores'] >= CONF_THRESHOLD
                    thresh_preds.append({
                        'boxes': p['boxes'][keep],
                        'scores': p['scores'][keep],
                        'labels': p['labels'][keep]
                    })

                tp, fp, fn, y_true, y_scores = calculate_batch_stats(thresh_preds, filtered_targets, IOU_THRESHOLD)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                all_y_true.extend(y_true)
                all_y_scores.extend(y_scores)

                # Wizualizacja
                if len(vis_images) < 5:
                    for img, pred, tgt in zip(images, filtered_preds, filtered_targets):
                        if len(vis_images) < 5:
                            vis_images.append(visualize_predictions(img, pred, tgt, class_names, CONF_THRESHOLD))
                
                if i % 10 == 0:
                    print(f" Batch {i+1}/{total_batches}...")
# ---------------------------------------------------
# OBLICZANIE WYNIKÓW KOŃCOWYCH
# ---------------------------------------------------
        print(" Obliczanie metryk...")
        
        # mAP
        res_map = metric_map.compute()
        map50 = res_map['map_50'].item()
        map50_95 = res_map['map'].item()

        epsilon = 1e-7
        precision = total_tp / (total_tp + total_fp + epsilon)
        recall = total_tp / (total_tp + total_fn + epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

        accuracy = total_tp / (total_tp + total_fp + total_fn + epsilon)

        try:
            if len(all_y_true) > 0 and len(set(all_y_true)) > 1:
                auc_score = roc_auc_score(all_y_true, all_y_scores)
            else:
                auc_score = 0.0
        except:
            auc_score = 0.0

        print(f" >> mAP@50: {map50:.4f}")
        print(f" >> F1: {f1_score:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")
        
        results_table.append({
            "Model": model_name,
            "mAP@50": map50,
            "mAP@50-95": map50_95,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score,
            "AUC (ROC)": auc_score
        })

        if vis_images:
            cols = min(len(vis_images), 5)
            fig, axs = plt.subplots(1, cols, figsize=(20, 5))
            if cols == 1: axs = [axs]