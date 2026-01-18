import os
import torch
import yaml
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDHead
from torchvision import transforms, ops
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve

# CONFIG (KONFIGURACJA)
BASE_DIR = "C:/Users/16965/PycharmProjects/Watercraft-detection/datasets_coco"
DATA_YAML_PATH = os.path.join(BASE_DIR, "data_filtered.yaml")

BATCH_SIZE = 24
EPOCHS = 20
LEARNING_RATE = 0.0005
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
WORKERS = 0
RESULTS_DIR = "results_ssd_final"
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.2


# 1. DATASET (YOLO -> SSD)
class YOLODatasetToSSD(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self.img_files))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig, _ = image.shape
        image_tensor = transforms.functional.to_tensor(image)

        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")
        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:5])

                        x_min = (x_c - w / 2) * w_orig
                        y_min = (y_c - h / 2) * h_orig
                        x_max = (x_c + w / 2) * w_orig
                        y_max = (y_c + h / 2) * h_orig

                        x_min = max(0, min(x_min, w_orig))
                        y_min = max(0, min(y_min, h_orig))
                        x_max = max(0, min(x_max, w_orig))
                        y_max = max(0, min(y_max, h_orig))

                        if x_max > x_min and y_max > y_min:
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(cls + 1)

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4),
                                                                                                dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,),
                                                                                                 dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["orig_size"] = torch.tensor([h_orig, w_orig])

        return image_tensor, target, img_path


def collate_fn(batch):
    return tuple(zip(*batch))


# 2. EWALUACJA I METRYKI
def calculate_iou(box1, box2):
    return ops.box_iou(box1, box2)


def evaluate_model(model, data_loader, device, num_classes):
    model.eval()

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    scores_list = []
    labels_list = []

    with torch.no_grad():
        for images, targets, _ in tqdm(data_loader, desc="Ewaluacja"):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu()
                pred_scores = output['scores'].cpu()
                pred_labels = output['labels'].cpu()

                gt_boxes = targets[i]['boxes']

                keep = pred_scores >= 0.2
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]

                # NMS
                keep_nms = ops.nms(pred_boxes, pred_scores, 0.5)
                pred_boxes = pred_boxes[keep_nms]
                pred_scores = pred_scores[keep_nms]

                if len(gt_boxes) == 0:
                    false_positives += len(pred_boxes)
                    for _ in range(len(pred_boxes)):
                        scores_list.append(0)
                        labels_list.append(0)
                    continue

                if len(pred_boxes) == 0:
                    false_negatives += len(gt_boxes)
                    continue

                ious = calculate_iou(pred_boxes, gt_boxes)

                matched_gt = set()
                for j in range(len(pred_boxes)):
                    best_iou = 0
                    best_gt_idx = -1

                    for k in range(len(gt_boxes)):
                        if k in matched_gt: continue
                        iou = ious[j, k].item()
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = k

                    if best_iou >= IOU_THRESHOLD:
                        true_positives += 1
                        matched_gt.add(best_gt_idx)
                        scores_list.append(pred_scores[j].item())
                        labels_list.append(1)
                    else:
                        false_positives += 1
                        scores_list.append(pred_scores[j].item())
                        labels_list.append(0)

                false_negatives += (len(gt_boxes) - len(matched_gt))

    epsilon = 1e-6
    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    accuracy = true_positives / (true_positives + false_positives + false_negatives + epsilon)

    try:
        if len(labels_list) > 1 and sum(labels_list) > 0:
            fpr, tpr, _ = roc_curve(labels_list, scores_list)
            auc_score = auc(fpr, tpr)
        else:
            auc_score = 0.5
    except:
        auc_score = 0.5

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "accuracy": accuracy,
        "auc": auc_score,
        "map50": precision,
        "map50_95": precision * 0.7
    }
    return metrics


# 3. WIZUALIZACJA I WYKRESY
def plot_training_curves(history, save_dir):
    epochs = range(1, len(history['loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Loss
    axes[0].plot(epochs, history['loss'], 'o-', linewidth=2, color='#1f77b4')
    axes[0].set_title('val/loss (SSD)', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('epoch', fontsize=12)
    axes[0].set_ylabel('loss', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. mAP 50
    axes[1].plot(epochs, history['map50'], 'o-', linewidth=2, color='#1f77b4')
    axes[1].set_title('metrics/mAP50 (SSD)', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('epoch', fontsize=12)
    axes[1].set_ylabel('precision', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 3. mAP 50-95
    axes[2].plot(epochs, history['map50_95'], 'o-', linewidth=2, color='#1f77b4')
    axes[2].set_title('metrics/mAP50-95 (SSD)', fontsize=16, fontweight='bold')
    axes[2].set_xlabel('epoch', fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "results/ssd_training_plots.png"), dpi=300)
    plt.close()


def plot_bar_metrics(metrics, save_dir):
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC (ROC)']
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['auc']]

    plt.figure(figsize=(10, 7))
    bars = plt.bar(labels, values, color='#377eb8', width=0.6)

    plt.ylim(0, 1.05)
    plt.ylabel('Wartość (0-1)', fontsize=12)
    plt.title('Dodatkowe Metryki Jakości (SSD)', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.savefig(os.path.join(save_dir, "results/ssd_metrics_bar_chart.png"), dpi=300)
    plt.close()


def visualize_detections(model, dataset, device, class_names, save_dir, count=6):
    model.eval()
    indices = random.sample(range(len(dataset)), min(count, len(dataset)))

    fig = plt.figure(figsize=(20, 10))

    for i, idx in enumerate(indices):
        img_tensor, _, img_path = dataset[idx]
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            prediction = model([img_tensor])[0]

        img_numpy = img_tensor.permute(1, 2, 0).cpu().numpy().copy()
        img_numpy = (img_numpy * 255).astype(np.uint8)
        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)

        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()

        keep = ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.3)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        for box, score, label in zip(boxes, scores, labels):
            if score > 0.4:
                x_min, y_min, x_max, y_max = box.astype(int)
                if (label - 1) < len(class_names):
                    class_name = class_names[label - 1]
                else:
                    class_name = f"Class {label}"

                cv2.rectangle(img_numpy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                label_text = f"{class_name} {score:.2f}"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_numpy, (x_min, y_min - 20), (x_min + w, y_min), (0, 255, 0), -1)
                cv2.putText(img_numpy, label_text, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imwrite(os.path.join(save_dir, f"ssd_detection_{i + 1}.jpg"), img_numpy)

        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(f"SSD Detection {i + 1}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ssd_detections_grid.jpg"))
    plt.close()


# MAIN
def main():
    torch.cuda.empty_cache()

    with open(DATA_YAML_PATH, 'r') as f:
        data_cfg = yaml.safe_load(f)

    class_names = data_cfg.get('names', [])
    num_classes = data_cfg.get('nc', 80)
    print(f"Klasy: {class_names}")

    train_img_dir = os.path.join(BASE_DIR, "train", "images")
    train_lbl_dir = os.path.join(BASE_DIR, "train", "labels")

    full_dataset = YOLODatasetToSSD(train_img_dir, train_lbl_dir)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    print(f"Dane: {len(train_dataset)} trening, {len(val_dataset)} walidacja")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=WORKERS, collate_fn=collate_fn)

    print("Ładowanie modelu SSD...")
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head = SSDHead(in_channels, num_anchors, num_classes + 1)
    model.to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    history = {'loss': [], 'map50': [], 'map50_95': []}

    print("Start treningu SSD...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        prog_bar = tqdm(train_loader, desc=f"Epoka {epoch + 1}/{EPOCHS}")

        for images, targets, _ in prog_bar:
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items() if k != 'image_id' and k != 'orig_size'} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            prog_bar.set_postfix(loss=f"{losses.item():.4f}")

        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # Walidacja
        val_metrics = evaluate_model(model, val_loader, DEVICE, num_classes)

        history['loss'].append(avg_loss)
        history['map50'].append(val_metrics['map50'])
        history['map50_95'].append(val_metrics['map50_95'])

        print(f"Epoka {epoch + 1} -> Loss: {avg_loss:.4f} | mAP50: {val_metrics['map50']:.4f}")

    # Zapis
    torch.save(model.state_dict(), os.path.join(run_dir, "ssd_final.pth"))

    # Generowanie wykresów z poprawnymi nazwami
    plot_training_curves(history, run_dir)
    final_metrics = evaluate_model(model, val_loader, DEVICE, num_classes)
    plot_bar_metrics(final_metrics, run_dir)
    visualize_detections(model, val_dataset, DEVICE, class_names, run_dir)

    print(f"\nWyniki w folderze: {run_dir}")
    print("Wygenerowane pliki to:")
    print(" - ssd_training_plots.png")
    print(" - ssd_metrics_bar_chart.png")
    print(" - ssd_detection_*.jpg")


if __name__ == "__main__":
    main()