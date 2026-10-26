import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import segmentation_models_pytorch as smp
from dataset import SegmentationDataset
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc
from tqdm import tqdm

# =============================
# 유틸 함수
# =============================
def squeeze_tensor(x):
    return x.squeeze()

# =============================
# AP 계산 함수
# =============================
def calculate_ap_at_iou(preds, masks, iou_thresholds):
    aps = []
    for thr in iou_thresholds:
        # IoU 계산
        intersection = np.logical_and(preds, masks).sum()
        union = np.logical_or(preds, masks).sum()
        iou = intersection / union if union > 0 else 0.0

        if iou >= thr:
            y_true = masks.reshape(-1)
            y_pred = preds.reshape(-1)
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            ap = auc(recall, precision)
        else:
            ap = 0.0
        aps.append(ap)
    return np.mean(aps)

# =============================
# Metric 함수 (클래스별 포함)
# =============================
def calculate_metrics(outputs, masks, num_classes):
    preds = torch.argmax(outputs, dim=1)
    preds = preds.cpu().numpy().astype(np.int32)
    masks = masks.cpu().numpy().astype(np.int32)

    pixel_acc = (preds == masks).sum() / np.prod(masks.shape)

    class_metrics = {}
    iou_list, precision_list, recall_list, f1_list = [], [], [], []
    ap50_list, ap5095_list = [], []

    for cls in range(num_classes):
        cls_preds = (preds == cls).astype(np.uint8)
        cls_masks = (masks == cls).astype(np.uint8)

        # IoU
        intersection = np.logical_and(cls_preds, cls_masks).sum()
        union = np.logical_or(cls_preds, cls_masks).sum()
        iou = intersection / union if union > 0 else np.nan
        iou_list.append(iou)

        # Precision / Recall / F1
        precision = precision_score(cls_masks.reshape(-1), cls_preds.reshape(-1), zero_division=0)
        recall = recall_score(cls_masks.reshape(-1), cls_preds.reshape(-1), zero_division=0)
        f1 = f1_score(cls_masks.reshape(-1), cls_preds.reshape(-1), zero_division=0)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        # AP@50
        ap50 = calculate_ap_at_iou(cls_preds, cls_masks, [0.5])
        # AP@50-95
        ap5095 = calculate_ap_at_iou(cls_preds, cls_masks, np.arange(0.5, 1.0, 0.05))

        ap50_list.append(ap50)
        ap5095_list.append(ap5095)

        class_metrics[cls] = {
            "IoU": iou,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AP50": ap50,
            "AP50-95": ap5095
        }

    mean_metrics = {
        "Pixel Accuracy": pixel_acc,
        "Mean IoU": np.nanmean(iou_list),
        "Mean Precision": np.mean(precision_list),
        "Mean Recall": np.mean(recall_list),
        "Mean F1": np.mean(f1_list),
        "mAP50": np.nanmean(ap50_list),
        "mAP50-95": np.nanmean(ap5095_list)
    }

    return mean_metrics, class_metrics

# =============================
# 평가 함수
# =============================
def evaluate(model, dataloader, device, num_classes):
    model.eval()
    all_class_metrics = {cls: {"IoU": [], "Precision": [], "Recall": [], "F1": [], "AP50": [], "AP50-95": []} for cls in range(num_classes)}
    pixel_accs, mean_ious, mean_precisions, mean_recalls, mean_f1s, ap50s, ap5095s = [], [], [], [], [], [], []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            mean_metrics, class_metrics = calculate_metrics(outputs, masks, num_classes)

            pixel_accs.append(mean_metrics["Pixel Accuracy"])
            mean_ious.append(mean_metrics["Mean IoU"])
            mean_precisions.append(mean_metrics["Mean Precision"])
            mean_recalls.append(mean_metrics["Mean Recall"])
            mean_f1s.append(mean_metrics["Mean F1"])
            ap50s.append(mean_metrics["mAP50"])
            ap5095s.append(mean_metrics["mAP50-95"])

            for cls in range(num_classes):
                for metric in class_metrics[cls]:
                    all_class_metrics[cls][metric].append(class_metrics[cls][metric])

    # 전체 평균
    results = {
        "Pixel Accuracy": np.mean(pixel_accs),
        "Mean IoU": np.mean(mean_ious),
        "Mean Precision": np.mean(mean_precisions),
        "Mean Recall": np.mean(mean_recalls),
        "Mean F1": np.mean(mean_f1s),
        "mAP50": np.mean(ap50s),
        "mAP50-95": np.mean(ap5095s)
    }

    # 클래스별 평균
    class_results = {}
    for cls in range(num_classes):
        class_results[cls] = {metric: np.nanmean(all_class_metrics[cls][metric]) for metric in all_class_metrics[cls]}

    return results, class_results

# =============================
# Main
# =============================
def main():
    num_classes = 3
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.NEAREST),
        transforms.PILToTensor(),
        transforms.Lambda(squeeze_tensor)
    ])

    # 평가용 데이터셋
    eval_dataset = SegmentationDataset(
        image_dir= r"C:\Users\dlxlr\Desktop\Dataset\Train_2\test\test_40_\val\images",
        mask_dir= r"C:\Users\dlxlr\Desktop\Dataset\Train_2\test\test_40_\D_mask",
        transform_img=transform,
        transform_mask=transform_mask
    )

    eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # 모델 로드
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",  # 백본으로 사용한 모델
        encoder_weights=None,
        in_channels=3,
        classes=num_classes
    )
    checkpoint_path = r"out\resner101_40%_F_\best_checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    # 평가 실행
    results, class_results = evaluate(model, eval_loader, device, num_classes)

    # 전체 결과 출력
    print("\nOverall Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # 클래스별 결과 출력
    print("\nClass-wise Results:")
    class_names = {0: "Background", 1: "Building", 2: "Road"}
    for cls, metrics in class_results.items():
        print(f"\nClass {cls} ({class_names[cls]}):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()


# train_2 R101-40%, test :40%_ 사용, 평가 시간: 19분 12초 