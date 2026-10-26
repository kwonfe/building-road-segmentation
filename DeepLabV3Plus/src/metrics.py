import torch
import numpy as np
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall

# =============================
# Metric 함수
# =============================
def calculate_metrics(outputs, masks, num_classes):
    """모델 출력과 마스크를 비교해 다양한 지표 계산"""
    preds = torch.argmax(outputs, dim=1).cpu()
    masks = masks.cpu()

    # Pixel Accuracy
    pixel_acc = (preds == masks).sum().item() / np.prod(masks.shape)

    # IoU per class
    iou_list = []
    for cls in range(num_classes):
        intersection = torch.logical_and(preds == cls, masks == cls).sum().item()
        union = torch.logical_or(preds == cls, masks == cls).sum().item()
        iou_list.append(intersection / union if union > 0 else np.nan)
    mIoU = np.nanmean(iou_list)

    # Precision & Recall
    metric_precision = MulticlassPrecision(num_classes=num_classes, average='macro')
    metric_recall = MulticlassRecall(num_classes=num_classes, average='macro')
    precision = metric_precision(preds, masks).item()
    recall = metric_recall(preds, masks).item()

    # mAP50 / mAP50-95 proxy using IoU thresholds
    thresholds_50 = 0.5
    thresholds_50_95 = np.arange(0.5, 1.0, 0.05)
    map50 = np.nanmean([iou_val >= thresholds_50 for iou_val in iou_list])
    map50_95 = np.nanmean([np.nanmean([iou_val >= t for iou_val in iou_list]) for t in thresholds_50_95])

    return mIoU, pixel_acc, precision, recall, map50, map50_95


# =============================
# metrics.py는 mIoU, PixelAcc, Precision, Recall, 
# mAP50, mAP50-95 같은 평가 지표를 계산하는 함수 모음
# =============================
