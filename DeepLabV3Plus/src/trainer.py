import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from metrics import calculate_metrics

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=100, device='cuda', num_classes=3, save_dir="results"):

    model.to(device)
    scaler = torch.amp.GradScaler()

    history = {k: [] for k in [
        'epoch', 'train_loss', 'val_loss',
        'val_mIoU', 'val_pix_acc', 'precision', 'recall',
        'mAP50', 'mAP50-95', 'epoch_time'
    ]}

    best_val_loss = float("inf")
    best_model_path = os.path.join(save_dir, "best_checkpoint.pth")

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        # ===== Training =====
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ===== Validation =====
        model.eval()
        val_loss_total, val_mIoU_total, val_acc_total = 0, 0, 0
        val_precision_total, val_recall_total = 0, 0
        val_map50_total, val_map50_95_total = 0, 0
        count_batches = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                val_loss_total += loss.item()
                mIoU, pix_acc, precision, recall, map50, map50_95 = calculate_metrics(outputs, masks, num_classes)

                val_mIoU_total += mIoU
                val_acc_total += pix_acc
                val_precision_total += precision
                val_recall_total += recall
                val_map50_total += map50
                val_map50_95_total += map50_95
                count_batches += 1

        val_loss = val_loss_total / count_batches
        avg_mIoU = val_mIoU_total / count_batches
        avg_pix_acc = val_acc_total / count_batches
        avg_precision = val_precision_total / count_batches
        avg_recall = val_recall_total / count_batches
        avg_map50 = val_map50_total / count_batches
        avg_map50_95 = val_map50_95_total / count_batches
        epoch_duration = time.time() - start_time

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"mIoU: {avg_mIoU:.4f} | Pixel Acc: {avg_pix_acc:.4f} | "
              f"Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} | "
              f"mAP50: {avg_map50:.4f} | mAP50-95: {avg_map50_95:.4f} | "
              f"Time: {epoch_duration:.2f}s")

        # 로그 저장
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mIoU'].append(avg_mIoU)
        history['val_pix_acc'].append(avg_pix_acc)
        history['precision'].append(avg_precision)
        history['recall'].append(avg_recall)
        history['mAP50'].append(avg_map50)
        history['mAP50-95'].append(avg_map50_95)
        history['epoch_time'].append(epoch_duration)

        # 🎯 매 에포크마다 모델 저장 (epoch1_checkpoint.pth, epoch2_checkpoint.pth, ...)
        epoch_checkpoint_path = os.path.join(save_dir, f"epoch{epoch+1}_checkpoint.pth")
        torch.save(model.state_dict(), epoch_checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated at epoch {epoch+1} with val_loss={val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(save_dir, "last_checkpoint.pth"))
        pd.DataFrame(history).to_csv(os.path.join(save_dir, 'training_log.csv'), index=False)

    return history
