import os
import yaml
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

from dataset import SegmentationDataset
from utils import set_seed, squeeze_tensor, get_unique_result_dir
from trainer import train_model

def main():
    set_seed(1)

    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    train_data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((512,512), interpolation=Image.NEAREST),
        transforms.PILToTensor(),
        transforms.Lambda(squeeze_tensor)
    ])

    train_dataset = SegmentationDataset(
        image_dir=train_data_cfg["train_images"],
        mask_dir=train_data_cfg["train_masks"],
        transform_img=transform,
        transform_mask=transform_mask
    )

    val_dataset = SegmentationDataset(
        image_dir=train_data_cfg["val_images"],
        mask_dir=train_data_cfg["val_masks"],
        transform_img=transform,
        transform_mask=transform_mask
    )

    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False,
                            num_workers=8, pin_memory=True)

    model = smp.DeepLabV3Plus(
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=model_cfg["in_channels"],
        classes=train_cfg["num_classes"],
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=1e-4)

    save_dir = get_unique_result_dir(base_path=train_cfg["save_base_dir"])
    print(f"Saving results to {save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=train_cfg["num_epochs"], device=device,
                num_classes=train_cfg["num_classes"], save_dir=save_dir)

if __name__ == '__main__':
    main()
