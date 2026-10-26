from ultralytics import YOLO


def main():
    model = YOLO("yolo11m-seg.pt")

    model.train(
        data="src/dataset.yaml",
        epochs=50,
        imgsz=512,  # DeepLabv3+와 동일 (default=640 → 512로 변경 )
        batch=16,
        # workers=0,
        lr0=1e-4,
        optimizer="AdamW",
        weight_decay=1e-4,
        device=0,
        name="_yolov11m-seg-2M",
    )


if __name__ == "__main__":
    main()
