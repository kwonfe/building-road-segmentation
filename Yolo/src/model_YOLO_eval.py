from ultralytics import YOLO


def run():
    model = YOLO("run/_yolov11m_seg_2M/weights/best.pt")  # 모델 파일 경로

    metrics = model.val(
        data="testset.yaml",
        save_json=True,
        save_txt=True,
        plots=True,
        name="y11-m-40%_50%_test_10_20_test2",  # 실험 이름....
    )

    print("Average precision for all classes:", metrics.seg.all_ap)
    print("Average precision:", metrics.seg.ap)
    print("Average precision at IoU=0.50:", metrics.seg.ap50)
    print("Mean average precision:", metrics.seg.map)
    print("Mean average precision at IoU=0.50:", metrics.seg.map50)
    print("Mean average precision at IoU=0.75:", metrics.seg.map75)
    print("Mean precision:", metrics.seg.mp)
    print("Mean recall:", metrics.seg.mr)
    print("Precision:", metrics.seg.p)
    print("Recall:", metrics.seg.r)
    print("F1 score:", metrics.seg.f1)


if __name__ == "__main__":
    run()
