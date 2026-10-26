import os
from PIL import Image
import numpy as np

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_class_mask(image_path, save_path, color_to_class, size=(256, 256)):
    """
    마스크 이미지 -> RGB 마스크 이미지를 변환
    클래스별 색상만 1,2로 표시하고 나머지는 0(배경)인 단일 채널 마스크 이미지로 저장
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size, resample=Image.NEAREST)
    img_np = np.array(img)

    mask = np.zeros((size[1], size[0]), dtype=np.uint8)  # (높이, 너비), 배경=0

    for color, cls in color_to_class.items():
        matches = np.all(img_np == color, axis=-1)
        mask[matches] = cls

    mask_img = Image.fromarray(mask, mode='L')  # 단일 채널 이미지 (0~255)
    mask_img.save(save_path)
    print(f"Saved class mask: {save_path}")

if __name__ == "__main__":
    color_to_class = {
        hex_to_rgb("#0A0A0A"): 1,  # 건물
        hex_to_rgb("#1E1E1E"): 2,  # 도로
    }

    input_mask_folder = r"C:\Users\dlxlr\Desktop\Dataset\Train_2\val\mask"  # 원본 컬러 마스크 폴더 경로
    output_mask_folder = r"C:\Users\dlxlr\Desktop\Dataset\Train_2\val\D_mask"  # 새로 저장할 단일 채널 마스크 경로

    os.makedirs(output_mask_folder, exist_ok=True)

    mask_files = sorted(os.listdir(input_mask_folder))

    for mask_file in mask_files:
        input_path = os.path.join(input_mask_folder, mask_file)
        base_name = os.path.splitext(mask_file)[0]
        output_path = os.path.join(output_mask_folder, base_name + ".tif")  # TIFF로 저장

        create_class_mask(input_path, output_path, color_to_class, size=(256,256))
