import cv2
import os
import numpy as np
import shutil

# 입력 및 출력 디렉토리 경로 설정
input_dir = r"/masks_512_no"  # TIF 파일(Segmentaion Mask)이 저장된 경로
output_dir = "Data/A"  # 변환된 TXT 파일이 저장될 경로
no_label_dir = "paper/Reviewer1_3/NO"  # 클래스 0,1 모두 없는 이미지 저장 폴더

os.makedirs(output_dir, exist_ok=True)
os.makedirs(no_label_dir, exist_ok=True)

# 대상 색상과 해당 카테고리 ID 매핑
# '#0A0A0A'는 클래스 0(건물), '#1E1E1E'는 클래스 1(도로)
target_colors = [
    ("#0A0A0A", 0),
    ("#1E1E1E", 1),
]

PIXEL_THRESHOLD = 0  # 색상별 픽셀 수가 이 값 이하이면 제외
AREA_THRESHOLD = 0  # 외곽 컨투어 면적이 이 값 이하이면 제외 (hole 은 면적 제한 없음)
# 0 = 모든 컨투어 포함 (degenerate 1~2 점 컨투어는 1x1 폴리곤으로 확장)


def hex_to_bgr(hex_color):
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return (rgb[2], rgb[1], rgb[0])


def expand_degenerate(pts):
    """
    1~2 점 짜리 degenerate 컨투어를 1x1 (또는 작은) 폴리곤(>= 3 점)으로 확장.
    YOLO 폴리곤 형식은 최소 3점이 필요.
    """
    if len(pts) >= 3:
        return pts
    if len(pts) == 1:
        x, y = pts[0]
        return np.array(
            [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]], dtype=np.int64
        )
    # len == 2
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    if x0 == x1 and y0 == y1:
        return np.array(
            [[x0, y0], [x0 + 1, y0], [x0 + 1, y0 + 1], [x0, y0 + 1]],
            dtype=np.int64,
        )
    # 두 점이 서로 다른 경우: 두 점 + 살짝 오프셋 한 점으로 삼각형 구성
    dx, dy = x1 - x0, y1 - y0
    # perpendicular vector (절대값 1로 정규화 대신 1픽셀 오프셋)
    nx, ny = -dy, dx
    if nx == 0 and ny == 0:
        nx, ny = 1, 0
    # 정수로 클램프
    nx = int(np.sign(nx)) if nx != 0 else 0
    ny = int(np.sign(ny)) if ny != 0 else 0
    return np.array([[x0, y0], [x1, y1], [x0 + nx, y0 + ny]], dtype=np.int64)


def merge_outer_with_holes(outer_cnt, hole_cnts):
    """
    외곽 컨투어와 내부 hole 컨투어들을 'bridge' 기법으로 단일 폴리곤으로 병합.

    cv2.findContours 출력 방향:
      - 외곽(RETR_CCOMP top-level): counter-clockwise
      - 내부 hole(children): clockwise
    각 hole 의 한 점을 외곽 컨투어의 가장 가까운 점에 연결하여
    self-intersecting (pinched) polygon 을 만든다.
    fillPoly 의 기본 even-odd rule 하에서 hole 내부가 정확히 비워진다.
    """
    outer = outer_cnt.reshape(-1, 2).astype(np.int64)
    for hole_cnt in hole_cnts:
        hole = hole_cnt.reshape(-1, 2).astype(np.int64)
        if len(hole) < 3:
            continue
        # outer 와 hole 사이 모든 점 쌍 거리 -> 가장 가까운 점 쌍
        dists = np.linalg.norm(outer[:, None, :] - hole[None, :, :], axis=2)
        i_outer, i_hole = np.unravel_index(np.argmin(dists), dists.shape)
        outer = np.vstack(
            [
                outer[: i_outer + 1],
                hole[i_hole:],
                hole[: i_hole + 1],
                outer[i_outer:],
            ]
        )
    return outer


for j in os.listdir(input_dir):
    if j.startswith("._"):
        continue

    image_path = os.path.join(input_dir, j)
    image = cv2.imread(image_path)

    if image is None:
        print(f"이미지를 불러오는데 실패했습니다: {image_path}")
        continue

    H, W = image.shape[:2]
    polygons_all = []
    any_color_detected = False

    for hex_color, category_id in target_colors:
        bgr_color = hex_to_bgr(hex_color)
        mask = cv2.inRange(image, bgr_color, bgr_color)

        pixel_count = np.count_nonzero(mask)
        if pixel_count == 0:
            continue
        if pixel_count <= PIXEL_THRESHOLD:
            continue

        any_color_detected = True

        # RETR_CCOMP: 2-level (외곽 + hole). hierarchy[i] = [next, prev, first_child, parent]
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None or len(contours) == 0:
            continue
        hierarchy = hierarchy[0]

        outer_count = 0
        hole_count = 0
        for i, cnt in enumerate(contours):
            if hierarchy[i][3] != -1:
                continue  # hole 은 외곽 처리 단계에서 함께 다루므로 스킵

            # AREA_THRESHOLD = 0 인 경우: 모든 외곽 컨투어 포함 (degenerate 도 1x1 로 확장)
            if cv2.contourArea(cnt) <= AREA_THRESHOLD and AREA_THRESHOLD > 0:
                continue

            # 자식 hole 모으기 (next sibling 체인을 따라가며)
            holes = []
            child = hierarchy[i][2]
            while child != -1:
                holes.append(contours[child])
                child = hierarchy[child][0]

            outer_pts = cnt.reshape(-1, 2).astype(np.int64)
            outer_pts = expand_degenerate(outer_pts)

            if holes:
                merged = merge_outer_with_holes(outer_pts.reshape(-1, 1, 2), holes)
            else:
                merged = outer_pts

            polygon = []
            for x, y in merged:
                polygon.append(float(x) / W)
                polygon.append(float(y) / H)
            polygons_all.append((category_id, polygon))

            outer_count += 1
            hole_count += len(holes)

        print(
            f"[{hex_color}] 외곽 {outer_count}개 / hole {hole_count}개 (픽셀 {pixel_count})"
        )

    if not any_color_detected:
        shutil.copy(image_path, os.path.join(no_label_dir, j))
        print(f"→ 라벨 없음: {j} → {no_label_dir} 복사됨")
        continue

    if len(polygons_all) > 0:
        txt_file_path = "{}.txt".format(os.path.join(output_dir, j)[:-4])
        with open(txt_file_path, "w") as f:
            for category_id, polygon in polygons_all:
                for p_idx, p in enumerate(polygon):
                    if p_idx == len(polygon) - 1:
                        f.write(f"{p}\n")
                    elif p_idx == 0:
                        f.write(f"{category_id} {p} ")
                    else:
                        f.write(f"{p} ")
        print(f"파일이 생성되었습니다: {txt_file_path}")
