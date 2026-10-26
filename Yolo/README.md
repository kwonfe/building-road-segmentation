# 코드 역할 설명

이 문서는 프로그램 소개가 아니라, 각 파일이 어떤 순서로 무엇을 하는지 코드 단위로 적는다.

실행 순서는 다음과 같다.

1. `src/label.py` — 세그멘테이션 마스크 이미지를 YOLO 폴리곤 라벨(`.txt`)로 변환한다.
2. `src/dataset.yaml` — 학습에 쓸 이미지·라벨 경로와 클래스 이름을 지정한다.
3. `src/yolov11.py` — 그 설정으로 YOLOv11 세그멘테이션 모델을 학습한다.
4. `src/testset.yaml` — 평가에 쓸 별도 데이터 경로를 지정한다.
5. `src/model_YOLO_eval.py` — 학습으로 나온 `best.pt`를 그 테스트 데이터로 평가하고 지표를 출력한다.
6. `run/_yolov11m_seg_2M/args.yaml` — 과거에 실제로 돌아간 학습의 인자 기록이다. 지금 `yolov11.py`에 적힌 값과 다를 수 있다.

---

## `src/label.py`

마스크 이미지(TIF 등)에서 특정 색 픽셀을 찾아, YOLO segmentation이 읽는 폴리곤 텍스트로 저장한다. 함수로 나뉜 부분은 컨투어를 폴리곤으로 만드는 처리이고, 파일 아래쪽 루프가 이미지마다 그 처리를 반복한다.

### 경로와 상수

| 이름 | 하는 일 |
| --- | --- |
| `input_dir` | 마스크 이미지가 있는 폴더. 현재 값은 `r"/masks_512_no"`. |
| `output_dir` | 변환된 `.txt`가 저장되는 폴더. `Data/A`. 없으면 만든다. |
| `no_label_dir` | 건물·도로 색이 하나도 없는 이미지를 복사해 두는 폴더. `paper/Reviewer1_3/NO`. 없으면 만든다. |
| `target_colors` | 찾을 색과 클래스 번호. `#0A0A0A`는 클래스 0(건물), `#1E1E1E`는 클래스 1(도로). |
| `PIXEL_THRESHOLD` | 그 색 픽셀 수가 이 값 이하면 그 색은 버린다. 현재 `0`이라, 픽셀이 1개라도 있으면 통과한다. |
| `AREA_THRESHOLD` | 외곽 컨투어 면적이 이 값 이하면 버린다. hole에는 적용하지 않는다. 현재 `0`이라 면적 필터는 꺼진 상태이고, 점 1~2개짜리 컨투어도 아래 `expand_degenerate`로 살린다. |

### `hex_to_bgr(hex_color)`

`#RRGGBB` 문자열을 OpenCV가 쓰는 BGR 튜플로 바꾼다. `#` 다음 두 글자씩 R, G, B로 읽고 순서를 B, G, R로 뒤집는다. `cv2.inRange`에 넘기려고 쓴다.

### `expand_degenerate(pts)`

YOLO 폴리곤은 점이 최소 3개여야 한다. `cv2.findContours`가 점 1개 또는 2개만 돌려주는 경우를 작은 다각형으로 늘린다.

- 점이 이미 3개 이상이면 그대로 반환한다.
- 점이 1개면 그 점을 왼쪽 위로 하는 1×1 사각형 4점을 만든다.
- 점이 2개인데 좌표가 같으면 같은 방식으로 1×1 사각형을 만든다.
- 점이 2개이고 서로 다르면, 두 점을 잇는 방향에 수직인 1픽셀 오프셋을 더해 삼각형을 만든다. 수직 성분이 둘 다 0이면 `(1, 0)`을 쓴다.

### `merge_outer_with_holes(outer_cnt, hole_cnts)`

건물이나 도로 안에 빈 영역(hole)이 있을 때, 외곽선과 hole을 폴리곤 하나로 잇는다. YOLO 라벨 한 줄은 폴리곤 하나라서, hole을 별도 객체로 두지 않고 외곽에 붙인다.

`cv2.RETR_CCOMP`에서 최상위 외곽은 반시계, 자식 hole은 시계 방향이다. 각 hole마다 외곽의 모든 점과 hole의 모든 점 사이 거리를 계산하고, 가장 가까운 점 쌍을 다리로 삼아 점 순서를 이렇게 이어 붙인다.

`외곽(시작 ~ 접점) → hole(접점 ~ 끝) → hole(시작 ~ 접점) → 외곽(접점 ~ 끝)`

이렇게 만든 폴리곤은 스스로 접히는(pinched) 형태가 되고, `fillPoly`의 even-odd 규칙에서는 hole 안이 비게 된다. 점이 3개 미만인 hole은 건너뛴다.

### 이미지 루프

`input_dir`의 파일 이름을 하나씩 처리한다.

1. 이름이 `._`로 시작하면 건너뛴다. macOS가 만드는 부가 파일을 제외하기 위한 조건이다.
2. `cv2.imread`로 읽는다. 실패하면 경로를 출력하고 다음 파일로 넘어간다.
3. 이미지 높이 `H`, 너비 `W`를 저장한다. 나중에 좌표를 0~1로 나눌 때 쓴다.
4. `target_colors`의 색마다 다음을 한다.
   - `hex_to_bgr`로 색을 바꾸고 `cv2.inRange`로 그 색만 하얀 이진 마스크를 만든다.
   - 픽셀 수가 0이거나 `PIXEL_THRESHOLD` 이하면 그 색은 건너뛴다.
   - 픽셀이 있으면 `any_color_detected`를 참으로 둔다.
   - `cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)`로 외곽과 hole을 한 번에 구한다. `CHAIN_APPROX_SIMPLE`은 직선 구간의 중간 점을 버려 점 개수를 줄인다. `hierarchy[i]`는 `[next, prev, first_child, parent]`이다.
   - `parent`(네 번째 값)가 `-1`이 아닌 컨투어는 hole이므로 이 단계에서는 건너뛴다. hole은 부모 외곽을 처리할 때 같이 붙인다.
   - `AREA_THRESHOLD > 0`일 때만, 외곽 면적이 그 값 이하인 컨투어를 버린다. 현재 값이 0이라 이 분기는 타지 않는다.
   - `first_child`에서 시작해 `next` 형제를 따라가며 자식 hole 목록을 모은다.
   - 외곽 점을 `expand_degenerate`에 넣는다.
   - hole이 있으면 `merge_outer_with_holes`, 없으면 외곽 점만 사용한다.
   - 각 점의 x는 `W`로, y는 `H`로 나눠 0~1 좌표로 만들고 `(클래스 번호, 좌표 리스트)`를 `polygons_all`에 넣는다.
   - 그 색에서 만든 외곽 개수, hole 개수, 픽셀 수를 출력한다.
5. 어떤 대상 색도 없으면 원본 이미지를 `no_label_dir`로 복사하고 `.txt`는 만들지 않는다.
6. 폴리곤이 하나라도 있으면 확장자를 뺀 파일명으로 `.txt`를 쓴다. 한 줄이 폴리곤 하나다.
   - 첫 숫자는 클래스 번호(`0` 또는 `1`).
   - 그 뒤로 정규화된 `x y x y ...`가 이어진다.
   - 줄의 마지막 좌표 뒤에는 공백 대신 줄바꿈을 넣는다.

만들어지는 한 줄의 형태는 `dataset/train/labels/*.txt`에 이미 있는 라벨과 같다.

```
<클래스> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

---

## `src/dataset.yaml`

`yolov11.py`의 `model.train(data=...)`가 읽는 학습용 데이터 설정이다.

| 키 | 하는 일 |
| --- | --- |
| `path` | 주석 처리된 절대 경로 자리. 현재는 비어 있고, 아래 `train`/`val`은 프로젝트 기준 상대 경로로 적혀 있다. |
| `train` | 학습 이미지가 있는 폴더. `dataset/train`. Ultralytics는 같은 이름 구조에서 `labels` 폴더의 `.txt`를 라벨로 찾는다. |
| `val` | 학습 중 검증에 쓰는 폴더. `dataset/val`. |
| `nc` | 클래스 개수. `2`. |
| `names` | 클래스 이름. 인덱스 0이 `building`, 1이 `road`. `label.py`의 클래스 번호와 같은 순서다. |

---

## `src/yolov11.py`

사전학습된 YOLOv11 medium segmentation 가중치로 학습을 시작하는 스크립트다. `python src/yolov11.py`로 실행하면 `main()`이 호출된다.

### `main()`

1. `YOLO("yolo11m-seg.pt")`  
   Ultralytics의 YOLOv11m 세그멘테이션 가중치를 불러온다. 파일이 없으면 라이브러리가 해당 이름으로 받아 온다. detection이 아니라 픽셀 영역(마스크)을 예측하는 `-seg` 가중치다.

2. `model.train(...)`에 넘기는 값:

   | 인자 | 값 | 하는 일 |
   | --- | --- | --- |
   | `data` | `"src/dataset.yaml"` | 학습/검증 경로와 클래스 정의. |
   | `epochs` | `50` | 전체 데이터를 50번 반복한다. |
   | `imgsz` | `512` | 입력 이미지를 512×512로 맞춘다. 주석대로 기본값 640이 아니라 DeepLabv3+ 실험과 같은 크기이다. |
   | `batch` | `16` | 한 스텝에 이미지 16장. |
   | `workers` | 주석 | `workers=0`은 꺼져 있다. 데이터 로더 프로세스 수는 Ultralytics 기본값을 쓴다. |
   | `lr0` | `1e-4` | 초기 학습률. |
   | `optimizer` | `"AdamW"` | AdamW 옵티마이저. |
   | `weight_decay` | `1e-4` | 가중치 감쇠. |
   | `device` | `0` | 0번 GPU. |
   | `name` | `"_yolov11m-seg-2M"` | 결과 폴더 이름. Ultralytics 기본 저장 위치는 `runs/segment/<name>`. |

학습이 끝나면 그 폴더의 `weights/best.pt`(검증 기준 최고)와 `weights/last.pt`(마지막 epoch)가 생긴다. 평가는 `best.pt`를 읽는다.

---

## `src/testset.yaml`

`model_YOLO_eval.py`의 `model.val(data=...)`가 읽는 평가용 설정이다. 키 의미는 `dataset.yaml`과 같고, 가리키는 폴더만 다르다.

| 키 | 값 | 하는 일 |
| --- | --- | --- |
| `train` | `dataset/Test_set/train` | Ultralytics `val` 모드도 이 키를 요구해서 적혀 있다. 지표 계산에 쓰는 쪽은 `val`이다. |
| `val` | `dataset/Test_set/val` | 평가 이미지 폴더. |
| `nc` | `2` | 클래스 수. |
| `names` | `building`, `road` | 학습 때와 같은 클래스 순서. |

이 파일은 `src/` 안에 있지만, 평가 스크립트는 `data="testset.yaml"`로 연다. 실행 위치(현재 작업 디렉터리)에 `testset.yaml`이 있어야 그 문자열이 맞다.

---

## `src/model_YOLO_eval.py`

학습된 가중치로 테스트 세트를 돌리고, segmentation 지표를 출력한다. `python src/model_YOLO_eval.py`로 실행하면 `run()`이 호출된다.

### `run()`

1. `YOLO("run/_yolov11m_seg_2M/weights/best.pt")`  
   해당 실험 폴더의 최고 가중치를 불러온다. 이 경로는 `yolov11.py`의 `name`(`runs/segment/_yolov11m-seg-2M`)과 다를 수 있으므로, 평가 전에 실제 `best.pt` 위치와 맞는지 확인해야 한다.

2. `model.val(...)`에 넘기는 값:

   | 인자 | 값 | 하는 일 |
   | --- | --- | --- |
   | `data` | `"testset.yaml"` | 평가 데이터 설정. |
   | `save_json` | `True` | 예측 결과를 JSON으로 저장한다. |
   | `save_txt` | `True` | 예측 폴리곤을 이미지별 `.txt`로 저장한다. |
   | `plots` | `True` | 혼동 행렬, PR 곡선 등 평가 그림을 저장한다. |
   | `name` | `"y11-m-40%_50%_test_10_20_test2"` | 이번 평가 결과 폴더 이름. |

3. 반환된 `metrics.seg`에서 아래 값을 출력한다. 전부 segmentation 태스크 기준이다.

   | 속성 | 출력 문구가 의미하는 것 |
   | --- | --- |
   | `all_ap` | 클래스별·IoU 임계값별 AP 전체. |
   | `ap` | 클래스별 AP. |
   | `ap50` | IoU 0.50에서의 클래스별 AP. |
   | `map` | 클래스 평균 AP (mAP, IoU 0.50~0.95). |
   | `map50` | IoU 0.50에서의 mAP. |
   | `map75` | IoU 0.75에서의 mAP. |
   | `mp` | 클래스 평균 precision. |
   | `mr` | 클래스 평균 recall. |
   | `p` | 클래스별 precision. |
   | `r` | 클래스별 recall. |
   | `f1` | 클래스별 F1. |

---

## `run/_yolov11m_seg_2M/args.yaml`

한 번 학습이 끝났을 때 Ultralytics가 그 실행에 쓰인 인자를 덤프한 파일이다. 코드를 실행하는 스크립트가 아니다. 같은 폴더의 `results.csv`는 epoch마다 기록된 손실·지표이다.

지금 `src/yolov11.py`에 직접 적힌 값과 이 파일의 값이 다른 항목이 있다. 이 파일은 “그때 그 실행”의 기록이다.

| 키 | 이 파일의 값 | 대응하는 코드 |
| --- | --- | --- |
| `task` | `segment` | 세그멘테이션 학습이었음을 표시. |
| `mode` | `train` | 학습 모드. |
| `model` | `yolo11m-seg.pt` | `yolov11.py`의 `YOLO(...)`와 동일. |
| `data` | `dataset10K.yaml` | 현재 스크립트는 `src/dataset.yaml`을 넘긴다. |
| `epochs` | `50` | `yolov11.py`와 동일. |
| `batch` | `16` | 동일. |
| `imgsz` | `512` | 동일. |
| `device` | `'0'` | 동일. |
| `workers` | `8` | 스크립트에서는 `workers`를 지정하지 않아 당시 기본값 8이 기록된 상태. |
| `name` | `_yolov11m_seg_2M_20%_NewVal` | 현재 스크립트의 `name`은 `_yolov11m-seg-2M`. |
| `optimizer` | `AdamW` | 동일. |
| `lr0` | `0.0001` | `1e-4`와 동일. |
| `weight_decay` | `0.0001` | `1e-4`와 동일. |
| `save_dir` | `runs\segment\_yolov11m_seg_2M_20%_NewVal` | 그 실행의 실제 출력 폴더. |

그 외 키(`patience`, `mosaic`, `fliplr`, `box`, `cls`, `dfl`, `hsv_*` 등)는 `yolov11.py`가 넘기지 않아서 Ultralytics 기본값으로 채워진 항목이다. 예를 들어 `box`/`cls`/`dfl`은 박스·분류·분포 손실 가중치, `mosaic`/`fliplr`/`hsv_*`는 학습 중 이미지 증강, `patience`는 조기 종료까지 기다릴 epoch 수, `overlap_mask`와 `mask_ratio`는 세그멘테이션 마스크를 학습할 때의 처리 설정이다.
