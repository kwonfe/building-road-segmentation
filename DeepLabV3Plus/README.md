# DeepLabV3+ 건물·도로 분할

항공영상에서 건물과 도로를 픽셀 단위로 나누는 코드입니다. 이 폴더에는 전처리, 학습, 학습 중 검증, 학습 후 테스트에 필요한 파일만 있습니다. 영상과 마스크 파일은 들어 있지 않고, `data`와 `dataset`은 원래 폴더 구조만 비어 있습니다.

학습에 넣는 정답은 배경 `0`, 건물 `1`, 도로 `2`인 단일 채널 마스크 `D_mask`입니다.

## 폴더 구조

```
DeepLabV3Plus/
├── configs/config.yaml          # 데이터 경로, 백본, 학습 설정
├── src/
│   ├── extract_mask_by_class.py # 전처리: 코드 마스크 → D_mask
│   ├── dataset.py               # 영상과 D_mask를 한 쌍으로 읽음
│   ├── train.py                 # 학습 시작
│   ├── trainer.py               # 학습 루프와 에포크마다 하는 검증
│   ├── metrics.py               # 검증 수치 계산
│   ├── evaluate_model.py        # 학습이 끝난 뒤 테스트
│   └── utils.py                 # 시드, 마스크 차원, 결과 폴더                      
├── dataset/                     # 같은 용도의 빈 폴더 구조
└── out/                         # 학습 결과가 저장되는 곳
```

## 실행 순서

1. `extract_mask_by_class.py`로 `D_mask`를 만듭니다.
2. `configs/config.yaml`에 영상 폴더와 `D_mask` 폴더 경로를 적습니다.
3. 이 폴더에서 아래처럼 학습합니다. 작업 폴더는 루트여야 `configs/config.yaml`을 찾고, `src`가 모듈 경로에 있어야 `dataset`을 찾습니다.

```powershell
$env:PYTHONPATH = "src"
python src/train.py
```

4. 학습이 끝나면 `out` 안에 가중치와 `training_log.csv`가 생깁니다.
5. 테스트는 `src/evaluate_model.py`입니다. 파일 안의 테스트 데이터 경로와 체크포인트 경로를 그 PC에 맞게 바꾼 뒤 실행합니다.

```powershell
$env:PYTHONPATH = "src"
python src/evaluate_model.py
```

`segmentation_models_pytorch`를 설치해야 `DeepLabV3Plus`와 그 안의 ASPP를 쓸 수 있습니다. ASPP 코드는 이 폴더에 없고, 라이브러리를 설치하면 가상환경 안에 생깁니다.

---

## 전처리

### `src/extract_mask_by_class.py`

수집한 마스크의 픽셀 값은 토지피복 코드입니다. `10`이 건물, `30`이 도로이고, 그 밖에 `20`, `50`, `100` 같은 코드가 함께 있습니다.

이 스크립트는 건물과 도로만 남깁니다. 건물은 `1`, 도로는 `2`, 나머지 코드는 `0`으로 둔 256×256 흑백 TIFF를 저장합니다. 저장 폴더 이름이 `D_mask`입니다.

이 파일이 필요한 이유: 학습 손실 `CrossEntropyLoss`는 픽셀마다 `0`, `1`, `2` 같은 클래스 번호를 요구합니다. 원본 코드 `10`, `30`, `100`을 그대로 넣으면 클래스 번호로 해석되지 않습니다.

설명할 때 볼 부분:

- `color_to_class`가 클래스 정의입니다. `#0A0A0A`는 픽셀 값 10(건물), `#1E1E1E`는 픽셀 값 30(도로)입니다.
- 마스크를 먼저 0으로 채운 뒤 건물과 도로 위치만 덮어씁니다. 목록에 없는 코드는 배경으로 남습니다.
- 크기 변경은 `Image.NEAREST`입니다. 평균을 내면 클래스 번호가 1과 2 사이의 값으로 섞입니다.

---

## 학습에 필요한 코드

### `configs/config.yaml`

데이터 경로와 학습 설정을 코드 밖에 둔 파일입니다. 장수나 백본을 바꿀 때 이 파일만 수정합니다.

| 항목 | 현재 값 | 필요한 이유 |
| --- | --- | --- |
| `train_images`, `train_masks` | 학습 영상, `D_mask` | 모델이 볼 입력과 정답 위치 |
| `val_images`, `val_masks` | 검증 영상, `D_mask` | 학습 중에 성능을 재는 데이터. 가중치 갱신에는 쓰지 않음 |
| `encoder_name` | `resnet101` | DeepLabV3+ 앞단 백본 |
| `encoder_weights` | `imagenet` | 백본만 ImageNet 가중치로 시작. ASPP와 디코더는 여기서 학습 |
| `num_classes` | `3` | 배경, 건물, 도로 |
| `batch_size` | `16` | 한 번에 넣는 영상 수 |
| `num_epochs` | `50` | 전체 데이터를 반복하는 횟수 |
| `learning_rate` | `0.0001` | AdamW 학습률 |
| `save_base_dir` | `out` | 체크포인트와 로그가 저장되는 폴더 |

`patience`는 이 파일에 있지만 학습 코드가 읽지 않습니다. 50 에포크를 모두 진행합니다.

### `src/dataset.py`

영상 TIFF와 같은 이름의 `D_mask`를 한 쌍으로 읽어 `DataLoader`에 넘깁니다.

필요한 이유: 학습 루프는 폴더를 직접 열지 않습니다. 배치마다 `(영상 텐서, 마스크 텐서)`가 필요합니다. 마스크는 RGB로 바꾸지 않고, 마지막에 `.long()`으로 바꿉니다. `CrossEntropyLoss`가 정수 클래스 번호를 받기 때문입니다.

### `src/train.py`

학습을 시작하는 파일입니다. 설정을 읽고, 데이터를 올리고, 모델을 만든 뒤 `trainer.py`에 넘깁니다.

필요한 이유: 전처리된 파일을 모델이 받을 수 있는 배치로 바꾸고, 어떤 네트워크를 어떤 손실로 학습할지 한곳에서 정합니다.

설명할 때 볼 부분:

- `set_seed(1)`은 난수 시작점입니다. 디코더 초기 가중치와 데이터를 섞는 순서가 실행할 때마다 같아집니다. 백본 가중치는 시드가 아니라 ImageNet에서 불러옵니다.
- 영상은 512×512로 맞춘 뒤 ImageNet 평균 `(0.485, 0.456, 0.406)`, 표준편차 `(0.229, 0.224, 0.225)`로 정규화합니다. `encoder_weights: imagenet`과 짝입니다.
- 마스크는 최근접 보간으로만 512×512가 됩니다. 정규화하지 않습니다.
- `smp.DeepLabV3Plus`는 라이브러리에 있는 DeepLabV3+입니다. 인코더, ASPP, 디코더가 이 호출 안에서 조립됩니다. 출력 채널은 `num_classes`인 3입니다.
- 손실은 `CrossEntropyLoss`, 옵티마이저는 `AdamW`(학습률 `0.0001`, weight decay `1e-4`)입니다.

### `src/trainer.py`

에포크를 실제로 반복하는 파일입니다. 한 에포크는 학습, 검증, 저장으로 이루어집니다.

필요한 이유: 가중치를 갱신하는 반복과, 갱신에 쓰지 않는 검증을 나눠 두어야 학습이 진행되는지 숫자로 확인할 수 있습니다.

설명할 때 볼 부분:

- `model.train()` 구간에서만 가중치가 갱신됩니다. `torch.amp.autocast`와 `GradScaler`로 혼합정밀도 학습을 합니다. 512 입력과 배치 16을 돌기 위한 설정입니다.
- `model.eval()`과 `torch.no_grad()` 구간이 검증입니다. 검증 데이터로 손실과 지표를 계산하고, 그 손실로 역전파하지 않습니다.
- 에포크가 끝날 때마다 `epochN_checkpoint.pth`와 `last_checkpoint.pth`를 저장합니다.
- 검증 손실이 지금까지 중 가장 낮으면 `best_checkpoint.pth`를 덮어씁니다. 선택 기준은 mIoU가 아니라 검증 손실입니다.
- 같은 에포크의 손실과 지표를 `training_log.csv`에 남깁니다.

### `src/metrics.py`

검증 배치의 예측과 정답을 비교해 숫자를 만듭니다. `trainer.py`가 에포크마다 호출합니다.

필요한 이유: 손실만으로는 건물과 도로가 맞았는지 말하기 어렵습니다. 배경 픽셀 비율이 커서 픽셀 정확도만 보면 높게 나옵니다. 클래스별 IoU를 평균한 mIoU가 이 과제의 기준입니다.

계산하는 값:

- 예측 클래스는 채널 방향 `argmax`입니다.
- Pixel Accuracy: 맞춘 픽셀 비율
- mIoU: 배경, 건물, 도로 IoU의 평균
- Precision, Recall: 클래스 균등 평균(macro)
- mAP50, mAP50-95: IoU 임계값으로 만든 보조 지표

### `src/utils.py`

학습과 데이터 처리가 같이 쓰는 작은 함수입니다.

- `set_seed`: 실험 재현을 위해 난수 시작점을 고정합니다.
- `squeeze_tensor`: 마스크를 `(1, H, W)`에서 `(H, W)`로 줄입니다. 차원이 남으면 손실 함수가 클래스 축을 잘못 읽습니다.
- `get_unique_result_dir`: `out` 아래에 실험 폴더를 만듭니다. 같은 이름이 있으면 번호를 붙여 이전 결과를 덮어쓰지 않습니다.

---

## 검증과 테스트

둘 다 학습된 예측을 정답 마스크와 비교합니다. 쓰는 시점과 데이터가 다릅니다.

| | 검증 | 테스트 |
| --- | --- | --- |
| 코드 | `trainer.py` + `metrics.py` | `evaluate_model.py` |
| 시점 | 50 에포크 동안 매 에포크 끝 | 학습이 끝난 뒤 따로 실행 |
| 데이터 | `config.yaml`의 `val_images`, `val_masks` | 스크립트에 적힌 테스트 영상과 `D_mask` |
| 가중치 | 지금 학습 중인 모델 | 저장된 `best_checkpoint.pth` |
| 하는 일 | 검증 손실로 최적 가중치를 고르고 로그를 남김 | 그 가중치의 최종 성능을 클래스별로 출력 |

### `src/evaluate_model.py`

학습에 쓰지 않은 테스트 세트로 최종 성능을 출력하는 파일입니다.

필요한 이유: 학습 중 검증은 가중치를 고르려고 매 에포크 반복하는 점수입니다. 테스트를 따로 두어야, 고른 체크포인트가 학습 중에 보지 않은 영상에서 어떤 점수인지 말할 수 있습니다.

하는 일:

- 영상과 마스크 전처리는 학습과 같습니다. 512×512, 영상만 ImageNet 정규화, 마스크는 최근접 보간입니다.
- `DeepLabV3Plus`를 다시 만들고 `encoder_weights=None`으로 둡니다. ImageNet 가중치 대신 `best_checkpoint.pth`를 읽습니다. 백본 이름은 학습 때와 같은 `resnet101`이어야 가중치가 맞습니다.
- 기본 체크포인트 경로는 `out\resner101_40%_F_\best_checkpoint.pth`입니다.
- 전체 점수와 클래스별 점수를 출력합니다. 클래스는 `0` 배경, `1` 건물, `2` 도로입니다.
- 테스트에서 추가로 보는 값은 클래스별 F1, AP50, AP50-95입니다.

현재 스크립트 안의 테스트 경로는 이 PC의 절대 경로입니다. 다른 PC에서는 `image_dir`, `mask_dir`, `checkpoint_path`를 그 환경의 경로로 바꿉니다.

---

## 데이터 폴더

`dataset` 구조:

```
dataset/
├── train/   images, D_mask
└── val/     val/images, val_D_masks, 
```

학습 코드가 짝으로 읽는 것은 영상 폴더와 `D_mask`입니다. 파일 이름이 같아야 합니다.

## 결과 폴더 `out`

`config.yaml`의 `save_base_dir`가 `out`입니다. 학습을 시작하면 이 아래에 실험 폴더가 생기고, 그 안에 아래 파일이 쌓입니다.

- `best_checkpoint.pth`: 검증 손실이 가장 낮았던 가중치. 테스트가 이 파일을 읽습니다.
- `last_checkpoint.pth`: 마지막 에포크 가중치
- `epoch1_checkpoint.pth` … : 에포크마다 저장한 가중치
- `training_log.csv`: 에포크별 학습 손실, 검증 손실, mIoU, Pixel Accuracy, Precision, Recall, mAP50, mAP50-95
