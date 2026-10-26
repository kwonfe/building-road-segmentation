import os
import random
import numpy as np
import torch

# =============================
# 유틸 함수
# =============================
def set_seed(seed=1):
    """랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def squeeze_tensor(x):
    """마스크 Tensor 차원 축소"""
    return x.squeeze()


def get_unique_result_dir(base_path='run', base_name='resner101_40%_F_'):
    """중복되지 않는 결과 디렉토리 생성"""
    i = 0
    while True:
        dir_name = os.path.join(base_path, base_name if i == 0 else f"{base_name}{i}")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            return dir_name
        i += 1


# =============================
# utils.py는 학습과 평가 과정에서 공통으로 쓰이는 
# 시드 고정, 텐서 처리, 디렉토리 관리 함수를 제공
# =============================
