import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import sys

sys.argv = ['run.py', 'test_0411',
    '--cfg', 'configs/003.yaml',
    '--mode', 'train',
    '--file', '/data/BraTS20_Degraded_4x_5/BraTS20_Training_003/BraTS20_Training_003_t1ce.nii',
    '--modality', 't1gd',
    '--resume_type', 'psnr']

from run import argParse
from src import Cfg
args = argParse()
cfg = Cfg(args)

print(f"train data shape: {cfg.trainset.data.shape}")
print(f"train data min={cfg.trainset.data.min():.6f}, max={cfg.trainset.data.max():.6f}")
print(f"train data 0인 픽셀 비율: {(cfg.trainset.data==0).float().mean():.4f}")

# 슬라이스 몇 개의 실제 값 분포 확인
for z in [0, 10, 19, 29, 38]:
    sl = cfg.trainset.data[z]
    nonzero = sl[sl > 0]
    print(f"  z={z:2d}: nonzero pixels={len(nonzero):5d}, mean={nonzero.mean():.4f} (전체 mean={sl.mean():.4f})")