import SimpleITK as sitk
import torch
import numpy as np

file = "/data/BraTS20_Degraded_4x_5/BraTS20_Training_003/BraTS20_Training_003_t1ce.nii"

data = sitk.GetArrayFromImage(sitk.ReadImage(file)).astype(float)
data = torch.from_numpy(data).float()

print(f"Shape: {data.shape}")          # 기대값: (39, 240, 240)
print(f"Min: {data.min():.4f}")
print(f"Max: {data.max():.4f}")
print(f"NaN count: {torch.isnan(data).sum()}")
print(f"Zero ratio: {(data == 0).float().mean():.4f}")

# normalization 후 확인
normed = (data - data.min()) / (data.max() - data.min())
print(f"Normalized min: {normed.min():.4f}, max: {normed.max():.4f}")