#!/usr/bin/env python
# run_test_to_nii.py
import argparse, math, os
import numpy as np
import SimpleITK as sitk
import torch
import yaml
from tqdm import tqdm

def argParse():
    parser = argparse.ArgumentParser()
    parser.add_argument("expname")
    parser.add_argument("--cfg",         default="configs/004.yaml")
    parser.add_argument("--file",        required=True)
    parser.add_argument("--ref_file",    required=True,
                        help="원본 BraTS 155슬라이스 nii.gz (공간 정보 복사용)")
    parser.add_argument("--modality",    default="t1gd")
    parser.add_argument("--out",         required=True)
    parser.add_argument("--n_out",       type=int, default=155)
    parser.add_argument("--resume_type", default="current")
    parser.add_argument("--bs",          type=int, default=4096)
    parser.add_argument("--scale",       type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    args = argParse()

    # Cfg가 요구하는 args 속성 채우기 — eval 모드 사용
    args.mode       = "eval"
    args.resume     = True
    args.N_eval     = args.n_out   # 155
    args.save_map   = False
    args.max_iter   = None
    args.eval_iter  = None
    args.zpos       = None
    args.scales     = None
    args.angles     = None
    args.axis       = None
    args.asteps     = None
    args.cam_scale  = None
    args.is_details = False
    args.is_gif     = False
    args.is_video   = False
    args.workers    = 0

    from src import Cfg, utils
    cfg = Cfg(args)

    # evalset.vals: N_eval=155개의 균등 샘플 z 인덱스 (0~38 범위에서 155개)
    # → 39슬라이스를 155개 위치로 보간하는 효과
    print(f"evalset.vals (처음 5개): {cfg.evalset.vals[:5]}")
    print(f"evalset.vals (마지막 5개): {cfg.evalset.vals[-5:]}")

    N = cfg.evalset.__len__()    # 155
    W = cfg.evalset.W            # 240
    H = cfg.evalset.H            # 240
    S = args.bs                  # 4096
    pds = np.zeros((N, H * W), dtype=np.float32)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(cfg.evalloader, desc="Rendering")):
            coords, depths = batch
            coords = coords.squeeze(0)
            for cidx in range(math.ceil(W * H / S)):
                select_coords = coords[list(range(S * cidx, min(S * (cidx + 1), len(coords))))]
                rgb, _ = cfg.Render(select_coords, depths, is_train=False)
                pds[idx, S * cidx : S * (cidx + 1)] = rgb.cpu().numpy()
        pds = pds.reshape(N, H, W)

    # float32 nii.gz 저장 + 원본 BraTS 공간 정보 복사
    ref     = sitk.ReadImage(args.ref_file)
    out_itk = sitk.GetImageFromArray(pds)
    out_itk.CopyInformation(ref)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sitk.WriteImage(out_itk, args.out)
    print(f"\n저장 완료: {args.out}")
    print(f"  shape={pds.shape}, min={pds.min():.4f}, max={pds.max():.4f}, mean={pds.mean():.6f}")
