#!/usr/bin/env python
# run_test_to_nii.py
import argparse, math, os, sys
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm
from functools import reduce

def argParse():
    parser = argparse.ArgumentParser()
    parser.add_argument("expname")
    parser.add_argument("--cfg",          default="configs/004.yaml")
    parser.add_argument("--file",         required=True,  help="입력 손상 nii 파일")
    parser.add_argument("--ref_file",     required=True,  help="원본 BraTS 155슬라이스 nii.gz (공간 정보 복사용)")
    parser.add_argument("--modality",     default="t1gd")
    parser.add_argument("--out",          required=True,  help="출력 nii.gz 경로")
    parser.add_argument("--n_out",        type=int, default=155)
    parser.add_argument("--resume_type",  default="current")
    parser.add_argument("--bs",           type=int, default=4096)
    parser.add_argument("--scale",        type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    args = argParse()

    # z 범위를 dataset.py z_trans 공식으로 역산
    import yaml
    with open(args.cfg) as f:
        bcfg = yaml.safe_load(f)
    raw = sitk.GetArrayFromImage(sitk.ReadImage(args.file))
    n_in = raw.shape[0]  # 39
    radius = bcfg["dataset"]["train"]["radius"]
    pad = int(max(radius, 1))
    denom = n_in + 2 * pad - 1
    z_min = 2 * np.pi * pad / denom - np.pi
    z_max = 2 * np.pi * (n_in - 1 + pad) / denom - np.pi
    print(f"입력 슬라이스: {n_in}장  |  z 범위: [{z_min:.6f}, {z_max:.6f}]")

    # Cfg가 요구하는 args 속성 채우기
    args.mode        = "test"
    args.resume      = False
    args.N_eval      = None
    args.save_map    = False
    args.max_iter    = None
    args.eval_iter   = None
    args.zpos        = [z_min, z_max]
    args.scales      = [1.0]
    args.angles      = [0]
    args.axis        = [0, 0, 1]
    args.asteps      = args.n_out
    args.cam_scale   = 1.0
    args.is_details  = False
    args.is_gif      = False
    args.is_video    = False
    args.workers     = 0

    from src import Cfg, utils
    cfg = Cfg(args)

    W, H, S = cfg.testset.W, cfg.testset.H, args.bs
    pds = np.zeros((args.n_out, H, W), dtype=np.float32)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(cfg.testloader, desc="Rendering")):
            coords, depths, R, zpos, angle, scale = batch
            coords, R = torch.squeeze(coords), torch.squeeze(R)
            flags = utils.judge_range(coords, R)
            slice_pred = np.zeros(H * W, dtype=np.float32)
            for cidx in range(math.ceil(H * W / S)):
                inds   = list(range(S * cidx, min(S * (cidx + 1), H * W)))
                valid  = torch.tensor(inds).long()[flags[inds]]
                if len(valid) > 0:
                    rgb, _ = cfg.Render(coords[valid], depths, is_train=False, R=R)
                    slice_pred[valid.cpu().numpy()] = rgb.cpu().numpy()
            pds[idx] = slice_pred.reshape(H, W)

    # 원본 BraTS 공간 정보를 그대로 복사해서 저장
    ref     = sitk.ReadImage(args.ref_file)
    out_itk = sitk.GetImageFromArray(pds)
    out_itk.CopyInformation(ref)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sitk.WriteImage(out_itk, args.out)
    print(f"\n저장 완료: {args.out}")
    print(f"  shape={pds.shape}, min={pds.min():.4f}, max={pds.max():.4f}, mean={pds.mean():.6f}")
