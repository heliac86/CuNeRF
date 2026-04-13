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

    # z 범위를 dataset.py z_trans 공식으로 역산
    with open(args.cfg) as f:
        bcfg = yaml.safe_load(f)
    raw    = sitk.GetArrayFromImage(sitk.ReadImage(args.file))
    n_in   = raw.shape[0]
    radius = int(bcfg.get("radius", 1))
    pad    = int(max(radius, 1))
    denom  = n_in + 2 * pad - 1
    z_min  = 2 * np.pi * pad / denom - np.pi
    z_max  = 2 * np.pi * (n_in - 1 + pad) / denom - np.pi
    print(f"입력 슬라이스: {n_in}장  |  z 범위: [{z_min:.6f}, {z_max:.6f}]")

    # Cfg가 요구하는 args 속성 채우기
    args.mode       = "test"
    args.resume     = True          # ← 반드시 True여야 Resume()이 호출됨
    args.N_eval     = None
    args.save_map   = False
    args.max_iter   = None
    args.eval_iter  = None
    args.zpos       = [z_min, z_max]
    args.scales     = [1.0]
    args.angles     = [0]
    args.axis       = [0, 0, 1]
    args.asteps     = args.n_out
    args.cam_scale  = 1.0
    args.is_details = False
    args.is_gif     = False
    args.is_video   = False
    args.workers    = 0

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
            # run.py 원본 test()와 동일한 루프 구조
            for cidx in range(H * W // S + 1):
                select_inds  = list(range(S * cidx, min(S * (cidx + 1), len(coords))))
                if len(select_inds) == 0:
                    break
                select_flags = flags[select_inds]
                valid_inds   = torch.tensor(select_inds).long()[select_flags]
                if len(valid_inds) > 0:
                    rgb, _ = cfg.Render(coords[valid_inds], depths,
                                        is_train=False, R=R)
                    slice_pred[valid_inds.cpu().numpy()] = rgb.cpu().numpy()
            pds[idx] = slice_pred.reshape(H, W)

    # float32 nii.gz로 저장 + 원본 BraTS 공간 정보 복사
    ref     = sitk.ReadImage(args.ref_file)
    out_itk = sitk.GetImageFromArray(pds)
    out_itk.CopyInformation(ref)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sitk.WriteImage(out_itk, args.out)
    print(f"\n저장 완료: {args.out}")
    print(f"  shape={pds.shape}, min={pds.min():.4f}, max={pds.max():.4f}, mean={pds.mean():.6f}")
