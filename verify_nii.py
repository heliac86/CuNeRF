#!/usr/bin/env python
"""
verify_nii.py — CuNeRF 출력 nii.gz 검증 스크립트
사용법:
  python verify_nii.py \
    --gt   /data/BraTS20_Training_003/BraTS20_Training_003_t1ce.nii.gz \
    --pred /dshome/.../BraTS20_Training_003_t1ce.nii.gz \
    --name CuNeRF \
    [--other /path/to/other_model.nii.gz --other_name ModelB]
"""

import argparse, os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
def load_sitk(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    return img, arr


def normalize_arr(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return arr
    return (arr - mn) / (mx - mn)


def safe_psnr(gt, pred, data_range=1.0):
    if gt.max() - gt.min() < 1e-8:
        return float("nan")
    return psnr(gt, pred, data_range=data_range)


def safe_ssim(gt, pred, data_range=1.0):
    return ssim(gt, pred, data_range=data_range)


def match_shape(arr, target_shape):
    """배열을 target_shape으로 zoom 리샘플링. shape이 이미 같으면 그대로 반환."""
    if arr.shape == target_shape:
        return arr
    factors = tuple(t / s for t, s in zip(target_shape, arr.shape))
    print(f"  [shape 리샘플링] {arr.shape} → {target_shape}  (zoom factors: {[f'{f:.4f}' for f in factors]})")
    return zoom(arr, factors, order=1).astype(np.float32)


# ─────────────────────────────────────────────
# 1. 메타데이터 비교
# ─────────────────────────────────────────────
def check_metadata(gt_img, pred_img, pred_name):
    print("\n" + "=" * 60)
    print("  1. 메타데이터 비교")
    print("=" * 60)

    def fmt(img, label):
        arr = sitk.GetArrayFromImage(img)
        print(f"\n  [{label}]")
        print(f"    shape     : {arr.shape}")
        print(f"    dtype     : {arr.dtype}")
        print(f"    spacing   : {img.GetSpacing()}")
        print(f"    origin    : {img.GetOrigin()}")
        print(f"    direction : {np.round(img.GetDirection(), 4)}")
        print(f"    intensity : min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.6f}")

    fmt(gt_img, "GT (원본 BraTS)")
    fmt(pred_img, f"PRED ({pred_name})")

    sp_diff = np.abs(np.array(gt_img.GetSpacing())   - np.array(pred_img.GetSpacing()))
    or_diff = np.abs(np.array(gt_img.GetOrigin())    - np.array(pred_img.GetOrigin()))
    di_diff = np.abs(np.array(gt_img.GetDirection()) - np.array(pred_img.GetDirection()))

    print(f"\n  [공간 좌표 차이]")
    print(f"    Δspacing   max = {sp_diff.max():.6f}  {'✓' if sp_diff.max() < 1e-4 else '⚠ 불일치!'}")
    print(f"    Δorigin    max = {or_diff.max():.4f}  {'✓' if or_diff.max() < 0.01 else '⚠ 불일치!'}")
    print(f"    Δdirection max = {di_diff.max():.6f}  {'✓' if di_diff.max() < 1e-4 else '⚠ 불일치!'}")

    gt_shape   = sitk.GetArrayFromImage(gt_img).shape
    pred_shape = sitk.GetArrayFromImage(pred_img).shape
    if gt_shape == pred_shape:
        print(f"\n    shape 일치 ✓  {gt_shape}")
    else:
        print(f"\n    ⚠ shape 불일치! GT={gt_shape}, PRED={pred_shape}")
        print(f"    → 지표 계산 시 자동으로 GT shape에 맞게 리샘플링합니다.")


# ─────────────────────────────────────────────
# 2. 축 방향 (nibabel axcodes)
# ─────────────────────────────────────────────
def check_orientation(gt_path, pred_path, pred_name):
    print("\n" + "=" * 60)
    print("  2. 축 방향 (nibabel axcodes)")
    print("=" * 60)

    for label, path in [("GT", gt_path), (pred_name, pred_path)]:
        nib_img = nib.load(path)
        codes = nib.aff2axcodes(nib_img.affine)
        print(f"  [{label}]  axcodes = {codes}")
        print(f"    affine diag = {np.diag(nib_img.affine).round(4)}")

    codes_gt   = nib.aff2axcodes(nib.load(gt_path).affine)
    codes_pred = nib.aff2axcodes(nib.load(pred_path).affine)
    if codes_gt == codes_pred:
        print("  → 축 방향 일치 ✓")
    else:
        print(f"  → ⚠ 축 방향 불일치! GT={codes_gt}, PRED={codes_pred}")


# ─────────────────────────────────────────────
# 2-B. --other 파일 메타데이터 요약
# ─────────────────────────────────────────────
def check_other_metadata(gt_arr, others, other_names):
    if not others:
        return
    print("\n" + "=" * 60)
    print("  2-B. 비교 모델 메타데이터 요약")
    print("=" * 60)
    gt_shape = gt_arr.shape
    for arr, name in zip(others, other_names):
        match = "✓" if arr.shape == gt_shape else f"⚠ GT={gt_shape}"
        print(f"  [{name}]  shape={arr.shape}  {match}")
        print(f"    intensity: min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.6f}")


# ─────────────────────────────────────────────
# 3. 뇌 알맹이 무게중심
# ─────────────────────────────────────────────
def check_brain_center(gt_arr, pred_arr, pred_name, threshold=0.05):
    print("\n" + "=" * 60)
    print("  3. 뇌 알맹이 무게중심 비교")
    print("=" * 60)

    def centroid(arr, thr):
        mask = arr > thr
        if mask.sum() == 0:
            return None, 0.0
        z, y, x = np.where(mask)
        return np.array([z.mean(), y.mean(), x.mean()]), mask.mean()

    gt_n   = normalize_arr(gt_arr)
    pred_n = normalize_arr(match_shape(pred_arr, gt_arr.shape))

    gt_c,   gt_ratio   = centroid(gt_n,   threshold)
    pred_c, pred_ratio = centroid(pred_n, threshold)

    print(f"  GT         무게중심 (z,y,x) = {np.round(gt_c, 2)}  nonzero ratio={gt_ratio:.4f}")
    print(f"  {pred_name:<10} 무게중심 (z,y,x) = {np.round(pred_c, 2)}  nonzero ratio={pred_ratio:.4f}")

    diff = np.abs(gt_c - pred_c)
    print(f"  Δ (voxel) = {np.round(diff, 2)}  max={diff.max():.2f}")
    if diff.max() < 5:
        print("  → 뇌 위치 정상 ✓")
    elif diff.max() < 20:
        print("  → △ 뇌 위치 차이 있음, 시각 확인 권장")
    else:
        print("  → ⚠ 뇌 위치 차이 큼! 축 반전 가능성 확인 필요")

    return gt_n, pred_n


# ─────────────────────────────────────────────
# 4. 화질 지표
# ─────────────────────────────────────────────
def compute_metrics(gt_arr, pred_arr, pred_name):
    print("\n" + "=" * 60)
    print(f"  4. 화질 지표 — {pred_name} vs GT")
    print("=" * 60)

    # shape 불일치 시 리샘플링
    pred_arr_r = match_shape(pred_arr, gt_arr.shape)

    gt_n   = normalize_arr(gt_arr)
    pred_n = normalize_arr(pred_arr_r)

    # 전체 볼륨 PSNR
    psnr_vol = safe_psnr(gt_n, pred_n)

    # 슬라이스별 SSIM 평균
    ssim_vals = [safe_ssim(gt_n[i], pred_n[i]) for i in range(len(gt_n))]
    ssim_mean = np.nanmean(ssim_vals)

    print(f"  전체 볼륨 PSNR      = {psnr_vol:.4f} dB")
    print(f"  슬라이스별 SSIM 평균 = {ssim_mean:.4f}")

    # 슬라이스별 PSNR
    slice_psnrs = np.array([safe_psnr(gt_n[i], pred_n[i]) for i in range(len(gt_n))])
    valid = slice_psnrs[~np.isnan(slice_psnrs)]
    print(f"  슬라이스별 PSNR: mean={valid.mean():.2f}  min={valid.min():.2f}  max={valid.max():.2f} dB")

    return slice_psnrs, gt_n, pred_n


# ─────────────────────────────────────────────
# 5. 시각화
# ─────────────────────────────────────────────
FIXED_SLICES = [10, 40, 77, 114, 144]


def visualize(gt_n, preds_n, names, slice_psnrs_list, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # (A) 고정 슬라이스 비교 (GT / pred / |diff|)
    n_cols = 1 + len(preds_n) * 2
    for sl in FIXED_SLICES:
        sl = min(sl, len(gt_n) - 1)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        fig.suptitle(f"Slice {sl}", fontsize=12)

        axes[0].imshow(gt_n[sl], cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("GT (원본)")
        axes[0].axis("off")

        for m, (pred_n, name) in enumerate(zip(preds_n, names)):
            pred_sl = pred_n[min(sl, len(pred_n) - 1)]
            diff_sl = np.abs(gt_n[sl] - pred_sl)
            col = 1 + m * 2

            axes[col].imshow(pred_sl, cmap="gray", vmin=0, vmax=1)
            axes[col].set_title(name)
            axes[col].axis("off")

            im = axes[col + 1].imshow(diff_sl, cmap="hot", vmin=0, vmax=0.3)
            axes[col + 1].set_title(f"|Δ| {name}")
            axes[col + 1].axis("off")
            plt.colorbar(im, ax=axes[col + 1], fraction=0.046)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"slice_{sl:03d}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # (B) 슬라이스별 PSNR 곡선
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, sp in zip(names, slice_psnrs_list):
        ax.plot(sp, label=name, alpha=0.8)
    ax.set_xlabel("Slice index")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Slice-wise PSNR vs GT")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "psnr_curve.png"), dpi=150)
    plt.close()

    # (C) 3축 단면 (axial / coronal / sagittal)
    mid_z = len(gt_n) // 2
    mid_y = gt_n.shape[1] // 2
    mid_x = gt_n.shape[2] // 2

    views = [
        ("axial",    lambda v, z=mid_z: v[z, :, :]),
        ("coronal",  lambda v, y=mid_y: v[:, y, :]),
        ("sagittal", lambda v, x=mid_x: v[:, :, x]),
    ]
    all_vols = [("GT", gt_n)] + list(zip(names, preds_n))

    for view_name, slicer in views:
        fig, axes = plt.subplots(1, len(all_vols), figsize=(4 * len(all_vols), 4))
        if len(all_vols) == 1:
            axes = [axes]
        fig.suptitle(f"{view_name.capitalize()} view (mid-slice)", fontsize=12)
        for ax_i, (label, vol) in enumerate(all_vols):
            sl = slicer(vol)
            axes[ax_i].imshow(sl, cmap="gray", vmin=0, vmax=1, aspect="auto")
            axes[ax_i].set_title(label)
            axes[ax_i].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"view_{view_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # (D) 슬라이스별 비-zero 픽셀 비율 비교
    fig, ax = plt.subplots(figsize=(12, 3))
    thr = 0.05
    ax.plot([((gt_n[i] > thr).mean()) for i in range(len(gt_n))],
            label="GT", color="black", alpha=0.7)
    for name, pred_n in zip(names, preds_n):
        ax.plot([((pred_n[i] > thr).mean()) for i in range(len(pred_n))],
                label=name, alpha=0.8)
    ax.set_xlabel("Slice index")
    ax.set_ylabel("Non-zero ratio")
    ax.set_title(f"Slice-wise non-zero pixel ratio (threshold={thr})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nonzero_curve.png"), dpi=150)
    plt.close()

    print(f"\n  시각화 저장 완료 → {out_dir}/")
    print(f"    slice_010/040/077/114/144.png   : 고정 슬라이스 비교")
    print(f"    psnr_curve.png                  : 슬라이스별 PSNR")
    print(f"    nonzero_curve.png               : 슬라이스별 비-zero 비율")
    print(f"    view_axial/coronal/sagittal.png : 3축 단면")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CuNeRF 출력 nii.gz 검증 스크립트"
    )
    parser.add_argument("--gt",         required=True,  help="원본 BraTS 155슬라이스 nii.gz")
    parser.add_argument("--pred",       required=True,  help="CuNeRF 출력 nii.gz")
    parser.add_argument("--name",       default="CuNeRF")
    parser.add_argument("--other",      nargs="+",      help="비교 모델 nii.gz (여러 개 가능)")
    parser.add_argument("--other_name", nargs="+",      help="비교 모델 이름 (--other 순서와 동일)")
    parser.add_argument("--out_dir",    default="verify_output")
    args = parser.parse_args()

    gt_img,   gt_arr   = load_sitk(args.gt)
    pred_img, pred_arr = load_sitk(args.pred)

    others, other_names = [], []
    if args.other:
        for path in args.other:
            _, arr = load_sitk(path)
            others.append(arr)
        other_names = args.other_name if args.other_name \
                      else [f"Model{i+1}" for i in range(len(others))]

    # 1~2: 메타데이터 / 축 방향 (--pred 기준)
    check_metadata(gt_img, pred_img, args.name)
    check_orientation(args.gt, args.pred, args.name)

    # 2-B: --other 파일 shape 요약
    check_other_metadata(gt_arr, others, other_names)

    # 3: 무게중심 (--pred 기준)
    gt_n, pred_n = check_brain_center(gt_arr, pred_arr, args.name)

    # 4: 지표 계산 (shape 불일치 시 자동 리샘플링)
    slice_psnrs, gt_n, pred_n = compute_metrics(gt_arr, pred_arr, args.name)
    psnrs_list = [slice_psnrs]
    names_list = [args.name]
    preds_n    = [pred_n]

    for arr, name in zip(others, other_names):
        sp, _, p_n = compute_metrics(gt_arr, arr, name)
        psnrs_list.append(sp)
        names_list.append(name)
        preds_n.append(normalize_arr(match_shape(arr, gt_arr.shape)))

    # 5: 시각화
    visualize(gt_n, preds_n, names_list, psnrs_list, args.out_dir)
    print("\n검증 완료.")


if __name__ == "__main__":
    main()
