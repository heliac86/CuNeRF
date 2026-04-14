#!/bin/bash

# ============================================================
# 설정 (여기만 수정)
# ============================================================
MODALITY="flair"        # "t1ce" 또는 "flair" 로 변경해서 사용

EXP_NAME="original"
CSV_FILE="/dshome/ddualab/dongnyeok/Cunerf_ori/test.csv"
DEGRADED_BASE="/data/BraTS20_Degraded_4x_5"
REF_BASE="/data/BraTS2020_TrainingData"
SAVE_BASE="/dshome/ddualab/dongnyeok/Cunerf_ori/save"
CFG="configs/004.yaml"
MAX_ITER=10000
EVAL_ITER=2000
N_EVAL=39
N_OUT=155

# ============================================================
# MODALITY에 따른 --modality 인자 자동 결정
# ============================================================
if [ "$MODALITY" = "t1ce" ]; then
    MODALITY_ARG="t1gd"
elif [ "$MODALITY" = "flair" ]; then
    MODALITY_ARG="FLAIR"
else
    echo "오류: MODALITY는 't1ce' 또는 'flair' 이어야 합니다."
    exit 1
fi

# ============================================================
# CSV에서 환자 ID 파싱 (헤더 제외, 첫 번째 컬럼)
# ============================================================
mapfile -t PATIENT_IDS < <(tail -n +2 "$CSV_FILE" | awk -F',' '{print $1}' | tr -d '\r')

echo "============================="
echo "모달리티 : ${MODALITY} (--modality ${MODALITY_ARG})"
echo "총 환자 수: ${#PATIENT_IDS[@]}"
echo "Max iter : ${MAX_ITER}"
echo "============================="

# ============================================================
# 케이스 순차 실행
# ============================================================
FAIL_LIST=()

for PID in "${PATIENT_IDS[@]}"; do

    NII_FILENAME="${PID}_${MODALITY}.nii"
    INPUT_FILE="${DEGRADED_BASE}/${PID}/${NII_FILENAME}"
    REF_FILE="${REF_BASE}/${PID}/${NII_FILENAME}"
    OUT_FILE="${SAVE_BASE}/${EXP_NAME}/${PID}/${NII_FILENAME}.gz"
    CKPT_EXP_NAME="${EXP_NAME}/${PID}_${MODALITY}"

    echo ""
    echo "[$(date '+%H:%M:%S')] 시작: ${PID} / ${MODALITY}"

    # --- 학습 ---
    python run.py "$CKPT_EXP_NAME" \
        --cfg "$CFG" \
        --file "$INPUT_FILE" \
        --scale 1 \
        --mode train \
        --modality "$MODALITY_ARG" \
        --max_iter "$MAX_ITER" \
        --eval_iter "$EVAL_ITER" \
        --N_eval "$N_EVAL" \
        --save_map \
        --resume

    if [ $? -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] 학습 실패: ${PID}" >&2
        FAIL_LIST+=("TRAIN_FAIL: ${PID}_${MODALITY}")
        continue
    fi

    # --- 추론 ---
    python run_test_to_nii.py "$CKPT_EXP_NAME" \
        --cfg "$CFG" \
        --file "$INPUT_FILE" \
        --ref_file "$REF_FILE" \
        --modality "$MODALITY_ARG" \
        --out "$OUT_FILE" \
        --n_out "$N_OUT" \
        --resume_type current

    if [ $? -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] 추론 실패: ${PID}" >&2
        FAIL_LIST+=("INFER_FAIL: ${PID}_${MODALITY}")
        continue
    fi

    echo "[$(date '+%H:%M:%S')] 완료: ${PID} / ${MODALITY}"

done

# ============================================================
# 최종 요약
# ============================================================
echo ""
echo "============================="
echo "전체 완료 [$(date '+%H:%M:%S')]"
if [ ${#FAIL_LIST[@]} -eq 0 ]; then
    echo "실패 케이스 없음"
else
    echo "실패 케이스 목록:"
    for FAIL in "${FAIL_LIST[@]}"; do
        echo "  - $FAIL"
    done
fi
echo "============================="
