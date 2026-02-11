#!/bin/bash
set -e

##############################################
# SpeechOnly Pipeline
# 1) preprocess_vad.py
# 2) postprocess_vad.py
# 3) speechonly_sample.py
##############################################

usage() {
  echo "Usage: $0 -d DATASET_ROOT -p PROTOCOL_FILE -o OUTPUT_DIR [options]"
  echo ""
  echo "Required:"
  echo "  -d    Dataset root path (원본 데이터셋 경로)"
  echo "  -p    Protocol file path (프로토콜 파일 경로)"
  echo "  -o    Final output directory (최종 저장 경로)"
  echo ""
  echo "Optional:"
  echo "  -g    GPU device (default: MIG-56c6e426-3d07-52cb-aa59-73892edacb69)"
  echo "  -w    Number of workers (default: 8)"
  echo "  -s    Start from step (1=preprocess, 2=postprocess, 3=sample, default: 1)"
  echo "  -h    Show this help"
  exit 1
}

# Defaults
GPU="MIG-56c6e426-3d07-52cb-aa59-73892edacb69"
NUM_WORKERS=8
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
START_STEP=1

while getopts "d:p:o:g:w:s:h" opt; do
  case $opt in
    d) DATASET_ROOT="$OPTARG" ;;
    p) PROTOCOL_FILE="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    g) GPU="$OPTARG" ;;
    w) NUM_WORKERS="$OPTARG" ;;
    s) START_STEP="$OPTARG" ;;
    h) usage ;;
    *) usage ;;
  esac
done

# Validate required arguments
if [ -z "$DATASET_ROOT" ] || [ -z "$PROTOCOL_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Error: -d, -p, -o are required."
  echo ""
  usage
fi

if [ ! -d "$DATASET_ROOT" ]; then
  echo "Error: Dataset root does not exist: $DATASET_ROOT"
  exit 1
fi

if [ ! -f "$PROTOCOL_FILE" ]; then
  echo "Error: Protocol file does not exist: $PROTOCOL_FILE"
  exit 1
fi

# Intermediate paths (OUTPUT_DIR 밑에 자동 생성)
PREPROCESS_DIR="${OUTPUT_DIR}/preprocess"
VAD_CSV_DIR="${PREPROCESS_DIR}/vad_csv"
VAD_CSV_POST_DIR="${PREPROCESS_DIR}/vad_csv_postprocess"
WAV_DIR="${PREPROCESS_DIR}/wav"

echo "======================================================"
echo " SpeechOnly Pipeline"
echo "======================================================"
echo " Dataset root   : $DATASET_ROOT"
echo " Protocol file  : $PROTOCOL_FILE"
echo " Final output   : $OUTPUT_DIR"
echo " Preprocess dir : $PREPROCESS_DIR"
echo " GPU            : $GPU"
echo " Num workers    : $NUM_WORKERS"
echo " Start step     : $START_STEP"
echo "======================================================"
echo ""

########################################
# Step 1: preprocess_vad
########################################
if [ "$START_STEP" -le 1 ]; then
  echo "======================================================"
  echo " [Step 1/3] preprocess_vad"
  echo "======================================================"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU python "${SCRIPT_DIR}/preprocess_vad.py" \
    --dataset_root "$DATASET_ROOT" \
    --protocol "$PROTOCOL_FILE" \
    --output_root "$PREPROCESS_DIR" \
    --num_workers "$NUM_WORKERS"
  echo "[Step 1/3] Done."
  echo ""
fi

########################################
# Step 2: postprocess_vad
########################################
if [ "$START_STEP" -le 2 ]; then
  echo "======================================================"
  echo " [Step 2/3] postprocess_vad"
  echo "======================================================"
  python "${SCRIPT_DIR}/postprocess_vad.py" \
    --csv_dir "$VAD_CSV_DIR" \
    --output_dir "$VAD_CSV_POST_DIR" \
    --num_workers "$NUM_WORKERS"
  echo "[Step 2/3] Done."
  echo ""
fi

########################################
# Step 3: speechonly_sample
########################################
if [ "$START_STEP" -le 3 ]; then
  echo "======================================================"
  echo " [Step 3/3] speechonly_sample"
  echo "======================================================"
  python "${SCRIPT_DIR}/speechonly_sample.py" \
    --csv_dir "$VAD_CSV_POST_DIR" \
    --wav_dir "$WAV_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --protocol "$PROTOCOL_FILE" \
    --num_workers "$NUM_WORKERS"
  echo "[Step 3/3] Done."
  echo ""
fi

echo "======================================================"
echo " Pipeline complete!"
echo " Results saved to: $OUTPUT_DIR"
echo "======================================================"
