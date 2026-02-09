#!/bin/bash

CSV_DIR="/home/woongjae/ADD_Dataset_SpeechOnly/Dataset/preprocess/vad_csv"
OUTPUT_DIR="/home/woongjae/ADD_Dataset_SpeechOnly/Dataset/preprocess/vad_csv_postprocess"
NUM_WORKERS=8

echo "======================================"
echo " CSV dir     : $CSV_DIR"
echo " Output dir  : $OUTPUT_DIR"
echo " Num workers : $NUM_WORKERS"
echo "======================================"

python postprocess_vad.py \
  --csv_dir "$CSV_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_workers "$NUM_WORKERS"
