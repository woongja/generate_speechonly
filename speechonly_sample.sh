#!/bin/bash

CSV_DIR="/home/woongjae/ADD_Dataset_SpeechOnly/Dataset/preprocess/vad_csv_postprocess"
WAV_DIR="/home/woongjae/ADD_Dataset_SpeechOnly/Dataset/preprocess/wav"
OUTPUT_DIR="/nvme3/Datasets/speechonly/20260204"
PROTOCOL="/home/woongjae/ADD_Dataset_SpeechOnly/protocols/20260125_014841_telephony_protocol.txt"
NUM_WORKERS=8

echo "======================================"
echo " CSV dir     : $CSV_DIR"
echo " WAV dir     : $WAV_DIR"
echo " Output dir  : $OUTPUT_DIR"
echo " Protocol    : $PROTOCOL"
echo " Num workers : $NUM_WORKERS"
echo "======================================"

python speechonly_sample.py \
  --csv_dir "$CSV_DIR" \
  --wav_dir "$WAV_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --protocol "$PROTOCOL" \
  --num_workers "$NUM_WORKERS"
