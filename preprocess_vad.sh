#!/bin/bash

GPU="MIG-56c6e426-3d07-52cb-aa59-73892edacb69"

DATASET_ROOT="/home/woongjae/ADD_Dataset_SpeechOnly/Dataset/0_large-corpus_toys"
PROTOCOL_FILE="/home/woongjae/ADD_Dataset_SpeechOnly/protocols/20260125_014841_telephony_protocol.txt"
OUTPUT_ROOT="/home/woongjae/ADD_Dataset_SpeechOnly/Dataset/preprocess"
NUM_WORKERS=8

echo "======================================"
echo " Dataset root : $DATASET_ROOT"
echo " Protocol     : $PROTOCOL_FILE"
echo " Output root  : $OUTPUT_ROOT"
echo " Num workers  : $NUM_WORKERS"
echo "======================================"

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU python preprocess_vad.py \
  --dataset_root "$DATASET_ROOT" \
  --protocol "$PROTOCOL_FILE" \
  --output_root "$OUTPUT_ROOT" \
  --num_workers "$NUM_WORKERS"
