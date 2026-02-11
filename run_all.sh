#!/bin/bash

##############################################
# 여러 파이프라인을 주석 처리해가면서 실행
# 각각 다른 screen에서 돌리면 됨
#
# 사용법:
#   1. 돌릴 블록만 주석 해제
#
# 옵션 설명:
#   -d    원본 데이터셋 경로 (DATASET_ROOT)
#   -p    프로토콜 파일 경로 (PROTOCOL_FILE)
#   -o    최종 출력 저장 경로 (OUTPUT_DIR)
#   -g    GPU 디바이스 ID
#   -w    워커 수 (default: 8)
#   -s    시작 스텝 (1=preprocess, 2=postprocess, 3=sample)
##############################################

# GPU 0: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (UUID: GPU-c07cf4f2-fe74-ec31-7035-f01beb777c12)
#   MIG 2g.48gb     Device  0: (UUID: MIG-8cdeef83-092c-5a8d-a748-452f299e1df0)
#   MIG 2g.48gb     Device  1: (UUID: MIG-6e4275af-2db0-51f1-a601-7ad8a1002745)
# GPU 1: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (UUID: GPU-cc2a8b10-275b-23c1-69ad-2562b1da62e1)
#   MIG 2g.48gb     Device  0: (UUID: MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494)
#   MIG 2g.48gb     Device  1: (UUID: MIG-57de94a5-be15-5b5a-b67e-e118352d8a59)
# GPU 2: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (UUID: GPU-32764ef0-1926-644a-ce12-3d843c305f1e)
#   MIG 2g.48gb     Device  0: (UUID: MIG-ad433dcf-e7b9-5a99-a0fa-6fdf3033b7cd)
#   MIG 2g.48gb     Device  1: (UUID: MIG-56c6e426-3d07-52cb-aa59-73892edacb69)
# GPU 3: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (UUID: GPU-589d86ff-f8c6-815f-a780-1afb008be925)
# GPU 4: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (UUID: GPU-a5880663-549a-0889-18a3-747f0ce3e006)
# GPU 5: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (UUID: GPU-01b2984c-0b2a-7db7-c16c-f2a50d009e5e)


PIPELINE="$(cd "$(dirname "$0")" && pwd)/run_pipeline.sh"

## ---- m_ailabs_v7_only_la_codec_aug ----
# bash "$PIPELINE" \
#   -d /nvme3/hungdx/DVC_DSD-Large-Corpus/pool/m_ailabs_v7_only_la_codec_aug \
#   -p /home/woongjae/ADD_Dataset_SpeechOnly/protocols/protocol_m_ailabs_v7_only_la_codec_aug.txt \
#   -o /nvme3/Datasets/m_ailabs_v7_only_la_codec_aug_pre_VAD \
#   -g "MIG-ad433dcf-e7b9-5a99-a0fa-6fdf3033b7cd" \
#   -w 8

## ---- m_ailabs_v8_only_la_codec_aug ----
# bash "$PIPELINE" \
#   -d /nvme3/hungdx/DVC_DSD-Large-Corpus/pool/m_ailabs_v8_only_la_codec_aug \
#   -p /home/woongjae/ADD_Dataset_SpeechOnly/protocols/protocol_m_ailabs_v8_only_la_codec_aug.txt \
#   -o /nvme3/Datasets/m_ailabs_v8_only_la_codec_aug_pre_VAD \
#   -g "MIG-ad433dcf-e7b9-5a99-a0fa-6fdf3033b7cd" \
#   -w 8

## ---- m_ailabs_v9_only_la_codec_aug ----
bash "$PIPELINE" \
  -d /nvme3/hungdx/DVC_DSD-Large-Corpus/pool/m_ailabs_v9_only_la_codec_aug \
  -p /home/woongjae/ADD_Dataset_SpeechOnly/protocols/protocol_m_ailabs_v9_only_la_codec_aug.txt \
  -o /nvme3/Datasets/m_ailabs_v9_only_la_codec_aug_pre_VAD \
  -g "MIG-56c6e426-3d07-52cb-aa59-73892edacb69" \
  -w 8
