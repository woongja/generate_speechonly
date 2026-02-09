# ADD Dataset SpeechOnly

A preprocessing pipeline that extracts speech-only segments from Audio Deepfake Detection datasets.

Uses Silero VAD to detect speech regions in audio and saves only those regions as separate files.

## Pipeline Overview

```
Raw Audio → [1. preprocess_vad] → [2. postprocess_vad] → [3. speechonly_sample] → Speech Segments
```

| Step | Script | Description |
|------|--------|-------------|
| 1 | `preprocess_vad.py` | Standardize audio (mono/16kHz) + extract per-frame speech probabilities via Silero VAD → CSV |
| 2 | `postprocess_vad.py` | Apply threshold/padding/merge/filter to VAD CSV → refined segment CSV |
| 3 | `speechonly_sample.py` | Cut audio based on segment CSV → save individual files + generate metadata |

## Requirements

```
torch
torchaudio
silero-vad
librosa
soundfile
tqdm
```

## Usage

### Step 1. VAD Preprocessing

git clone https://github.com/snakers4/silero-vad.git

Converts audio to mono/16kHz and runs Silero VAD to extract per-frame (32ms) speech probabilities into CSV files.

```bash
python preprocess_vad.py \
  --dataset_root /path/to/dataset \
  --protocol /path/to/protocol.txt \
  --output_root /path/to/output \
  --num_workers 8
```

| Argument | Description |
|----------|-------------|
| `--dataset_root` | Root directory of original audio files |
| `--protocol` | Path to protocol file (first column = relative audio path) |
| `--output_root` | Output directory (creates `wav/` and `vad_csv/` subdirectories) |
| `--num_workers` | Number of parallel workers (default: CPU count) |

**Output structure:**
```
output_root/
├── wav/          # Standardized audio files
├── vad_csv/      # Per-frame speech_prob CSV files
└── vad_error.log
```

### Step 2. VAD Postprocessing

Applies thresholding to per-frame probabilities and refines speech segments.

```bash
python postprocess_vad.py \
  --csv_dir /path/to/output/vad_csv \
  --output_dir /path/to/output/vad_csv_postprocess \
  --num_workers 8
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--csv_dir` | VAD CSV directory from Step 1 | (required) |
| `--output_dir` | Directory to save refined segment CSVs | (required) |
| `--num_workers` | Number of parallel workers | CPU count |
| `--threshold` | Speech probability threshold | 0.5 |
| `--pad_start` | Expand segment start (seconds) | 0.1 |
| `--pad_end` | Expand segment end (seconds) | 0.15 |
| `--merge_gap` | Merge segments closer than this gap (seconds) | 0.3 |
| `--min_speech_region` | Minimum segment duration (seconds) | 0.7 |
| `--min_total_speech` | Minimum total speech duration per file (seconds) | 2.0 |

### Step 3. Speech-Only Segment Extraction

Cuts audio into individual speech segments based on refined CSVs and generates metadata.

```bash
python speechonly_sample.py \
  --csv_dir /path/to/output/vad_csv_postprocess \
  --wav_dir /path/to/output/wav \
  --output_dir /path/to/final_output \
  --protocol /path/to/protocol.txt \
  --num_workers 8
```

| Argument | Description |
|----------|-------------|
| `--csv_dir` | Postprocessed CSV directory from Step 2 |
| `--wav_dir` | Standardized audio directory from Step 1 |
| `--output_dir` | Directory to save final segment audio files |
| `--protocol` | Path to protocol file (for subset and label mapping) |
| `--num_workers` | Number of parallel workers (default: CPU count) |

**Output filename format:** `{original_id}_r{region}_s{segment}_t{start_ms}-{end_ms}_len{length_ms}.wav`

**Metadata (`metadata.csv`) columns:**
`current_filename`, `original_filename`, `subset`, `label`, `region_index`, `segment_index`, `orig_start_sec`, `orig_end_sec`, `segment_length_sec`, `sample_rate`, `vad_model`, `vad_threshold`, `pad_start_sec`, `pad_end_sec`, `merge_gap_sec`, `min_speech_region_sec`

## Shell Scripts

Example shell scripts are provided for each step.

```bash
# Step 1
bash preprocess_vad.sh

# Step 2
bash postprocess_vad.sh

# Step 3
bash speechonly_sample.sh
```

Update the paths in each `.sh` file to match your environment before running.
