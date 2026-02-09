import argparse
import csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

import soundfile as sf
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# VAD 설정 (메타데이터용)
VAD_MODEL = "silero-vad"
VAD_THRESHOLD = 0.5
PAD_START_SEC = 0.1
PAD_END_SEC = 0.15
MERGE_GAP_SEC = 0.3
MIN_SPEECH_REGION_SEC = 0.7


def load_protocol(protocol_path):
    """
    Protocol 파일 로드 → {relative_path: (subset, label)} 매핑
    bonafide → real, spoof → fake
    """
    protocol_map = {}
    with open(protocol_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            rel_path = parts[0]
            subset = parts[1]  # eval, train 등
            label = parts[2]  # bonafide or spoof

            # 확장자 제거한 stem으로 매핑 (wav → csv 매칭용)
            stem = Path(rel_path).stem
            parent = Path(rel_path).parent

            protocol_map[str(parent / stem)] = (subset, label)

    return protocol_map


def load_postprocess_csv(csv_path):
    """
    postprocess CSV 로드 → [(start, end), ...]
    """
    segments = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append((
                float(row["start"]),
                float(row["end"])
            ))
    return segments


def find_audio_file(csv_path, csv_dir, wav_dir):
    """
    postprocess CSV 경로에서 대응하는 오디오 파일 경로 찾기
    """
    rel_path = Path(csv_path).relative_to(csv_dir)
    stem = rel_path.stem
    parent = rel_path.parent

    audio_parent = wav_dir / parent

    # 같은 stem을 가진 파일 찾기
    for ext in [".wav", ".flac", ".mp3", ".opus", ".m4a"]:
        audio_path = audio_parent / (stem + ext)
        if audio_path.exists():
            return audio_path

    return None


def process_single_csv(csv_path, csv_dir, wav_dir, output_dir, protocol_map):
    """
    단일 CSV 처리: 오디오 로드 → 세그먼트별로 자르기 → 저장
    반환: (csv_path, status, metadata_list)
    metadata_list: [(current_filename, original_filename, region_idx, segment_idx, start, end, length, sr, label), ...]
    """
    csv_path = Path(csv_path)
    metadata_list = []

    try:
        # 1. 대응하는 오디오 파일 찾기
        audio_path = find_audio_file(csv_path, csv_dir, wav_dir)
        if audio_path is None:
            return (str(csv_path), "audio_not_found", [])

        # 2. CSV 로드
        segments = load_postprocess_csv(csv_path)
        if not segments:
            return (str(csv_path), "no_segments", [])

        # 3. 오디오 로드
        try:
            wav, sr = sf.read(audio_path)
        except Exception:
            wav, sr = librosa.load(audio_path, sr=None, mono=False)
            if wav.ndim == 2:
                wav = wav.T  # librosa는 (channels, samples) 반환

        # mono로 변환
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        # 4. 출력 디렉토리 구조 생성
        rel_path = csv_path.relative_to(csv_dir)
        output_parent = output_dir / rel_path.parent
        output_parent.mkdir(parents=True, exist_ok=True)

        # 5. 원본 파일 정보
        original_id = csv_path.stem
        audio_ext = audio_path.suffix  # .wav, .flac 등

        # 6. subset, label 찾기
        label_key = str(rel_path.parent / original_id)
        subset, label = protocol_map.get(label_key, ("unknown", "unknown"))

        # 7. 원본 파일명 (확장자 포함)
        original_filename = str(rel_path.parent / (original_id + audio_ext))

        # 8. 각 세그먼트별로 자르기 및 저장
        for region_idx, (start, end) in enumerate(segments):
            start_sample = int(start * sr)
            end_sample = int(end * sr)

            # 범위 체크
            end_sample = min(end_sample, len(wav))
            if start_sample >= len(wav):
                continue

            segment_wav = wav[start_sample:end_sample]

            # 파일명 생성
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            segment_len_ms = end_ms - start_ms
            segment_len_sec = round(end - start, 3)

            filename = f"{original_id}_r{region_idx}_s{region_idx}_t{start_ms}-{end_ms}_len{segment_len_ms}{audio_ext}"
            output_path = output_parent / filename

            # 저장
            if audio_ext == ".flac":
                sf.write(str(output_path), segment_wav, sr, format="FLAC")
            else:
                sf.write(str(output_path), segment_wav, sr)

            # 메타데이터 수집
            current_filename = str(rel_path.parent / filename)
            metadata_list.append((
                current_filename,
                original_filename,
                subset,
                label,
                region_idx,
                region_idx,  # segment_idx = region_idx
                round(start, 3),
                round(end, 3),
                segment_len_sec,
                sr,
                VAD_MODEL,
                VAD_THRESHOLD,
                PAD_START_SEC,
                PAD_END_SEC,
                MERGE_GAP_SEC,
                MIN_SPEECH_REGION_SEC
            ))

        return (str(csv_path), "success", metadata_list)

    except Exception as e:
        return (str(csv_path), f"error: {e}", [])


def process_all(csv_dir, wav_dir, output_dir, protocol_path, num_workers):
    csv_dir = Path(csv_dir)
    wav_dir = Path(wav_dir)
    output_dir = Path(output_dir)

    # Protocol 로드
    print("Loading protocol...")
    protocol_map = load_protocol(protocol_path)
    print(f"Loaded {len(protocol_map)} entries")

    # 모든 postprocess CSV 파일 수집
    csv_files = list(csv_dir.rglob("*.csv"))
    csv_files = [f for f in csv_files if f.suffix == ".csv" and not f.name.endswith(".log")]

    print(f"Found {len(csv_files)} CSV files")
    print(f"Using {num_workers} workers")

    worker_fn = partial(
        process_single_csv,
        csv_dir=csv_dir,
        wav_dir=wav_dir,
        output_dir=output_dir,
        protocol_map=protocol_map
    )

    success = audio_not_found = no_segments = errors = 0
    total_segments = 0
    all_metadata = []

    # 로그 파일
    log_path = output_dir / "extract_segments.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_file:
        with Pool(processes=num_workers) as pool:
            for csv_path, status, metadata_list in tqdm(
                pool.imap_unordered(worker_fn, csv_files, chunksize=100),
                total=len(csv_files),
                desc="Extracting segments"
            ):
                if status == "success":
                    success += 1
                    total_segments += len(metadata_list)
                    all_metadata.extend(metadata_list)
                elif status == "audio_not_found":
                    audio_not_found += 1
                    rel_path = Path(csv_path).relative_to(csv_dir)
                    log_file.write(f"[AUDIO_NOT_FOUND] {rel_path}\n")
                elif status == "no_segments":
                    no_segments += 1
                    rel_path = Path(csv_path).relative_to(csv_dir)
                    log_file.write(f"[NO_SEGMENTS] {rel_path}\n")
                else:
                    errors += 1
                    rel_path = Path(csv_path).relative_to(csv_dir)
                    log_file.write(f"[ERROR] {rel_path} | {status}\n")

    # 메타데이터 CSV 저장
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "current_filename",
            "original_filename",
            "subset",
            "label",
            "region_index",
            "segment_index",
            "orig_start_sec",
            "orig_end_sec",
            "segment_length_sec",
            "sample_rate",
            "vad_model",
            "vad_threshold",
            "pad_start_sec",
            "pad_end_sec",
            "merge_gap_sec",
            "min_speech_region_sec"
        ])
        for row in all_metadata:
            writer.writerow(row)

    print("========== SUMMARY ==========")
    print(f"Success         : {success}")
    print(f"Total segments  : {total_segments}")
    print(f"Audio not found : {audio_not_found}")
    print(f"No segments     : {no_segments}")
    print(f"Errors          : {errors}")
    print(f"Log file        : {log_path}")
    print(f"Metadata        : {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="postprocess CSV 파일들이 있는 디렉토리")
    parser.add_argument("--wav_dir", type=str, required=True,
                        help="원본 오디오 파일들이 있는 디렉토리")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="세그먼트 오디오 저장 디렉토리")
    parser.add_argument("--protocol", type=str, required=True,
                        help="Protocol 파일 경로")
    parser.add_argument("--num_workers", type=int, default=cpu_count())

    args = parser.parse_args()

    process_all(
        csv_dir=args.csv_dir,
        wav_dir=args.wav_dir,
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        num_workers=args.num_workers
    )
