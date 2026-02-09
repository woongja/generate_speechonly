import argparse
import csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

from tqdm import tqdm


# =====================
# Config
# =====================
THRESHOLD = 0.5        # speech_prob >= threshold → 음성
PAD_START = 0.1        # 세그먼트 시작 확장 (초)
PAD_END = 0.15         # 세그먼트 끝 확장 (초)
MERGE_GAP = 0.3        # 이보다 작은 gap은 병합 (초)
MIN_SPEECH_REGION = 0.7  # 이보다 짧은 세그먼트 제거 (초)
MIN_TOTAL_SPEECH = 2.0   # 파일 내 총 음성 시간 최소값 (초)


def load_vad_csv(csv_path):
    """
    CSV 로드 → [(start_time, end_time, speech_prob), ...]
    """
    frames = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append((
                float(row["start_time"]),
                float(row["end_time"]),
                float(row["speech_prob"])
            ))
    return frames


def frames_to_segments(frames, threshold=THRESHOLD):
    """
    프레임별 확률 → 연속된 음성 세그먼트 추출
    반환: [(start, end), ...]
    """
    segments = []
    in_speech = False
    seg_start = None

    for start, end, prob in frames:
        if prob >= threshold:
            if not in_speech:
                seg_start = start
                in_speech = True
        else:
            if in_speech:
                segments.append((seg_start, end))
                in_speech = False

    # 마지막 세그먼트 처리
    if in_speech:
        segments.append((seg_start, frames[-1][1]))

    return segments


def expand_segments(segments, pad_start=PAD_START, pad_end=PAD_END):
    """
    세그먼트 확장 (시작: -pad_start, 끝: +pad_end)
    """
    expanded = []
    for start, end in segments:
        new_start = max(0, start - pad_start)
        new_end = end + pad_end
        expanded.append((new_start, new_end))
    return expanded


def merge_segments(segments, merge_gap=MERGE_GAP):
    """
    gap이 merge_gap보다 작으면 병합
    """
    if not segments:
        return []

    merged = [segments[0]]

    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]

        if start - prev_end < merge_gap:
            # 병합
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged


def filter_short_segments(segments, min_duration=MIN_SPEECH_REGION):
    """
    min_duration보다 짧은 세그먼트 제거
    """
    return [(s, e) for s, e in segments if (e - s) >= min_duration]


def total_speech_duration(segments):
    """
    총 음성 시간 계산
    """
    return sum(e - s for s, e in segments)


def save_postprocess_csv(csv_path, segments, csv_dir, output_dir):
    """
    postprocess 결과를 별도 폴더에 저장
    폴더 구조는 유지
    """
    csv_path = Path(csv_path)
    rel_path = csv_path.relative_to(csv_dir)
    output_path = output_dir / rel_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_idx", "start", "end"])

        for idx, (start, end) in enumerate(segments):
            writer.writerow([idx, f"{start:.3f}", f"{end:.3f}"])


def process_single_csv(csv_path, csv_dir, output_dir, threshold, pad_start, pad_end, merge_gap, min_speech_region, min_total_speech):
    """
    단일 CSV 처리 + 결과 저장
    반환: (csv_path, status)
    status: 'valid', 'insufficient_speech', 'no_speech', 'error:...'
    """
    csv_path = Path(csv_path)

    try:
        # 1. CSV 로드
        frames = load_vad_csv(csv_path)

        # 2. threshold 적용 → 세그먼트 추출
        segments = frames_to_segments(frames, threshold)

        if not segments:
            return (str(csv_path), "no_speech")

        # 3. 세그먼트 확장
        segments = expand_segments(segments, pad_start, pad_end)

        # 4. 세그먼트 병합
        segments = merge_segments(segments, merge_gap)

        # 5. 짧은 세그먼트 제거
        segments = filter_short_segments(segments, min_speech_region)

        if not segments:
            return (str(csv_path), "no_speech")

        # 6. 총 음성 시간 체크
        total_duration = total_speech_duration(segments)

        if total_duration < min_total_speech:
            return (str(csv_path), "insufficient_speech")

        # 7. 결과 저장
        save_postprocess_csv(csv_path, segments, csv_dir, output_dir)

        return (str(csv_path), "valid")

    except Exception as e:
        return (str(csv_path), f"error: {e}")


def process_all_csvs(csv_dir, output_dir, num_workers, threshold, pad_start, pad_end, merge_gap, min_speech_region, min_total_speech):
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)

    # 모든 CSV 파일 수집
    csv_files = list(csv_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    print(f"Using {num_workers} workers")

    worker_fn = partial(
        process_single_csv,
        csv_dir=csv_dir,
        output_dir=output_dir,
        threshold=threshold,
        pad_start=pad_start,
        pad_end=pad_end,
        merge_gap=merge_gap,
        min_speech_region=min_speech_region,
        min_total_speech=min_total_speech
    )

    valid = insufficient = no_speech = errors = 0

    # 로그 파일 경로
    log_path = output_dir / "postprocess_skipped.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_file:
        with Pool(processes=num_workers) as pool:
            for csv_path, status in tqdm(
                pool.imap_unordered(worker_fn, csv_files, chunksize=100),
                total=len(csv_files),
                desc="Post-processing VAD"
            ):
                if status == "valid":
                    valid += 1
                elif status == "insufficient_speech":
                    insufficient += 1
                    rel_path = Path(csv_path).relative_to(csv_dir)
                    log_file.write(f"[INSUFFICIENT] {rel_path}\n")
                elif status == "no_speech":
                    no_speech += 1
                    rel_path = Path(csv_path).relative_to(csv_dir)
                    log_file.write(f"[NO_SPEECH] {rel_path}\n")
                else:
                    errors += 1
                    rel_path = Path(csv_path).relative_to(csv_dir)
                    log_file.write(f"[ERROR] {rel_path} | {status}\n")

    print("========== SUMMARY ==========")
    print(f"Valid        : {valid}")
    print(f"Insufficient : {insufficient}")
    print(f"No speech    : {no_speech}")
    print(f"Errors       : {errors}")
    print(f"Log file     : {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="VAD CSV 파일들이 있는 디렉토리")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="postprocess 결과 저장 디렉토리")
    parser.add_argument("--num_workers", type=int, default=cpu_count())
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--pad_start", type=float, default=PAD_START)
    parser.add_argument("--pad_end", type=float, default=PAD_END)
    parser.add_argument("--merge_gap", type=float, default=MERGE_GAP)
    parser.add_argument("--min_speech_region", type=float, default=MIN_SPEECH_REGION)
    parser.add_argument("--min_total_speech", type=float, default=MIN_TOTAL_SPEECH)

    args = parser.parse_args()

    process_all_csvs(
        csv_dir=args.csv_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        threshold=args.threshold,
        pad_start=args.pad_start,
        pad_end=args.pad_end,
        merge_gap=args.merge_gap,
        min_speech_region=args.min_speech_region,
        min_total_speech=args.min_total_speech
    )
