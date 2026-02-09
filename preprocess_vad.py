import argparse
import csv
from pathlib import Path
import shutil
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

import librosa
import soundfile as sf
import torch
import torchaudio
from silero_vad import load_silero_vad
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =====================
# Global config
# =====================
SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
FRAME_DURATION = FRAME_SAMPLES / SAMPLE_RATE  # 32ms

# Worker별 모델 (프로세스마다 초기화)
_model = None


def init_worker():
    """각 worker 프로세스에서 모델 초기화"""
    global _model
    _model = load_silero_vad(onnx=False)
    torch.set_num_threads(1)


# =====================
# 1. Audio standardization
# =====================
def standardize_audio(input_path, output_path=None):
    """
    Load audio, convert to mono / 16kHz for VAD.
    1차: soundfile로 로드 시도
    2차: 실패 시 librosa로 로드 + 원본 파일 복사
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else None

    use_fallback = False

    # 1차: soundfile로 시도
    try:
        wav_np, sr = sf.read(input_path)

        if wav_np.ndim == 2:
            wav_np = wav_np.mean(axis=1)

        wav = torch.from_numpy(wav_np).float()

        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    except Exception:
        # 2차: librosa로 시도
        wav_np, _ = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        wav = torch.from_numpy(wav_np).float()
        use_fallback = True

    # 저장
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if use_fallback:
            # librosa로 읽은 경우: 원본 파일 복사
            shutil.copy2(input_path, output_path)
        else:
            # soundfile로 읽은 경우: WAV로 저장
            try:
                sf.write(
                    str(output_path),
                    wav.numpy(),
                    samplerate=SAMPLE_RATE,
                    subtype="PCM_16"
                )
            except Exception:
                # WAV 저장 실패 시 원본 복사
                shutil.copy2(input_path, output_path)

    return wav


# =====================
# 2. Frame-level VAD → CSV
# =====================
def extract_vad_frames_to_csv(wav_tensor, csv_path):
    global _model
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        probs = _model.audio_forward(wav_tensor, sr=SAMPLE_RATE)

    probs = probs.squeeze().flatten()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "start_time", "end_time", "speech_prob"])

        for i, p in enumerate(probs):
            writer.writerow([
                i,
                f"{i * FRAME_DURATION:.3f}",
                f"{(i + 1) * FRAME_DURATION:.3f}",
                f"{p.item():.6f}"
            ])


# =====================
# 3. 단일 파일 처리 함수 (worker용)
# =====================
def process_single_file(args, dataset_root, output_root):
    """
    단일 파일 처리. 반환: (status, audio_path, error_msg)
    status: 'processed', 'skipped', 'missing', 'error'
    """
    rel_path = Path(args)
    audio_path = dataset_root / rel_path
    parent_dir = rel_path.parent
    stem = rel_path.stem
    filename = rel_path.name

    wav_out = output_root / "wav" / parent_dir / filename
    csv_out = output_root / "vad_csv" / parent_dir / f"{stem}.csv"

    try:
        wav = standardize_audio(audio_path, wav_out)
        extract_vad_frames_to_csv(wav, csv_out)
        return ("processed", str(audio_path), None)

    except Exception as e:
        return ("error", str(audio_path), str(e))


# =====================
# 4. Main processing
# =====================
def process_from_protocol(dataset_root, protocol_path, output_root, num_workers):
    dataset_root = Path(dataset_root)
    protocol_path = Path(protocol_path)
    output_root = Path(output_root)

    error_log_path = output_root / "vad_error.log"
    error_log_path.parent.mkdir(parents=True, exist_ok=True)

    # 프로토콜 읽기
    with open(protocol_path, "r") as f:
        lines = [line.strip().split()[0] for line in f if line.strip()]

    print(f"Total files: {len(lines)}")
    print(f"Using {num_workers} workers")

    # partial로 고정 인자 전달
    worker_fn = partial(
        process_single_file,
        dataset_root=dataset_root,
        output_root=output_root
    )

    processed = failed = 0

    with open(error_log_path, "w") as error_log:
        with Pool(processes=num_workers, initializer=init_worker) as pool:
            results = pool.imap_unordered(worker_fn, lines, chunksize=100)

            for status, audio_path, error_msg in tqdm(results, total=len(lines), desc="VAD preprocessing"):
                if status == "processed":
                    processed += 1
                elif status == "error":
                    failed += 1
                    error_log.write(f"[ERROR] {audio_path} | {error_msg}\n")

    print("========== SUMMARY ==========")
    print(f"Processed: {processed}")
    print(f"Failed   : {failed}")
    print(f"Error log: {error_log_path}")


# =====================
# Entry point
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--protocol", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=cpu_count(),
                        help="Number of parallel workers (default: CPU count)")

    args = parser.parse_args()

    process_from_protocol(
        dataset_root=args.dataset_root,
        protocol_path=args.protocol,
        output_root=args.output_root,
        num_workers=args.num_workers
    )
