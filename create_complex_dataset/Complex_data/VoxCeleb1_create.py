#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VoxCeleb1 "complex" augmentation @16 kHz (single output + CSV + optional Excel)

• Per-speaker class-balanced split (≈25% each): babble / music / loudness / clean
• All audio (speech + noise) forced to 16 kHz mono
• SNR logging: 25 ms window, 10 ms hop (babble & music only)
• Loudness logging: block-wise gains
• Output structure preserves relative VoxCeleb1 path under out_root
"""

import os, random, math, tempfile, shutil
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

try:
    import soxr
    HAS_SOXR = True
except Exception:
    HAS_SOXR = False

try:
    from tqdm import tqdm
    USE_TQDM = True
except Exception:
    USE_TQDM = False

# ================== USER CONFIG ==================
indir             = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/VoxCeleb1"          # VoxCeleb1 root (contains id*/ subfolders)
musan_speech_dir  = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/musan/speech"
musan_music_dir   = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/musan/music"
out_root          = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/VoxCeleb1_complex"

target_sr         = 16000

# SNR frame logging params (✅ per your ask)
frame_ms          = 25.0   # 25 ms window
hop_ms            = 10.0   # 10 ms hop
frame_len         = int(round(frame_ms/1000.0 * target_sr))
hop_len           = int(round(hop_ms/1000.0 * target_sr))

# Mix params
babble_snr_range  = (0.0, 15.0)   # dB
music_snr_range   = (0.0, 10.0)   # dB

# Dynamic-loudness params
dyn_block_len_s   = (0.25, 1.0)   # seconds
dyn_gain_db_range = (-8.0, 8.0)   # dB per block
dyn_fade_s        = (0.04, 0.10)  # seconds (cosine fade)

# Reproducibility
random_seed       = 20250830
random.seed(random_seed)
np.random.seed(random_seed)

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".aac", ".m4a")

# ================== UTILS ==================
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def walk_wavs(root: str) -> List[Tuple[str,str]]:
    """
    Return list of (full_path, rel_path) for all audio files under root.
    rel_path is relative to root so we can preserve subfolder structure.
    """
    files = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(AUDIO_EXTS):
                full = os.path.join(r, f)
                rel  = os.path.relpath(full, root)
                files.append((full, rel))
    files.sort(key=lambda x: x[1])
    return files

def speaker_from_rel(rel_path: str) -> str:
    """
    VoxCeleb1 layout: idXXXX/…/file.wav  -> speaker = first directory name.
    """
    parts = rel_path.replace("\\","/").split("/")
    return parts[0] if parts else "unknown"

def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float32)
    return np.mean(x, axis=1).astype(np.float32)

def resample_1d(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    if HAS_SOXR:
        return soxr.resample(x.astype(np.float32), sr_in, sr_out).astype(np.float32)
    return librosa.resample(x.astype(np.float32), orig_sr=sr_in, target_sr=sr_out, res_type="kaiser_best").astype(np.float32)

def load_mono_resampled_any(path: str, sr: int) -> np.ndarray:
    """
    Robust loader: try soundfile, fallback to librosa; enforce mono @ sr; DC-remove.
    """
    try:
        y, srin = sf.read(path, always_2d=False)
        y = to_mono(np.asarray(y, dtype=np.float32))
    except Exception:
        y, srin = librosa.load(path, sr=None, mono=True)
        y = y.astype(np.float32)
    y = resample_1d(y, srin, sr)
    y = y - np.mean(y)
    return y.astype(np.float32)

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)

def normalize_peak(x: np.ndarray, peak: float = 0.999) -> np.ndarray:
    m = np.max(np.abs(x)) + 1e-12
    if m > peak:
        x = x * (peak / m)
    return x.astype(np.float32)

def collect_noise_files(noise_root: str) -> List[str]:
    files = []
    for r, _, fs in os.walk(noise_root):
        for f in fs:
            if f.lower().endswith(".wav"):
                files.append(os.path.join(r, f))
    files.sort()
    return files

def make_noise_track(y: np.ndarray, N: int) -> np.ndarray:
    """
    Create an N-length slice from noise y:
    - if y >= N: random start slice
    - else      : tile and take a random slice of length N
    """
    L = len(y)
    if L == 0:
        return np.zeros(N, dtype=np.float32)
    if L >= N:
        start = np.random.randint(0, L - N + 1)
        return y[start:start+N].astype(np.float32)
    reps = int(np.ceil((N + L) / L))
    yy = np.tile(y, reps)
    start = np.random.randint(0, len(yy) - N + 1)
    return yy[start:start+N].astype(np.float32)

def snr_gain_for_mix(signal: np.ndarray, noise: np.ndarray, target_snr_db: float) -> float:
    p_sig = np.mean(signal**2) + 1e-12
    p_noi = np.mean(noise**2) + 1e-12
    return float(np.sqrt(p_sig / (p_noi * 10**(target_snr_db/10.0))))

# ================== AUG FUNCTIONS (all @16k mono) ==================
def augment_babble(sig: np.ndarray, speech_noises: List[str]) -> Tuple[np.ndarray, float]:
    N = len(sig)
    k = random.randint(2, 4)
    picks = random.sample(speech_noises, k)
    noise_sum = np.zeros(N, dtype=np.float32)
    for p in picks:
        n = load_mono_resampled_any(p, target_sr)
        n = make_noise_track(n, N)
        noise_sum += n
    nr = rms(noise_sum)
    if nr > 0:
        noise_sum /= nr
    target_snr = random.uniform(*babble_snr_range)
    gain = snr_gain_for_mix(sig, noise_sum, target_snr)
    mix = normalize_peak(sig + noise_sum * gain)
    return mix, target_snr

def augment_music(sig: np.ndarray, music_noises: List[str]) -> Tuple[np.ndarray, float]:
    N = len(sig)
    p = random.choice(music_noises)
    n = load_mono_resampled_any(p, target_sr)
    n = make_noise_track(n, N)
    nr = rms(n)
    if nr > 0:
        n /= nr
    target_snr = random.uniform(*music_snr_range)
    gain = snr_gain_for_mix(sig, n, target_snr)
    mix = normalize_peak(sig + n * gain)
    return mix, target_snr

def augment_dynamic_loudness(sig: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int,int,float]]]:
    N = len(sig)
    out = np.zeros_like(sig)
    edits = []
    pos = 0
    while pos < N:
        block_len = int(np.random.uniform(*dyn_block_len_s) * target_sr)
        end = min(pos + max(1, block_len), N)
        seg = sig[pos:end].copy()

        gain_db = float(np.random.uniform(*dyn_gain_db_range))
        seg *= 10**(gain_db/20.0)

        fade_len = int(np.random.uniform(*dyn_fade_s) * target_sr)
        fade_len = min(fade_len, len(seg)//2)
        if fade_len > 0:
            fade = 0.5*(1-np.cos(np.linspace(0, np.pi, fade_len, dtype=np.float32)))
            seg[:fade_len]  *= fade
            seg[-fade_len:] *= fade[::-1]

        out[pos:end] = seg
        edits.append((pos, end, gain_db))
        pos = end
    return normalize_peak(out), edits

def per_frame_snr_rows(orig: np.ndarray, mixed: np.ndarray,
                       frame: int, hop: int, rel_fn: str, target_snr_db: float,
                       sr: int = target_sr):
    """
    25 ms frame, 10 ms hop -> SNR per hop.
    """
    N = len(orig)
    start = 0
    while start < N:
        end = min(start + frame, N)
        seg_sig   = orig[start:end]
        seg_noise = mixed[start:end] - seg_sig
        ps = np.mean(seg_sig**2) + 1e-12
        pn = np.mean(seg_noise**2) + 1e-12
        snr_db = 10.0 * np.log10(ps / pn)
        yield {
            "filename": rel_fn,
            "start": start,
            "end": end,
            "start_sec": round(start / sr, 6),
            "end_sec": round(end / sr, 6),
            "target_snr_db": round(target_snr_db, 2),
            "measured_snr_db": round(float(snr_db), 2)
        }
        start += hop

# ================== CLASS-BALANCED SPLIT (per speaker) ==================
def class_balanced_4way(files: List[Tuple[str,str]]) -> Dict[str, str]:
    """
    files: list of (full_path, rel_path)
    Return: {rel_path -> bucket in ['babble','music','loudness','clean']}
    Ensure per-speaker ~25% split for each bucket.
    """
    by_spk = defaultdict(list)
    for full, rel in files:
        spk = speaker_from_rel(rel)
        by_spk[spk].append(rel)

    assign = {}
    buckets = ['babble', 'music', 'loudness', 'clean']

    for spk in sorted(by_spk.keys()):
        lst = by_spk[spk]
        random.shuffle(lst)
        n = len(lst)
        if n < 4:
            for i, rel in enumerate(lst):
                assign[rel] = buckets[i % 4]
            continue
        idx = [
            (0,              int(round(0.25*n))),
            (int(round(0.25*n)), int(round(0.50*n))),
            (int(round(0.50*n)), int(round(0.75*n))),
            (int(round(0.75*n)), n)
        ]
        for b, (s, e) in enumerate(idx):
            for rel in lst[s:e]:
                assign[rel] = buckets[b]

    # sanity
    if len(assign) != len(files):
        have = set(assign.keys())
        expected = set([rel for _, rel in files])
        missing = list(expected - have)[:5]
        raise RuntimeError(f"Some files were not assigned: {missing}")
    return assign

# --------- optional: safe Excel writer ----------
def write_excel_safe(excel_path: str, sheets: Dict[str, pd.DataFrame]):
    engine = None
    try:
        import xlsxwriter  # noqa
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa
            engine = "openpyxl"
        except Exception:
            engine = None

    if engine is None:
        print("[WARN] No Excel engine available. Skipping .xlsx (CSVs are already saved).")
        return

    tmpdir = tempfile.mkdtemp(prefix="xlsx_tmp_", dir=os.path.dirname(excel_path) or None)
    tmpfile = os.path.join(tmpdir, "tmp.xlsx")
    try:
        with pd.ExcelWriter(tmpfile, engine=engine) as xw:
            for sheet, df in sheets.items():
                (df if df is not None else pd.DataFrame()).to_excel(xw, index=False, sheet_name=sheet)
        os.replace(tmpfile, excel_path)
        print(f"[OK] Wrote Excel: {excel_path}")
    except Exception as e:
        print(f"[WARN] Excel write failed: {e}. Keeping CSVs only.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ================== MAIN ==================
def main():
    ensure_dir(out_root)

    # Gather source wavs
    files = walk_wavs(indir)
    if not files:
        raise RuntimeError(f"No audio files found under {indir}")

    # Per-speaker balanced 4-way split
    assignment = class_balanced_4way(files)

    # Noise pools (WAV only; they will be resampled & sliced)
    speech_noises = collect_noise_files(musan_speech_dir)
    music_noises  = collect_noise_files(musan_music_dir)
    if len(speech_noises) == 0:
        raise RuntimeError(f"No WAV files under musan_speech_dir: {musan_speech_dir}")
    if len(music_noises) == 0:
        raise RuntimeError(f"No WAV files under musan_music_dir: {musan_music_dir}")

    # Logs
    manifest_rows, snr_rows, loud_block_rows = [], [], []

    iterator = tqdm(files, desc="VoxCeleb1 @16k Processing") if USE_TQDM else files
    for full, rel in iterator:
        spk = speaker_from_rel(rel)
        bucket = assignment[rel]

        # Load & resample speech
        sig = load_mono_resampled_any(full, target_sr)

        # Output path (preserve rel structure)
        out_path = os.path.join(out_root, rel)
        ensure_dir(os.path.dirname(out_path))

        if bucket == "babble":
            mixed, target_snr = augment_babble(sig, speech_noises)
            sf.write(out_path, mixed, target_sr)
            for row in per_frame_snr_rows(sig, mixed, frame_len, hop_len, rel, target_snr):
                row["noise_type"] = "babble"
                snr_rows.append(row)
            manifest_rows.append({
                "orig_relpath": rel,
                "speaker": spk,
                "noise_type": "babble",
                "out_relpath": os.path.relpath(out_path, out_root),
                "param": f"target_snr_db={target_snr:.2f}"
            })

        elif bucket == "music":
            mixed, target_snr = augment_music(sig, music_noises)
            sf.write(out_path, mixed, target_sr)
            for row in per_frame_snr_rows(sig, mixed, frame_len, hop_len, rel, target_snr):
                row["noise_type"] = "music"
                snr_rows.append(row)
            manifest_rows.append({
                "orig_relpath": rel,
                "speaker": spk,
                "noise_type": "music",
                "out_relpath": os.path.relpath(out_path, out_root),
                "param": f"target_snr_db={target_snr:.2f}"
            })

        elif bucket == "loudness":
            proc, edits = augment_dynamic_loudness(sig)
            sf.write(out_path, proc, target_sr)
            for (s, e, gdb) in edits:
                loud_block_rows.append({
                    "filename": rel,
                    "block_start": s,
                    "block_end": e,
                    "gain_db": round(gdb, 2)
                })
            manifest_rows.append({
                "orig_relpath": rel,
                "speaker": spk,
                "noise_type": "loudness",
                "out_relpath": os.path.relpath(out_path, out_root),
                "param": f"blocks={len(edits)}"
            })

        elif bucket == "clean":
            out = normalize_peak(sig)
            sf.write(out_path, out, target_sr)
            manifest_rows.append({
                "orig_relpath": rel,
                "speaker": spk,
                "noise_type": "clean",
                "out_relpath": os.path.relpath(out_path, out_root),
                "param": ""
            })

        else:
            raise RuntimeError(f"Unknown bucket: {bucket}")

    # ------ Write CSVs ------
    manifest_csv = os.path.join(out_root, "voxceleb1_manifest.csv")
    snr_csv      = os.path.join(out_root, "voxceleb1_snr_frames.csv")
    loud_csv     = os.path.join(out_root, "voxceleb1_loudness.csv")

    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    pd.DataFrame(snr_rows).to_csv(snr_csv, index=False)
    pd.DataFrame(loud_block_rows).to_csv(loud_csv, index=False)

    print("CSV logs written:")
    print(" -", manifest_csv)
    print(" -", snr_csv)
    print(" -", loud_csv)

    # ------ Optional Excel ------
    excel_path = os.path.join(out_root, "voxceleb1_aug.xlsx")
    write_excel_safe(excel_path, {
        "manifest": pd.DataFrame(manifest_rows),
        "snr_frames": pd.DataFrame(snr_rows),
        "loudness_blocks": pd.DataFrame(loud_block_rows),
    })

    # ------ Summary (by bucket / speaker count) ------
    print("\n=== Bucket counts by speaker ===")
    counts = defaultdict(lambda: defaultdict(int))
    for _, rel in files:
        b = assignment[rel]
        s = speaker_from_rel(rel)
        counts[b][s] += 1
    for b in ["babble","music","loudness","clean"]:
        total = sum(counts[b].values())
        print(f"{b:9s}: {total} files, {len(counts[b])} speakers covered")

    print(f"\nDone! Outputs under: {out_root}")
    print(f"[INFO] Resampler: {'soxr' if HAS_SOXR else 'librosa.kaiser_best'}; target_sr={target_sr} Hz")
    print(f"[INFO] SNR frames: {frame_ms} ms window, {hop_ms} ms hop")

if __name__ == "__main__":
    main()
