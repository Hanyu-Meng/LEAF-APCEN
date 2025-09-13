#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESC-50 4-way augmentation @16 kHz (single output folder + CSV + optional Excel)

Key changes:
• 强制所有输入（ESC-50 + MUSAN 噪声）统一重采样到 16 kHz、单声道、float32。
• 噪声长度处理：随机起点截取；若不足则循环拼接再裁切，避免零填充造成音色异常。
• 更稳健的加载与重采样（优先 soxr，高质量；退回 librosa.resample）。
• 其余逻辑保持：按“文件名最后数字”为类做类内四等分；记录逐帧 SNR；输出 CSV(+可选 xlsx)。

Outputs:
• esc50_manifest.csv
• esc50_snr_frames.csv
• esc50_loudness.csv
• (optional) esc50_aug.xlsx
"""

import os, random, math, tempfile, shutil
from collections import defaultdict
from typing import List, Dict, Tuple

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

in_dir             = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/ESC50/audio"
musan_speech_dir   = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/musan/speech"
musan_music_dir    = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/musan/music"
out_root           = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/ESC50/esc50_complex/audio"
target_sr          = 16000

frame_ms           = 25.0
hop_ms             = 10.0
frame_len          = int(round(frame_ms/1000.0 * target_sr))
hop_len            = int(round(hop_ms/1000.0 * target_sr))

babble_snr_range   = (0.0, 15.0) 
music_snr_range    = (0.0, 15.0)  

dyn_block_len_s    = (0.25, 1.0)  
dyn_gain_db_range  = (-8.0, 8.0) 
dyn_fade_s         = (0.04, 0.10)

random_seed        = 86
random.seed(random_seed)
np.random.seed(random_seed)

AUDIO_EXTS = (".wav", ".flac", ".mp3")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def list_audio_files(d: str) -> List[str]:
    return sorted([f for f in os.listdir(d)
                   if f.lower().endswith(AUDIO_EXTS) and os.path.isfile(os.path.join(d, f))])

def parse_classid_from_filename(fn: str) -> int:
    stem = os.path.splitext(os.path.basename(fn))[0]
    parts = stem.split('-')
    if len(parts) >= 1:
        last = parts[-1]
        return int(last)
    raise ValueError(f"Cannot parse class id from filename: {fn}")

#convert stereo to mono
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

def load_mono_resampled(path: str, sr: int) -> np.ndarray:
    y, srin = sf.read(path, always_2d=False)  
    y = to_mono(np.asarray(y, dtype=np.float32))
    if not np.isfinite(y).all():
        y = np.nan_to_num(y)
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
    return files

def make_noise_track(y: np.ndarray, N: int) -> np.ndarray:
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

def augment_babble(sig: np.ndarray, speech_noises: List[str]) -> Tuple[np.ndarray, float]:
    N = len(sig)
    k = random.randint(2, 4)
    picks = random.sample(speech_noises, k)
    noise_sum = np.zeros(N, dtype=np.float32)
    for p in picks:
        n = load_mono_resampled(p, target_sr)         
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
    n = load_mono_resampled(p, target_sr)
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
                       frame: int, hop: int, filename: str, target_snr_db: float,
                       sr: int = target_sr):
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
            "filename": filename,
            "start": start,
            "end": end,
            "start_sec": round(start / sr, 6),
            "end_sec": round(end / sr, 6),
            "target_snr_db": round(target_snr_db, 2),
            "measured_snr_db": round(float(snr_db), 2)
        }
        start += hop

def class_balanced_4way(files: List[str]) -> Dict[str, str]:
    by_cls = defaultdict(list)
    for fn in files:
        cid = parse_classid_from_filename(fn)
        by_cls[cid].append(fn)

    assign = {}
    buckets = ['babble', 'music', 'loudness', 'clean']

    for cid in sorted(by_cls.keys()):
        lst = by_cls[cid]
        random.shuffle(lst)
        n = len(lst)
        if n < 4:
            for i, f in enumerate(lst):
                assign[f] = buckets[i % 4]
            continue
        idx = [
            (0,              int(round(0.25*n))),
            (int(round(0.25*n)), int(round(0.50*n))),
            (int(round(0.50*n)), int(round(0.75*n))),
            (int(round(0.75*n)), n)
        ]
        for b, (s, e) in enumerate(idx):
            for f in lst[s:e]:
                assign[f] = buckets[b]

    if len(assign) != len(files):
        missing = set(files) - set(assign.keys())
        raise RuntimeError(f"Some files were not assigned: {sorted(missing)[:10]}")
    return assign

def main():
    ensure_dir(out_root)

    manifest_rows, snr_rows, loud_block_rows = [], [], []

    files = list_audio_files(in_dir)
    if not files:
        raise RuntimeError(f"No audio files found in {in_dir}")

    assignment = class_balanced_4way(files)

    speech_noises = collect_noise_files(musan_speech_dir)
    music_noises  = collect_noise_files(musan_music_dir)
    if len(speech_noises) == 0:
        raise RuntimeError(f"No WAV files under musan_speech_dir: {musan_speech_dir}")
    if len(music_noises) == 0:
        raise RuntimeError(f"No WAV files under musan_music_dir: {musan_music_dir}")

    iterator = tqdm(files, desc="ESC-50 @16k Processing") if USE_TQDM else files
    for fn in iterator:
        src = os.path.join(in_dir, fn)
        class_id = parse_classid_from_filename(fn)
        bucket = assignment[fn]
        sig = load_mono_resampled(src, target_sr)

        out_path = os.path.join(out_root, fn)

        if bucket == "babble":
            mixed, target_snr = augment_babble(sig, speech_noises)
            sf.write(out_path, mixed, target_sr)
            for row in per_frame_snr_rows(sig, mixed, frame_len, hop_len, fn, target_snr):
                row["noise_type"] = "babble"
                snr_rows.append(row)
            manifest_rows.append({
                "orig_filename": fn,
                "class_id": class_id,
                "noise_type": "babble",
                "out_relpath": os.path.relpath(out_path, out_root),
                "param": f"target_snr_db={target_snr:.2f}"
            })

        elif bucket == "music":
            mixed, target_snr = augment_music(sig, music_noises)
            sf.write(out_path, mixed, target_sr)
            for row in per_frame_snr_rows(sig, mixed, frame_len, hop_len, fn, target_snr):
                row["noise_type"] = "music"
                snr_rows.append(row)
            manifest_rows.append({
                "orig_filename": fn,
                "class_id": class_id,
                "noise_type": "music",
                "out_relpath": os.path.relpath(out_path, out_root),
                "param": f"target_snr_db={target_snr:.2f}"
            })

        elif bucket == "loudness":
            proc, edits = augment_dynamic_loudness(sig)
            sf.write(out_path, proc, target_sr)
            for (s, e, gdb) in edits:
                loud_block_rows.append({
                    "filename": fn,
                    "block_start": s,
                    "block_end": e,
                    "gain_db": round(gdb, 2)
                })
            manifest_rows.append({
                "orig_filename": fn,
                "class_id": class_id,
                "noise_type": "loudness",
                "out_relpath": os.path.relpath(out_path, out_root),
                "param": f"blocks={len(edits)}"
            })

        elif bucket == "clean":
            out = normalize_peak(sig)
            sf.write(out_path, out, target_sr)
            manifest_rows.append({
                "orig_filename": fn,
                "class_id": class_id,
                "noise_type": "clean",
                "out_relpath": os.path.relpath(out_path, out_root),
                "param": ""
            })

        else:
            raise RuntimeError(f"Unknown bucket: {bucket}")

    # ------ Write CSVs ------
    manifest_csv = os.path.join(out_root, "esc50_manifest.csv")
    snr_csv      = os.path.join(out_root, "esc50_snr_frames.csv")
    loud_csv     = os.path.join(out_root, "esc50_loudness.csv")

    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    pd.DataFrame(snr_rows).to_csv(snr_csv, index=False)
    pd.DataFrame(loud_block_rows).to_csv(loud_csv, index=False)

    print("CSV logs written:")
    print(" -", manifest_csv)
    print(" -", snr_csv)
    print(" -", loud_csv)

    print("\n--- Bucket counts by class_id ---")
    counts = defaultdict(lambda: defaultdict(int))
    for fn, b in assignment.items():
        cid = parse_classid_from_filename(fn)
        counts[b][cid] += 1
    for b in ["babble","music","loudness","clean"]:
        cls_counts = ", ".join([f"{cid}:{counts[b][cid]}" for cid in sorted(counts[b].keys())])
        print(f"{b:9s}: {cls_counts}")

    print(f"\nOutputs under: {out_root}")
    print(f"[INFO] Resampler: {'soxr' if HAS_SOXR else 'librosa.kaiser_best'}; target_sr={target_sr} Hz")

if __name__ == "__main__":
    main()