"""
Create CREMA-D under complex environment dataset 
Author: Hanyu Meng 
Time: July 2025
• Writes three CSV files under out_root:
    - crema_manifest.csv      one row per utterance (noise type, params, paths)
    - crema_snr_frames.csv    per-frame SNR for babble & music
    - crema_loudness.csv      per-block gains for loudness
"""

import os, random, math
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

try:
    from tqdm import tqdm
    USE_TQDM = True
except Exception:
    USE_TQDM = False

in_dir             = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/crema_d"  # clean CREMA-D directory
musan_speech_dir   = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/musan/speech"
musan_music_dir    = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/musan/music"
out_root           = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/crema_d_complex"

target_sr          = 16000

frame_ms           = 25.0
hop_ms             = 10.0
frame_len          = int(round(frame_ms/1000.0 * target_sr))
hop_len            = int(round(hop_ms/1000.0 * target_sr))

babble_snr_range   = (0.0, 15.0)   # dB
music_snr_range    = (0.0, 15.0)   # dB

# dynamic-loudness params
dyn_block_len_s    = (0.25, 1.0)
dyn_gain_db_range  = (-8.0, 8.0)
dyn_fade_s         = (0.04, 0.10)

random_seed        = 86
random.seed(random_seed)
np.random.seed(random_seed)

CLASSES = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".aac")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def list_audio_files(d: str) -> List[str]:
    return sorted([f for f in os.listdir(d) if f.lower().endswith(AUDIO_EXTS) and os.path.isfile(os.path.join(d, f))])

def parse_emotion_from_filename(fn: str) -> str:
    parts = os.path.splitext(fn)[0].split('_')
    if len(parts) >= 3 and parts[2] in CLASSES:
        return parts[2]
    for c in CLASSES:
        if f"_{c}_" in fn:
            return c
    raise ValueError(f"Cannot parse emotion from filename: {fn}")

def load_audio(path: str, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr)
    return y.astype(np.float32)

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)

def normalize_peak(x: np.ndarray, peak: float = 0.999) -> np.ndarray:
    m = np.max(np.abs(x)) + 1e-12
    if m > peak:
        x = x * (peak / m)
    return x

def collect_noise_files(noise_root: str) -> List[str]:
    files = []
    for r, _, fs in os.walk(noise_root):
        for f in fs:
            if f.lower().endswith(".wav"):
                files.append(os.path.join(r, f))
    return files

def make_same_length(x: np.ndarray, N: int) -> np.ndarray:
    if len(x) >= N:
        return x[:N]
    out = np.zeros(N, dtype=np.float32)
    out[:len(x)] = x
    return out

def snr_gain_for_mix(signal: np.ndarray, noise: np.ndarray, target_snr_db: float) -> float:
    p_sig = np.mean(signal**2) + 1e-12
    p_noi = np.mean(noise**2) + 1e-12
    return np.sqrt(p_sig / (p_noi * 10**(target_snr_db/10.0)))


def augment_babble(sig: np.ndarray, speech_noises: List[str]) -> Tuple[np.ndarray, float]:
    N = len(sig)
    k = random.randint(2, 4)
    picks = random.sample(speech_noises, k)
    noise = np.zeros(N, dtype=np.float32)
    for p in picks:
        n = load_audio(p, target_sr)
        n = make_same_length(n, N)
        noise += n
    if rms(noise) > 0:
        noise /= rms(noise)
    target_snr = random.uniform(*babble_snr_range)
    gain = snr_gain_for_mix(sig, noise, target_snr)
    return normalize_peak(sig + noise * gain), target_snr

def augment_music(sig: np.ndarray, music_noises: List[str]) -> Tuple[np.ndarray, float]:
    N = len(sig)
    p = random.choice(music_noises)
    noise = load_audio(p, target_sr)
    noise = make_same_length(noise, N)
    if rms(noise) > 0:
        noise /= rms(noise)
    target_snr = random.uniform(*music_snr_range)
    gain = snr_gain_for_mix(sig, noise, target_snr)
    return normalize_peak(sig + noise * gain), target_snr

def augment_dynamic_loudness(sig: np.ndarray):
    N = len(sig)
    out = np.zeros_like(sig)
    edits = []
    pos = 0
    while pos < N:
        block_len = int(np.random.uniform(*dyn_block_len_s) * target_sr)
        end = min(pos + block_len, N)
        seg = sig[pos:end].copy()
        gain_db = float(np.random.uniform(*dyn_gain_db_range))
        seg *= 10**(gain_db/20.0)
        fade_len = int(np.random.uniform(*dyn_fade_s) * target_sr)
        fade_len = min(fade_len, len(seg)//2)
        if fade_len > 0:
            fade = 0.5*(1-np.cos(np.linspace(0,np.pi,fade_len,dtype=np.float32)))
            seg[:fade_len] *= fade
            seg[-fade_len:] *= fade[::-1]
        out[pos:end] = seg
        edits.append((pos, end, gain_db))
        pos = end
    return normalize_peak(out), edits

def per_frame_snr_rows(orig, mixed, frame, hop, filename, target_snr_db, sr=16000):
    N = len(orig)
    start = 0
    while start < N:
        end = min(start+frame, N)
        seg_sig, seg_noise = orig[start:end], mixed[start:end]-orig[start:end]
        ps, pn = np.mean(seg_sig**2)+1e-12, np.mean(seg_noise**2)+1e-12
        snr_db = 10*np.log10(ps/pn)
        yield {
            "filename": filename,
            "target_snr_db": round(target_snr_db,2),
            "start": start,
            "end": end,
            "start_sec": round(start/sr,6),
            "end_sec": round(end/sr,6),
            "measured_snr_db": round(float(snr_db),2)
        }
        start += hop

def class_balanced_4way(files: List[str]) -> Dict[str,str]:
    by_cls = defaultdict(list)
    for fn in files:
        by_cls[parse_emotion_from_filename(fn)].append(fn)
    assign = {}
    buckets = ["babble","music","loudness","clean"]
    for c,lst in by_cls.items():
        random.shuffle(lst)
        n=len(lst)
        idx=[(0,int(0.25*n)),(int(0.25*n),int(0.5*n)),
             (int(0.5*n),int(0.75*n)),(int(0.75*n),n)]
        if n<4:
            for i,f in enumerate(lst): assign[f]=buckets[i%4]
        else:
            for b,(s,e) in enumerate(idx):
                for f in lst[s:e]: assign[f]=buckets[b]
    return assign

def main():
    ensure_dir(out_root)
    manifest_rows, snr_rows, loud_block_rows = [], [], []

    files = list_audio_files(in_dir)
    assignment = class_balanced_4way(files)
    speech_noises, music_noises = collect_noise_files(musan_speech_dir), collect_noise_files(musan_music_dir)

    iterator = tqdm(files, desc="Processing") if USE_TQDM else files
    for fn in iterator:
        sig = load_audio(os.path.join(in_dir,fn), target_sr)
        emo, bucket = parse_emotion_from_filename(fn), assignment[fn]
        out_path = os.path.join(out_root, fn)

        if bucket=="babble":
            mixed, target_snr = augment_babble(sig, speech_noises)
            sf.write(out_path, mixed, target_sr)
            for row in per_frame_snr_rows(sig,mixed,frame_len,hop_len,fn,target_snr): snr_rows.append(row)
            manifest_rows.append({"orig_filename":fn,"emotion":emo,"noise_type":"babble",
                                  "out_relpath":os.path.relpath(out_path,out_root),"param":f"target_snr_db={target_snr:.2f}"})
        elif bucket=="music":
            mixed, target_snr = augment_music(sig, music_noises)
            sf.write(out_path, mixed, target_sr)
            for row in per_frame_snr_rows(sig,mixed,frame_len,hop_len,fn,target_snr): snr_rows.append(row)
            manifest_rows.append({"orig_filename":fn,"emotion":emo,"noise_type":"music",
                                  "out_relpath":os.path.relpath(out_path,out_root),"param":f"target_snr_db={target_snr:.2f}"})
        elif bucket=="loudness":
            proc, edits = augment_dynamic_loudness(sig)
            sf.write(out_path, proc, target_sr)
            for (s,e,gdb) in edits: loud_block_rows.append({"filename":fn,"block_start":s,"block_end":e,"gain_db":round(gdb,2)})
            manifest_rows.append({"orig_filename":fn,"emotion":emo,"noise_type":"loudness",
                                  "out_relpath":os.path.relpath(out_path,out_root),"param":f"blocks={len(edits)}"})
        elif bucket=="clean":
            sf.write(out_path, normalize_peak(sig), target_sr)
            manifest_rows.append({"orig_filename":fn,"emotion":emo,"noise_type":"clean",
                                  "out_relpath":os.path.relpath(out_path,out_root),"param":""})

    pd.DataFrame(manifest_rows).to_csv(os.path.join(out_root,"crema_manifest.csv"), index=False)
    pd.DataFrame(snr_rows).to_csv(os.path.join(out_root,"crema_snr_frames.csv"), index=False)
    pd.DataFrame(loud_block_rows).to_csv(os.path.join(out_root,"crema_loudness.csv"), index=False)

    print("CSV logs saved in:", out_root)

if __name__=="__main__":
    main()