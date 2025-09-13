#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FMA-Small "complex" augmentation @16 kHz (single output folder + CSV + optional Excel)

• 统一重采样到 16 kHz 单声道
• 四路增强：babble（MUSAN/speech）、music（MUSAN/music）、dynamic-loudness、clean
• 类内均衡：使用 tracks.csv + genres.csv 将每个 track_id 映射到顶层流派（genre），
  然后在“每个 genre 内”做 25% 四等分 → babble / music / loudness / clean
• 输出：
    - fma_small_manifest.csv
    - fma_small_snr_frames.csv      （25ms 窗，10ms hop）
    - fma_small_loudness.csv
    - (optional) fma_small_aug.xlsx  [manifest / snr_frames / loudness_blocks]
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

in_dir             = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/fma/fma_small"  
musan_speech_dir   = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/musan/speech"
musan_music_dir    = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/musan/music"
out_root           = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/fma/fma_small_complex"

# 直接指定你的 metadata CSV 路径
tracks_csv         = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/fma/fma_metadata/fma_metadata/tracks.csv"
genres_csv         = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/fma/fma_metadata/fma_metadata/genres.csv"

target_sr          = 16000

# Frame SNR params
frame_ms           = 25.0
hop_ms             = 10.0
frame_len          = int(round(frame_ms/1000.0 * target_sr))
hop_len            = int(round(hop_ms/1000.0 * target_sr))

# Mix params
babble_snr_range   = (0.0, 15.0)   # dB
music_snr_range    = (0.0, 15.0)   # dB

# Dynamic loudness params
dyn_block_len_s    = (0.25, 1.0)   # s
dyn_gain_db_range  = (-8.0, 8.0)   # dB per block
dyn_fade_s         = (0.04, 0.10)  # s (cosine fade)

# Reproducibility
random_seed        = 20250830
random.seed(random_seed)
np.random.seed(random_seed)

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".aac", ".m4a")

# ================== UTILS ==================
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def walk_audio_files(root: str) -> List[str]:
    files = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(AUDIO_EXTS):
                files.append(os.path.join(r, f))
    files.sort()
    return files

def change_ext_to_wav(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return base + ".wav"

def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float32)
    return np.mean(x, axis=1).astype(np.float32)

def resample_1d(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    if HAS_SOXR:
        return soxr.resample(x.astype(np.float32), sr_in, sr_out).astype(np.float32)
    # 高质量重采样
    return librosa.resample(x.astype(np.float32), orig_sr=sr_in, target_sr=sr_out, res_type="kaiser_best").astype(np.float32)

def load_mono_resampled_any(path: str, sr: int) -> np.ndarray:
    """优先 soundfile 读取，不行再用 librosa；统一到单声道 & 指定采样率。"""
    try:
        y, srin = sf.read(path, always_2d=False)
        y = to_mono(np.asarray(y, dtype=np.float32))
    except Exception:
        y, srin = librosa.load(path, sr=None, mono=True)  # 保持原采样率
        y = y.astype(np.float32)
    y = resample_1d(y, srin, sr)
    # 去直流
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
    """
    从噪声 y 中取长度 N 的片段：
    • 若 y 长 >= N：随机起点截取 N；
    • 若 y 长 < N：循环拼接后再随机截取 N。
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

# ================== METADATA (tracks.csv + genres.csv) ==================
def build_track2genre(tracks_csv: str, genres_csv: str) -> Dict[int, str]:
    """
    读取 genres.csv 建立 {genre_id -> title}，
    读取 tracks.csv（多级表头），优先从顶层 genre id 列取 id→映射为标题，得 {track_id -> genre_name}
    """
    if not (os.path.isfile(tracks_csv) and os.path.isfile(genres_csv)):
        print("[WARN] Metadata CSVs not found; using single-class 'unknown'.")
        return {}

    # genre_id -> title
    gdf = pd.read_csv(genres_csv)
    # 兼容列名大小写或不同写法
    g_id_col = None
    for c in gdf.columns:
        if str(c).lower() in ("genre_id", "id"):
            g_id_col = c
            break
    title_col = None
    for c in gdf.columns:
        if str(c).lower() in ("title", "genre_title", "name"):
            title_col = c
            break
    if g_id_col is None or title_col is None:
        print("[WARN] genres.csv missing expected columns; using raw ids as names.")
        gid2name = {}
    else:
        gid2name = dict(zip(gdf[g_id_col], gdf[title_col]))

    # tracks.csv 读为 MultiIndex 列（官方格式）
    tdf = pd.read_csv(tracks_csv, header=[0, 1], index_col=0)

    # 可能的顶层 genre 列（不同版本命名各异）
    candidates = [
        ("track", "genre_top_level_id"),
        ("track", "genre_top_level"),
        ("track", "genre_top_id"),
        ("track", "genre_top"),
        ("genres", "top_level"),
    ]
    col = None
    for cand in candidates:
        if cand in tdf.columns:
            col = cand
            break
    if col is None:
        # 再尝试所有列里包含 "genre" 且是数字的
        for c in tdf.columns:
            if "genre" in "".join(map(str, c)).lower():
                col = c
                break
    if col is None:
        print("[WARN] Could not locate a top-level genre column; using 'unknown'.")
        return {}

    track2genre: Dict[int, str] = {}
    series = tdf[col]
    for track_id, raw in series.items():
        name: str
        try:
            gid = int(raw)
            name = gid2name.get(gid, str(gid))
        except Exception:
            # 有的版本可能直接是字符串标题
            name = str(raw)
        try:
            tid = int(track_id)
        except Exception:
            # index 不是纯数字时跳过
            continue
        # 清洗空/NaN
        if name is None or str(name).lower() in ("nan", "", "none"):
            name = "unknown"
        track2genre[tid] = str(name)
    return track2genre

def parse_track_id_from_path(path: str) -> Optional[int]:
    """FMA small 的文件名通常是 '000123.mp3'；提取数字作为 track_id。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem)
    except Exception:
        return None

def get_class_for_file(path: str, track2genre: Dict[int, str]) -> str:
    tid = parse_track_id_from_path(path)
    if tid is not None and tid in track2genre:
        return track2genre[tid]
    return "unknown"

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
                       frame: int, hop: int, filename: str, target_snr_db: float,
                       sr: int = target_sr):
    """25ms 窗, 10ms hop 的逐帧 SNR 记录。"""
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

# ================== CLASS-BALANCED SPLIT (by genre) ==================
def class_balanced_4way(files: List[str], track2genre: Dict[int,str]) -> Dict[str, str]:
    """
    对每个 genre 内部做 4 路均分（约 25%：babble / music / loudness / clean）。
    返回: {file_path -> bucket}
    """
    by_cls = defaultdict(list)
    for fp in files:
        c = get_class_for_file(fp, track2genre)
        by_cls[c].append(fp)

    assign = {}
    buckets = ['babble', 'music', 'loudness', 'clean']

    for c in sorted(by_cls.keys()):
        lst = by_cls[c]
        random.shuffle(lst)
        n = len(lst)
        if n < 4:
            # 类别样本过少时轮转分配
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
        raise RuntimeError(f"Some files were not assigned: {list(missing)[:5]}")
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

    # 收集 FMA 源文件
    files = walk_audio_files(in_dir)
    if not files:
        raise RuntimeError(f"No audio files found under {in_dir}")

    # 读取元数据（必须指定 tracks.csv 与 genres.csv）
    track2genre = build_track2genre(tracks_csv, genres_csv)
    if not track2genre:
        print("[WARN] Failed to build track->genre map; using single-class 'unknown' split.")

    # 类内四等分分配
    assignment = class_balanced_4way(files, track2genre)

    # 噪声池
    speech_noises = collect_noise_files(musan_speech_dir)
    music_noises  = collect_noise_files(musan_music_dir)
    if len(speech_noises) == 0:
        raise RuntimeError(f"No WAV files under musan_speech_dir: {musan_speech_dir}")
    if len(music_noises) == 0:
        raise RuntimeError(f"No WAV files under musan_music_dir: {musan_music_dir}")

    manifest_rows, snr_rows, loud_block_rows = [], [], []
    iterator = tqdm(files, desc="FMA-Small @16k Processing") if USE_TQDM else files

    for src in iterator:
        # 统一重采样到 16k 单声道
        sig = load_mono_resampled_any(src, target_sr)

        # 解析类别（用于 manifest）
        genre = get_class_for_file(src, track2genre)

        # 输出文件名（仅 basename，改后缀为 .wav）
        out_fn = change_ext_to_wav(src)
        out_path = os.path.join(out_root, out_fn)
        ensure_dir(os.path.dirname(out_path))

        bucket = assignment[src]

        if bucket == "babble":
            mixed, target_snr = augment_babble(sig, speech_noises)
            sf.write(out_path, mixed, target_sr)
            for row in per_frame_snr_rows(sig, mixed, frame_len, hop_len, out_fn, target_snr):
                row["noise_type"] = "babble"
                snr_rows.append(row)
            manifest_rows.append({
                "orig_relpath": os.path.relpath(src, in_dir),
                "track_id": parse_track_id_from_path(src),
                "genre": genre,
                "noise_type": "babble",
                "out_relpath": os.path.relpath(out_path, out_root),
                "param": f"target_snr_db={target_snr:.2f}"
            })

        elif bucket == "music":
            mixed, target_snr = augment_music(sig, music_noises)
            sf.write(out_path, mixed, target_sr)
            for row in per_frame_snr_rows(sig, mixed, frame_len, hop_len, out_fn, target_snr):
                row["noise_type"] = "music"
                snr_rows.append(row)
            manifest_rows.append({
                "orig_relpath": os.path.relpath(src, in_dir),
                "track_id": parse_track_id_from_path(src),
                "genre": genre,
                "noise_type": "music",
                "out_relpath": os.path.relpath(out_path, out_root),
                "param": f"target_snr_db={target_snr:.2f}"
            })

        elif bucket == "loudness":
            proc, edits = augment_dynamic_loudness(sig)
            sf.write(out_path, proc, target_sr)
            for (s, e, gdb) in edits:
                loud_block_rows.append({
                    "filename": out_fn,
                    "block_start": s,
                    "block_end": e,
                    "gain_db": round(gdb, 2)
                })
            manifest_rows.append({
                "orig_relpath": os.path.relpath(src, in_dir),
                "track_id": parse_track_id_from_path(src),
                "genre": genre,
                "noise_type": "loudness",
                "out_relpath": os.path.relpath(out_path, out_root),
                "param": f"blocks={len(edits)}"
            })

        elif bucket == "clean":
            out = normalize_peak(sig)
            sf.write(out_path, out, target_sr)
            manifest_rows.append({
                "orig_relpath": os.path.relpath(src, in_dir),
                "track_id": parse_track_id_from_path(src),
                "genre": genre,
                "noise_type": "clean",
                "out_relpath": os.path.relpath(out_path, out_root),
                "param": ""
            })
        else:
            raise RuntimeError(f"Unknown bucket: {bucket}")

    # ------ 写 CSV ------
    ensure_dir(out_root)
    manifest_csv = os.path.join(out_root, "fma_small_manifest.csv")
    snr_csv      = os.path.join(out_root, "fma_small_snr_frames.csv")
    loud_csv     = os.path.join(out_root, "fma_small_loudness.csv")

    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    pd.DataFrame(snr_rows).to_csv(snr_csv, index=False)
    pd.DataFrame(loud_block_rows).to_csv(loud_csv, index=False)

    print("CSV logs written:")
    print(" -", manifest_csv)
    print(" -", snr_csv)
    print(" -", loud_csv)

    # ------ 可选：写 Excel ------
    excel_path = os.path.join(out_root, "fma_small_aug.xlsx")
    write_excel_safe(excel_path, {
        "manifest": pd.DataFrame(manifest_rows),
        "snr_frames": pd.DataFrame(snr_rows),
        "loudness_blocks": pd.DataFrame(loud_block_rows),
    })

    # ------ 汇总（按 bucket / genre）------
    print("\n=== Bucket counts by genre ===")
    counts = defaultdict(lambda: defaultdict(int))
    for fp, b in assignment.items():
        g = get_class_for_file(fp, track2genre)
        counts[b][g] += 1
    for b in ["babble","music","loudness","clean"]:
        gs = ", ".join([f"{g}:{counts[b][g]}" for g in sorted(counts[b].keys())])
        print(f"{b:9s}: {gs}")

    print(f"\nDone! Outputs under: {out_root}")
    print(f"[INFO] Resampler: {'soxr' if HAS_SOXR else 'librosa.kaiser_best'}; target_sr={target_sr} Hz")

if __name__ == "__main__":
    main()
