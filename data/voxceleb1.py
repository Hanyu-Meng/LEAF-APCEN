import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torchaudio
import librosa                     
from torch.utils.data import Dataset

# -------------------- helpers --------------------
def _parse_split_file(split_file: str) -> List[Tuple[int, str]]:
    rows = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad line in split file: {line}")
            sid, rel = int(parts[0]), parts[1]
            rows.append((sid, rel))
    return rows

def _speaker_from_relpath(rel: str) -> str:
    rel = rel.replace("\\", "/")
    return rel.split("/")[0]

def _build_spk2idx(rows: List[Tuple[int, str]]) -> Dict[str, int]:
    speakers = sorted({_speaker_from_relpath(rel) for _, rel in rows})
    return {spk: i for i, spk in enumerate(speakers)}

def _frame_1d_np(sig_1d: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Return frames as (n_frames, frame_len). Assumes sig_1d is 1D np.float."""
    # librosa.util.frame -> shape (frame_len, n_frames); transpose to (n_frames, frame_len)
    if sig_1d.ndim != 1:
        sig_1d = np.squeeze(sig_1d)
    return np.transpose(librosa.util.frame(sig_1d, frame_length=frame_len, hop_length=hop))

# -------------------- dataset --------------------
class Dataset_Voxceleb1(Dataset):
    """
    VoxCeleb1 identification dataset from a split file.

    Args:
        root_dir: VoxCeleb1 wav root containing idxxxxx folders (NOT dev/test).
        subset: 'train' (split_id==1) or 'test' (split_id==3).
        front_end: if in {'Ada_FE','Simplified_AdaLeaf', ...}, emit framed chunks.
        max_len: segment length in samples (default 16000 = 1s at 16kHz).
        dynamic/data_type: keep your existing augmentation dir scheme.
    """
    def __init__(
        self,
        root_dir: str,
        subset: str = "train",
        front_end: Optional[str] = None,
        dynamic: bool = False,
        data_type: str = "babble",
        max_len: int = 16000,
        small_frame_len: int = 400,   # ~20.3 ms @16k
        small_hop: int = 160,         # ~11.0 ms @16k
    ):
        super().__init__()
        assert subset in {"train", "test"}, "subset must be 'train' or 'test'"

        # dynamic input root selection (your original logic)
        if dynamic:
            if data_type in ['babble', 'music']:
                self.root_dir = root_dir + '_' + data_type
            elif data_type == 'loudness':
                self.root_dir = root_dir + '_' + data_type
            else:
                self.root_dir = root_dir
        else:
            self.root_dir = root_dir

        self.split_file = os.path.join(root_dir, "iden_split_list.txt")
        self.subset = subset
        self.max_len = int(max_len)
        self.front_end = front_end
        self.target_sr = 16000

        # which frontends want framed input
        self._adaptive_front_ends = {
            "Ada_FE", "AdaFE",
            "Simplify_AdaLeaf"
        }
        self.small_frame_len = int(small_frame_len)
        self.small_hop = int(small_hop)

        all_rows = _parse_split_file(self.split_file)
        self.spk2idx = _build_spk2idx(all_rows)

        wanted_sid = 1 if subset == "train" else 3
        rows = [(sid, rel) for sid, rel in all_rows if sid == wanted_sid]
        if len(rows) == 0:
            raise RuntimeError(f"No entries with split_id={wanted_sid} in {self.split_file}")

        self.files: List[str] = [os.path.join(self.root_dir, rel) for _, rel in rows]
        self.labels: List[int] = [self.spk2idx[_speaker_from_relpath(rel)] for _, rel in rows]

        torchaudio.set_audio_backend("soundfile")

    def __len__(self) -> int:
        return len(self.files)

    def _load_wav_16k(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)  # (C, T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        return wav  # (1, T), float32

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.files[idx]
        label = self.labels[idx]
        wav = self._load_wav_16k(path)          # (1, T)
        T = wav.shape[-1]
        L = self.max_len

        # Whether to produce framed output
        is_adaptive = (self.front_end in self._adaptive_front_ends)

        if self.subset == "train":
            # ---- Random crop/pad to L ----
            if T >= L:
                start = np.random.randint(0, T - L + 1)
                seg = wav[:, start:start + L]   # (1, L)
            else:
                seg = torch.nn.functional.pad(wav, (0, L - T))  # (1, L)

            if not is_adaptive:
                # Return shape (1, L) -> model can treat as 1 frame
                return seg.squeeze(0).unsqueeze(0).contiguous().float(), label

            # ---- Adaptive front-ends: frame the 1-D segment into small frames ----
            sig = seg.squeeze(0).cpu().numpy().astype(np.float32)               # (L,)
            frames_np = _frame_1d_np(sig, self.small_frame_len, self.small_hop) # (T_small, L_small)
            frames = torch.from_numpy(frames_np)                                 # (T_small, L_small)
            return frames.contiguous().float(), label

        else:
            # ----------------------------- TEST -----------------------------
            if not is_adaptive:
                # Non-overlapping segments of length L: (nseg, L)
                if T < L:
                    wav = torch.nn.functional.pad(wav, (0, L - T))
                    T = wav.shape[-1]
                nseg = (T + L - 1) // L
                need = nseg * L - T
                if need:
                    wav = torch.nn.functional.pad(wav, (0, need))  # (1, nseg*L)
                frames = wav.squeeze(0).view(nseg, L)               # (nseg, L)
                return frames.contiguous().float(), label

            # Adaptive: frame each L-sized segment, then concatenate across all segments
            if T < L:
                wav = torch.nn.functional.pad(wav, (0, L - T))
                T = wav.shape[-1]
            nseg = (T + L - 1) // L
            need = nseg * L - T
            if need:
                wav = torch.nn.functional.pad(wav, (0, need))  # (1, nseg*L)
            sig_full = wav.squeeze(0).cpu().numpy().astype(np.float32)          # (nseg*L,)
            segs = sig_full.reshape(nseg, L)                                     # (nseg, L)

            all_small = []
            for i in range(nseg):
                f_i = _frame_1d_np(segs[i], self.small_frame_len, self.small_hop)  # (Ti, L_small)
                all_small.append(f_i)
            frames_np = np.concatenate(all_small, axis=0)                        # (T_total, L_small)
            frames = torch.from_numpy(frames_np)
            return frames.contiguous().float(), label
