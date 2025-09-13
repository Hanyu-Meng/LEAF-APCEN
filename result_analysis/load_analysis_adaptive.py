import torch
from efficientnet_pytorch import EfficientNet
from model import Ada_AudioClassifier
from model.leaf import *
from model.adaleaf import *
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator, LinearLocator, FormatStrFormatter

def load_wav_for_model(wav_path, target_sr=16000, max_len=16000):
    wav, sr = torchaudio.load(wav_path)  # wav: (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.squeeze(0)  # (T,)
    T = wav.shape[0]
    # Pad or crop to max_len (1s)
    if T >= max_len:
        wav = wav[:max_len]
    else:
        wav = torch.nn.functional.pad(wav, (0, max_len - T))
    # Add batch and channel dims: (1, 1, L)
    wav = wav.unsqueeze(0).unsqueeze(0)
    return wav

def frame_waveform(
    wav: torch.Tensor,
    sr: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    mono: bool = True,
) -> torch.Tensor:
    """
    wav: [N], [1, N], [B, N], or [B, C, N] (float tensor)
    returns frames: [B, T, L] with L = frame_len samples
    """
    # Ensure float tensor on same device
    wav = wav.float()

    # Normalize shapes -> [B, C, N]
    if wav.ndim == 1:                     # [N]
        wav = wav.unsqueeze(0).unsqueeze(0)
    elif wav.ndim == 2:                   # [B, N] or [C, N]
        wav = wav.unsqueeze(1)            # [B, 1, N]
    elif wav.ndim == 3:                   # [B, C, N]
        pass
    else:
        raise ValueError(f"Unexpected wav.ndim={wav.ndim}, expected 1/2/3.")

    B, C, N = wav.shape

    # Optional: downmix to mono
    if mono and C > 1:
        wav = wav.mean(dim=1, keepdim=True)  # [B, 1, N]
        C = 1

    # Sizes
    frame_len = int(round(sr * frame_ms / 1000.0))   # 25 ms @ 16kHz -> 400
    hop = int(round(sr * hop_ms / 1000.0))           # 10 ms @ 16kHz -> 160

    # Right pad so unfold produces full coverage (at least 1 frame)
    if N < frame_len:
        pad_right = frame_len - N
    else:
        remainder = (N - frame_len) % hop
        pad_right = 0 if remainder == 0 else (hop - remainder)

    if pad_right > 0:
        wav = F.pad(wav, (0, pad_right))  # pad last dim

    # Unfold over time (last dim) → [B, C, T, L]
    frames = wav.unfold(dimension=-1, size=frame_len, step=hop)

    # Squeeze channel → [B, T, L]
    frames = frames.mean(dim=1) if C > 1 else frames.squeeze(1)
    return frames  # [B, T, L]
# --- Model parameters (match your training config) ---
n_filters = 40
sample_rate = 16000
window_len = 25.0
window_stride = 10.0
min_freq = 60.0
max_freq = 7800.0
s = 0.04
dataset = 'CREMAD'  # 'ESC50', 'SpeechCOM_V1_30', 'SpeechCOM_V2', 'VoxCeleb1', 'GTZAN', 'CREMAD', 'IEMOCAP', 'FMA_Small', 'FMA_Medium'

# -------- audio calssification task -------- #
if dataset == 'ESC50': num_classes=50
elif dataset == 'SpeechCOM_V1_30': num_classes=30
elif dataset == 'SpeechCOM_V2': num_classes=35
elif dataset == 'VoxCeleb1': num_classes=1251
elif dataset == 'GTZAN': num_classes=10
elif dataset == 'CREMAD': num_classes=6
elif dataset == 'IEMOCAP': num_classes=4
elif dataset == 'FMA_Small': num_classes=8
elif dataset == 'FMA_Medium': num_classes=16

# --- Build frontend ---
compression_fn = Simplify_AdaptivePCEN(init_s=0.04, init_alpha=0.48, init_r=0.5, eps=1e-12, clamp=1e-5)
frontend = AdaptiveLeaf(
    n_filters=n_filters,
    min_freq=min_freq,
    max_freq=max_freq,
    sample_rate=sample_rate,
    window_len=window_len,
    window_stride=window_stride,
    compression=compression_fn
)

# --- Build encoder ---
encoder = EfficientNet.from_name("efficientnet-b0", num_classes=num_classes, include_top=False, in_channels=1)
encoder._avg_pooling = torch.nn.Identity()

# --- Build classifier ---
network = Ada_AudioClassifier(
    num_outputs=num_classes,
    n_filters=n_filters,
    frontend=frontend,
    encoder=encoder
)

# --- Load checkpoint ---
# ckpt_path = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/new_adaptive_pcen/checkpoints/Simplify_AdaLeaf/CREMAD_Simplify_AdaLeaf_model_100_0.0001_0.0001_original_EfficientNetB0_FC_CrossEn_Seg1S_clean_revised_20250831/epoch_165.pth"
ckpt_path = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/new_adaptive_pcen/checkpoints/Simplify_AdaLeaf/CREMAD_Simplify_AdaLeaf_model_100_0.0001_0.0001_original_EfficientNetB0_FC_CrossEn_Seg1S_complex_revised_20250901/epoch_129.pth"
state = torch.load(ckpt_path, map_location="cuda")
network.load_state_dict(state, strict=False)
network.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = network.to(device)
# wav_path = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/new_adaptive_pcen/audios/1001_IWW_SAD_XX.wav"
wav_path = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/new_adaptive_pcen/audios/1001_IWW_SAD_XX_babble_1.51.wav"
wav_tensor = load_wav_for_model(wav_path, target_sr=16000)
wav_tensor = wav_tensor.to(device)

# For Simplify_AdaLeaf, you may need to frame the waveform if your model expects [B, T, L]
# Example: frame into overlapping windows (20ms frame, 10ms hop)
# frame_len = 400   # 25ms @ 16kHz
# hop = 160         # 10ms @ 16kHz
# frames = wav_tensor.unfold(-1, frame_len, hop)  # shape: (1, 1, num_frames, frame_len)
# frames = frames.squeeze(1)  # (1, num_frames, frame_len)
frames = frame_waveform(wav_tensor, sr=16000, frame_ms=25.0, hop_ms=10.0)
# Inference
network.eval()
with torch.no_grad():
    fb_outputs = []
    num_frames = frames.shape[1]
    all_pcen_outputs = []
    all_alpha = []
    all_r = []
    all_fb_out = []
    M_prev = None
    pre_frame = None
    for i in range(num_frames):
        fb_out = frontend.filterbank(frames[:, i, :].unsqueeze(1))  # [B, n_filters, W]
        cur_wave = frames[:, i, :].unsqueeze(1)
        # Get filterbank output
        fb_out = frontend.filterbank(cur_wave)  # [B, n_filters, W]
        # Get PCEN output and params
        pcen_out, M_prev, alpha, r = frontend(cur_wave, pre_frame, M_prev)
        all_pcen_outputs.append(pcen_out)
        all_alpha.append(alpha)
        all_r.append(r)
        all_fb_out.append(fb_out)
        pre_frame = pcen_out

    fb_outputs = torch.stack(all_fb_out, dim=1)  # [B, T, n_filters, W]
    all_pcen_outputs = torch.stack(all_pcen_outputs, dim=1)  # [B, T, n_filters, W]
    all_alpha = torch.stack(all_alpha, dim=1)                # [B, T, n_filters, W] or [B, T, n_filters, 1]
    all_r = torch.stack(all_r, dim=1)
    all_fb_energy = fb_outputs.mean(dim=-1)  # [B, T, n_filters]
    all_pcen_outputs_energy = all_pcen_outputs.mean(dim=-1)  # [B, T, n_filters]
    gains = (all_pcen_outputs_energy + 1e-12) / (all_fb_energy + 1e-12)  # [B, T, n_filters]
    gains = gains.squeeze(0).cpu().numpy()
    gains_db = 20 * np.log10(gains + 1e-12)  # [T, n_filters]

    fb_energy_db = 10 * np.log10(all_fb_energy.squeeze(0).cpu().numpy() + 1e-12)         # [T, n_filters]
    pcen_energy_db = 10 * np.log10(all_pcen_outputs_energy.squeeze(0).cpu().numpy() + 1e-12)  # [T, n_filters]

# 1) Adaptive PCEN Gain
with plt.rc_context({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"]
}):
    plt.figure(figsize=(4, 2.3))
    im = plt.imshow(
        gains_db.T,           # [n_filters, T] → 频率 x 时间
        aspect='auto', origin='lower', interpolation='nearest', cmap='inferno'
    )
    plt.xlim(0,98)
    plt.ylim(0,39)
    cbar = plt.colorbar(im)
    cbar.set_label('Gain (dB)', fontsize=15)
    cbar.locator = LinearLocator(numticks=3)       # 想更密的话改 5/7/9
    cbar.formatter = FormatStrFormatter('%.0f')
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=14)

    plt.xlabel('Frame Index', fontsize=16)
    plt.ylabel('Filter Index', fontsize=16)
    # plt.title('Adaptive PCEN Gain', fontsize=16)

    ax = plt.gca()
    ax.set_facecolor("none")
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1)
    ax.tick_params(axis='both', which='minor', labelsize=12, length=3, width=0.8)

    plt.tight_layout()
    plt.savefig(
        '/media/mengh/SharedData/hanyu/Adaptive_PCEN/new_adaptive_pcen/pcen_gain_spectrogram_db.png',
        dpi=300, bbox_inches="tight", transparent=True)
    plt.show()


# 2) Filterbank Output (before PCEN)
with plt.rc_context({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"]
}):
    plt.figure(figsize=(4, 2.3))
    im1 = plt.imshow(
        fb_energy_db.T, aspect='auto', origin='lower', interpolation='nearest', cmap='inferno'
    )
    plt.xlim(0,98)
    plt.ylim(0,39)
    cbar = plt.colorbar(im1)
    cbar.set_label('Energy (dB)', fontsize=15)
    cbar.locator = LinearLocator(numticks=3)
    cbar.formatter = FormatStrFormatter('%.0f')
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=14)

    plt.xlabel('Frame Index', fontsize=16)
    plt.ylabel('Filter Index', fontsize=16)
    # plt.title('Filterbank Output (before PCEN)', fontsize=16)

    ax = plt.gca()
    ax.set_facecolor("none")
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1)
    ax.tick_params(axis='both', which='minor', labelsize=12, length=3, width=0.8)

    plt.tight_layout()
    plt.savefig(
        '/media/mengh/SharedData/hanyu/Adaptive_PCEN/new_adaptive_pcen/filterbank_energy_db.png',
        dpi=300, bbox_inches="tight", transparent=True)
    plt.show()


# 3) LEAF-APCEN Output Representation
with plt.rc_context({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"]
}):
    plt.figure(figsize=(4, 2.3))
    im2 = plt.imshow(
        pcen_energy_db.T, aspect='auto', origin='lower', interpolation='nearest', cmap='inferno'
    )
    plt.xlim(0,98)
    plt.ylim(0,39)
    cbar = plt.colorbar(im2)
    cbar.set_label('Energy (dB)', fontsize=15)
    cbar.locator = LinearLocator(numticks=3)
    cbar.formatter = FormatStrFormatter('%.0f')
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=14)

    plt.xlabel('Frame Index', fontsize=16)
    plt.ylabel('Filter Index', fontsize=16)
    # plt.title('LEAF-APCEN Output Representation', fontsize=16)

    ax = plt.gca()
    ax.set_facecolor("none")
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1)
    ax.tick_params(axis='both', which='minor', labelsize=12, length=3, width=0.8)

    plt.tight_layout()
    plt.savefig(
        '/media/mengh/SharedData/hanyu/Adaptive_PCEN/new_adaptive_pcen/adaptive_pcen_output_energy_db.png',
        dpi=300, bbox_inches="tight", transparent=True)
    plt.show()

frame_idx = 10  # choose a frame index in range [0, T-1]
alpha_frame = all_alpha.squeeze()[frame_idx, :]  # [n_filters]
r_frame = all_r.squeeze()[frame_idx, :]          # [n_filters]

alpha_frame = alpha_frame.cpu().numpy()
r_frame = r_frame.cpu().numpy()
# SPL range (in dB)
test_sample = 2000
Max_Energy_dB = 20
Min_Energy_dB = 50
spl_db = np.linspace(Min_Energy_dB, Max_Energy_dB, test_sample)
E_lin = pow(10, spl_db/20)  # convert dB SPL to linear
M = np.zeros([test_sample, n_filters]) 
M[:,0] = E_lin
M0 = np.zeros(test_sample)
PCEN = np.zeros([test_sample, n_filters])
pcen_gain = np.zeros([test_sample, n_filters])
for i in range(1, n_filters):
    M[:,i] = (1-s) * M[:,i-1] + s * E_lin  # M smoothing

plt.figure()
for i in range(n_filters):
    for j in range(test_sample):
        PCEN[j,i] = (E_lin[j]**r_frame[i]) / ((M[j,i] + 1e-12)**alpha_frame[i])
        pcen_gain[j,i] = PCEN[j,i]/spl_db[j]
    plt.plot(spl_db, pcen_gain[:,i], label=f'Filter {i+1}')

plt.xlabel('Input SPL (dB)', fontsize=16)
plt.ylabel('Gain', fontsize=16)
plt.show()