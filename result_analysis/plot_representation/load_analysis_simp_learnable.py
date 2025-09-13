import torch
from efficientnet_pytorch import EfficientNet
from model import AudioClassifier
from model.leaf import *
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, MaxNLocator, LinearLocator, FormatStrFormatter

def load_wav_for_model(wav_path, target_sr=16000, max_len=16000):
    wav, sr = torchaudio.load(wav_path)  
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.squeeze(0)
    T = wav.shape[0]
    if T >= max_len:
        wav = wav[:max_len]
    else:
        wav = torch.nn.functional.pad(wav, (0, max_len - T))
    wav = wav.unsqueeze(0).unsqueeze(0)
    return wav

# --- Model parameters ---
n_filters = 40
sample_rate = 16000
window_len = 25.0
window_stride = 10.0
min_freq = 60.0
max_freq = 7800.0
dataset = 'CREMAD'

if dataset == 'ESC50': num_classes=50
elif dataset == 'SpeechCOM_V1_30': num_classes=30
elif dataset == 'SpeechCOM_V2': num_classes=35
elif dataset == 'VoxCeleb1': num_classes=1251
elif dataset == 'GTZAN': num_classes=10
elif dataset == 'CREMAD': num_classes=6
elif dataset == 'IEMOCAP': num_classes=4
elif dataset == 'FMA_Small': num_classes=8
elif dataset == 'FMA_Medium': num_classes=16

# --- Build frontend (SimplifyLeaf) ---
compression_fn = Simplify_PCEN(num_bands=n_filters,
                               s=0.04,
                               alpha=0.48,
                               delta=2.0,
                               r=0.5,
                               eps=1e-12,
                               learn_logs=False,
                               clamp=1e-5)
frontend = Leaf(
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
network = AudioClassifier(
    num_outputs=num_classes,
    frontend=frontend,
    encoder=encoder
)

# --- Load checkpoint ---
ckpt_path = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/new_adaptive_pcen/test_models/complex_SimpLEAF/CREMAD/epoch_149.pth" 
state = torch.load(ckpt_path, map_location="cuda")
network.load_state_dict(state, strict=False)
network.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = network.to(device)
wav_path = "/media/mengh/SharedData/hanyu/Adaptive_PCEN/new_adaptive_pcen/audios/1001_IWW_SAD_XX_babble_1.51.wav"
wav_tensor = load_wav_for_model(wav_path, target_sr=32000)
wav_tensor = wav_tensor.to(device)

# Frame the waveform
frame_len = 400   # 25ms @ 16kHz
hop = 160         # 10ms @ 16kHz
frames = wav_tensor.unfold(-1, frame_len, hop) 
frames = frames.squeeze(1) 

# Inference: get representation after SimpPCEN
frontend.eval()
with torch.no_grad():
    num_frames = frames.shape[1]
    pcen_outputs = []
    pcen_outputs = frontend(wav_tensor)
    pcen_energy_db = 10 * np.log10(pcen_outputs.squeeze(0).cpu().numpy() + 1e-12)

with plt.rc_context({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"]
}):

    plt.figure(figsize=(4, 2.3))
    im = plt.imshow(
        pcen_energy_db, aspect='auto', origin='lower', interpolation='nearest', cmap='inferno'
    )
    plt.xlim(0, 98)
    plt.ylim(0, 39)
    cbar = plt.colorbar(im)
    cbar.set_label('Energy (dB)', fontsize=15)
    cbar.locator = LinearLocator(numticks=3)      
    cbar.formatter = FormatStrFormatter('%.0f')
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel('Frame Index', fontsize=16)
    plt.ylabel('Filter Index', fontsize=16)
    ax = plt.gca()
    ax.set_facecolor("none")
    ax.xaxis.set_major_locator(MultipleLocator(20))  
    ax.xaxis.set_minor_locator(MultipleLocator(10))   
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4)) 
    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1)
    ax.tick_params(axis='both', which='minor', labelsize=12, length=3, width=0.8)
    plt.tight_layout()
    plt.savefig(
        '/media/mengh/SharedData/hanyu/Adaptive_PCEN/new_adaptive_pcen/simplifyleaf_pcen_output_energy_db.png',
        dpi=300,
        transparent=True,
        bbox_inches="tight"
    )
    plt.show()