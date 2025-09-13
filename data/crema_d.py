import os, sys, pdb, glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
import torch
from padding import *
import scipy, soundfile
labels_cremad = sorted(['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'])


def getfile_cremad(dir, subset):
    f_list = []
    dictkeys = []
    fname = os.path.join(dir, 'cremad.csv')
    with open(fname, 'r') as f:
        next(f)
        lines = f.readlines()
        for line in lines:
            # file_path = os.path.join(dir, 'AudioWAV', line.strip().split(',')[0])
            file_path = os.path.join(dir,line.strip().split(',')[0])
            file_wav, label, _, split_set = line.strip().split(',')
            if split_set == subset:
                f_list.append({'wav': file_path, 'label': label})
            dictkeys.append(label)
            # pdb.set_trace()
        dictkeys = sorted(set(dictkeys))
        dictkeys = {key: i for i, key in enumerate(dictkeys)}
        return f_list
        

# trainset = getfile_crema_d('/root/2T/DataSets/crema_d/', 'train')

class Dataset_CREMAD(Dataset):
    def __init__(self, base_dir, subset, front_end, max_len=16000, dynamic=False, data_type='babble'):
        if dynamic == True:
            if data_type in ['babble', 'music']:
                self.base_dir = os.path.join(base_dir, 'crema_d_{}_noise_varying_nature'.format(data_type))       
            elif data_type == 'loudness':
                self.base_dir = os.path.join(base_dir, 'crema_d_dynamic_{}_nature'.format(data_type))
            elif data_type == 'complex':  
                self.base_dir = os.path.join(base_dir, 'crema_d_{}'.format(data_type))
        else:
            self.base_dir = base_dir
        self.max_len = max_len
        self.subset = subset
        self.file_list = getfile_cremad(self.base_dir, self.subset)
        self.front_end = front_end
        self.labels = labels_cremad

    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, index):
        # gain = RandomGain()
        waveform, sr = librosa.load(self.file_list[index]['wav'], sr=16000) # resample to 16 KHz
        label = self.labels.index(self.file_list[index]['label'])
        # pdb.set_trace()
        if self.subset == 'train':
            # --- randomly select 1-second audio segment or clip --- #
            start = np.random.randint(0, waveform.shape[0] - self.max_len+1)
            waveform = waveform[start:start + self.max_len]
            if self.front_end == 'Simplify_AdaLeaf':
                waveform = pad(waveform, self.max_len)
                waveform = np.transpose(librosa.util.frame(waveform, frame_length=400, hop_length=160))
            else: 
                waveform = np.expand_dims(waveform, axis=0)
            # else: waveform = np.expand_dims(waveform, axis=0)
        if self.subset == 'test':
            # waveform = gain(waveform, 16000)
            waveform = np.transpose(librosa.util.frame(waveform, frame_length=self.max_len, hop_length=self.max_len))
            if self.front_end == 'Simplify_AdaLeaf':
                waveform = [np.transpose(librosa.util.frame(pad(waveform[i], self.max_len), frame_length=400, hop_length=160)) for i in range(waveform.shape[0])]
                waveform = np.stack(waveform)
        waveform = torch.from_numpy(waveform) # [99,320] --> 99 frames
        return waveform.float(), label


# train_set = Dataset_CREMAD('/root/2T/DataSets/crema_d/', 'test', None)
# label = []
# for A in train_set:
#     label.append(A[1])
# pdb.set_trace()

def get_file_image2reverb(base_dir):
    rir_list = glob.glob(os.path.join(base_dir, '*/*/*.wav'))
    return rir_list

def get_file_cremad(base_dir):
    cremad_list = glob.glob(os.path.join(base_dir, 'AudioWAV/*.wav'))
    return cremad_list


def add_noise(s ,d, snr):
    d = d * np.sqrt(np.mean(s**2)/(np.mean(d**2))) * np.power(10, -0.05*snr)
    x = s + d
    return x

if __name__ == '__main__':
    # ------- reverbration ------ #
    # rir_list = get_file_image2reverb('/g/data3/wa66/qiquan/DataSets/image2reverb')
    # cremad_files = get_file_cremad('/g/data3/wa66/qiquan/DataSets/crema_d')
    # select_rir = np.random.choice(len(rir_list), len(cremad_files), replace=False)
    # pdb.set_trace()
    # rir_shape = []
    # for i in range(len(cremad_files)):
    #     waveform, fs = librosa.load(cremad_files[i], sr=16000)
    #     rir, fs_rir = librosa.load(rir_list[select_rir[i]], sr=16000)
    #     rir_shape.append(len(rir.shape))
    #     if len(rir.shape) == 2:
    #         print(rir.shape)
    #     assert fs == fs_rir
    #     # rir =rir/np.sqrt(np.sum(rir**2))
    #     rir /= np.max(np.abs(rir))
    #     rir[np.where(np.abs(rir)<0.1)] = 0
    #     mix = 0.1 
    #     reverb_wav = waveform + scipy.signal.convolve(waveform, rir, mode='full')[:len(waveform)]*mix
    #     file_split = cremad_files[i].rsplit('/',2)
    #     file_path = os.path.join(file_split[0], 'AudioWAV_reverb', file_split[2])
    #     soundfile.write(file_path, reverb_wav, fs)
    # --------- add noise -------- #
    noise_file = os.path.join('/g/data3/wa66/qiquan/DataSets', 'SIGNAL019-20kHz.wav')
    noise, fs_noise = librosa.load(noise_file, sr=16000)
    cremad_files = get_file_cremad('/g/data3/wa66/qiquan/DataSets/crema_d')
    # snr_list = np.random.randint(5, 15, size=len(cremad_files))
    snr = 10 # {0, 5, 10} dB
    for i in range(len(cremad_files)):
        waveform, fs = librosa.load(cremad_files[i], sr=16000)
        assert fs==fs_noise
        len_audio = waveform.shape[0]
        start = np.random.randint(0, noise.shape[0] - len_audio+1)
        noise_clip = noise[start:start + len_audio]
        # pdb.set_trace()
        # noisy = add_noise(waveform, noise_clip, snr_list[i])
        noisy = add_noise(waveform, noise_clip, snr)
        file_split = cremad_files[i].rsplit('/',2)
        file_path = os.path.join(file_split[0], 'AudioWAV_noisy_{}dB'.format(str(snr)), file_split[2])
        # pdb.set_trace()
        if not os.path.exists(file_path.rsplit('/',1)[0]): os.makedirs(file_path.rsplit('/',1)[0])
        soundfile.write(file_path, noisy, fs)


