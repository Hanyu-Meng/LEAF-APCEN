
import os, sys, pdb, glob
import numpy as np
from torch.utils.data import Dataset
import librosa
import torch
from padding import *
import scipy, soundfile
from crema_d import add_noise

def getfile_esc50(base_dir, fold, subset):
    meta_path = os.path.join(base_dir, 'meta/esc50.csv')
    meta = np.loadtxt(meta_path, delimiter=',', dtype='str', skiprows=1)
    train_list = []
    eval_list = []
    for i in range(len(meta)):
        # pdb.set_trace()
        label = int(meta[i][2])
        curfold = int(meta[i][1])
        cur_path = meta[i][0]
        cur_dic = {'wav': base_dir+'/audio/'+cur_path, 'label': label}
        if curfold == fold:
            eval_list.append(cur_dic)
        else:
            train_list.append(cur_dic)
    if subset == 'train':
        return train_list
    if subset == 'test':
        return eval_list

class Dataset_ESC50(Dataset):
    def __init__(self, base_dir, fold, subset, front_end, max_len=16000, dynamic=False, data_type='babble'):
        
        if dynamic == True:
            if data_type in ['babble', 'music']:
                self.base_dir = os.path.join(base_dir, 'esc_50_{}_noise_varying_nature'.format(data_type))       
            elif data_type == 'loudness':
                self.base_dir = os.path.join(base_dir, 'esc_50_dynamic_{}_nature'.format(data_type)) 
            elif data_type == 'complex':
                self.base_dir = os.path.join(base_dir, 'esc_50_{}'.format(data_type)) 
        else:
            self.base_dir = base_dir
        self.fold = fold
        self.max_len = max_len
        self.subset = subset
        self.file_list = getfile_esc50(self.base_dir, self.fold, self.subset)
        self.front_end = front_end
        # pdb.set_trace()

    def __len__(self):
        return len(self.file_list)
        # return 200

    def __getitem__(self, index):
        # gain = RandomGain()
        waveform, sr = librosa.load(self.file_list[index]['wav'], sr=16000) # resample 44.1 to 16 KHz
        label = self.file_list[index]['label']
        if self.subset == 'train':
            # randomly select 1-second audio segment or clip
            start = np.random.randint(0, waveform.shape[0] - self.max_len+1)
            waveform = waveform[start:start + self.max_len]
            if self.front_end == 'Ada_FE' or self.front_end == 'Simplify_AdaLeaf':
                waveform = pad(waveform, self.max_len)
                waveform = np.transpose(librosa.util.frame(waveform, frame_length=400, hop_length=160))
            else: 
                waveform = np.expand_dims(waveform, axis=0).copy()
        if self.subset == 'test':
            # waveform = gain(waveform, 16000)
            waveform = np.transpose(librosa.util.frame(waveform, frame_length=self.max_len, hop_length=self.max_len)).copy()
            if self.front_end == 'Ada_FE' or self.front_end == 'Simplify_AdaLeaf':
                waveform = [np.transpose(librosa.util.frame(pad(waveform[i], self.max_len), frame_length=400, hop_length=160)) for i in range(waveform.shape[0])]
                waveform = np.stack(waveform)
        waveform = torch.from_numpy(waveform)
        # return waveform.double(), label
        return waveform.float(), label


def get_file_image2reverb(base_dir):
    rir_list = glob.glob(os.path.join(base_dir, '*/*/*.wav'))
    return rir_list

def get_file_esc50(base_dir):
    esc50_list = glob.glob(os.path.join(base_dir, 'audio/*.wav'))
    return esc50_list

