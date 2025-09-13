import os, sys, pdb, glob
import numpy as np
from torch.utils.data import Dataset
import librosa
import torch
from padding import *
import pandas as pd
# import pdb

fma_small_labels = sorted(['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock'])
fma_medium_labels = sorted(['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic', 'Experimental', 'Folk', 'Hip-Hop', \
'Instrumental', 'International', 'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken'])
BAD_TRACKS = {98567, 98565, 98569, 99041, 99134, 108925, 133297}

def getfile_fma(base_dir, subset, split, dynamic=False, data_type='babble', must_exist=False):
    """
    subset: {'small','medium'}
    split : {'training','validation','test'}
    dynamic/data_type: choose your dynamic subfolders
    must_exist: if True, only keep files that actually exist on disk
    """
    assert subset in {'small', 'medium'}, f"subset should be small/medium, got {subset}"
    assert split in {'training', 'validation', 'test'}, f"split should be training/validation/test, got {split}"

    meta_path = os.path.join(base_dir, 'fma_metadata', 'tracks.csv')
    meta = pd.read_csv(meta_path, index_col=0, header=[0, 1])

    keep_cols = [('set', 'subset'), ('set', 'split'), ('track', 'genre_top')]
    meta = meta.loc[:, keep_cols].copy()

    total_rows = len(meta)
    meta = meta.loc[meta[('set', 'subset')].eq(subset)].copy()
    after_subset = len(meta)

    meta = meta.loc[meta[('set', 'split')].eq(split)].copy()
    after_split = len(meta)

    meta = meta.reset_index().rename(columns={'index': 'track_id'})
    meta['track_id'] = pd.to_numeric(meta['track_id'], errors='coerce').astype('Int64')
    meta = meta.dropna(subset=['track_id']).copy()
    meta['track_id'] = meta['track_id'].astype(int)

    if meta.empty:
        print(f"[WARN] No rows after filtering. "
              f"total={total_rows}, after_subset={after_subset}, after_split={after_split}. "
              f"Check your subset/split values.")
        return []

    file_list = []
    for _, row in meta.iterrows():
        track_id = int(row['track_id'])
        if track_id in BAD_TRACKS:
            continue

        label = str(row[('track', 'genre_top')])
        track = f"{track_id:06d}"

        if dynamic:
            if data_type in ['babble', 'music']:
                file_path = os.path.join(base_dir, f"fma_{subset}_{data_type}_noise_varying_nature", f"{track}.wav")
            elif data_type == 'loudness':
                file_path = os.path.join(base_dir, f"fma_{subset}_dynamic_{data_type}_nature", f"{track}.wav")
            else:
                raise ValueError(f"Unknown data_type for dynamic=True: {data_type}")
        else:
            file_path = os.path.join(base_dir, f"fma_{subset}", f"{track}.wav")

        if not must_exist or os.path.isfile(file_path):
            file_list.append({'wav': file_path, 'label': label})

    if not file_list:
        print("[WARN] File list is empty after building paths. "
              "If you want to keep only existing files, set must_exist=True to diagnose missing paths.")
    else:
        print(f"[INFO] Built {len(file_list)} files for subset={subset}, split={split}")

    return file_list

def getfile_fma(base_dir, subset, split, dynamic, data_type):
    file_list = []
    assert subset in {'small', 'medium'}
    meta_path = os.path.join(base_dir, 'fma_metadata/tracks.csv')
    meta = pd.read_csv(meta_path, index_col=0, header=[0,1])
    keep_colums = [('set', 'subset'), ('set', 'split'), ('track', 'genre_top')]
    meta = meta[keep_colums]
    meta = meta[meta[('set', 'subset')]==subset] # subset {'small', 'medium'}
    meta['track_id'] = meta.index
    meta = meta[meta[('set', 'split')]==split] # spit {'training', 'test', 'validation'}
    for index, row in meta.iterrows():
        track_id = int(row['track_id'].iloc[0])
        if track_id in [98567, 98565, 98569, 99041, 99134, 108925, 133297]: # bad cases
            continue
        label = str(row[('track', 'genre_top')])
        track = '{:06d}'.format(track_id)
        if dynamic == True:
            if data_type in ['babble', 'music']:
                file_path = os.path.join(base_dir, 'fma_{}_{}_noise_varying_nature'.format(subset, data_type), track+'.wav')       
            elif data_type == 'loudness':
                file_path = os.path.join(base_dir, 'fma_{}_dynamic_{}_nature'.format(subset, data_type), track+'.wav') 
            elif data_type == 'complex':
                file_path = os.path.join(base_dir, 'fma_{}_{}'.format(subset, data_type), track+'.wav') 
        else:
            file_path = os.path.join(base_dir, 'fma_{}'.format(subset), track+'.wav')      

        file_list.append({'wav': file_path, 'label': label})

    return file_list
# trainset = getfile_fma('/root/2T/DataSets/FMA', 'medium', 'training')
# pdb.set_trace()      
class Dataset_FMA(Dataset):
    def __init__(self, base_dir, subset, split, front_end, max_len=16000, dynamic=False, data_type='babble'): 
        if dynamic == True:
            if data_type in ['babble', 'music']:
                self.base_dir = os.path.join(base_dir, 'fma_small_{}_noise_varying_nature'.format(data_type))       
            elif data_type == 'loudness':
                self.base_dir = os.path.join(base_dir, 'fma_small_dynamic_{}_nature'.format(data_type))  
        else:
            self.base_dir = base_dir

        self.max_len = max_len
        self.subset = subset
        self.split = split
        self.file_list = getfile_fma(base_dir, self.subset, self.split, dynamic, data_type)
        self.front_end = front_end
        if self.subset == 'small':
            self.labels = fma_small_labels
        if self.subset == 'medium':
            self.labels = fma_medium_labels

    def __len__(self):
        return len(self.file_list)
        # return 200

    def __getitem__(self, index):
        waveform, sr = librosa.load(self.file_list[index]['wav'], sr=16000) # resample 44.1 to 16 KHz
        label = self.labels.index(self.file_list[index]['label'])
        if self.split == 'training':
            # randomly select 1-second audio segment or clip
            start = np.random.randint(0, waveform.shape[0] - self.max_len+1)
            waveform = waveform[start:start + self.max_len]
            if self.front_end == 'Ada_FE' or self.front_end == 'Simplify_AdaLeaf':
                waveform = pad(waveform, self.max_len)
                waveform = np.transpose(librosa.util.frame(waveform, frame_length=400, hop_length=160))
            else: 
                waveform = np.expand_dims(waveform, axis=0).copy() 
        if self.split == 'test':
            # waveform = gain(waveform, 16000)
            waveform = np.transpose(librosa.util.frame(waveform, frame_length=self.max_len, hop_length=self.max_len)).copy()
            if self.front_end == 'Ada_FE' or self.front_end == 'Simplify_AdaLeaf':
                waveform = [np.transpose(librosa.util.frame(pad(waveform[i], self.max_len), frame_length=400, hop_length=160)) for i in range(waveform.shape[0])]
                waveform = np.stack(waveform)
        waveform = torch.from_numpy(waveform)
        return waveform.float(), label # change to float for FMA dataset (previous is double)

