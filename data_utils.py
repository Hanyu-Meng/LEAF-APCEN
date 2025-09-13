import os, sys, glob, csv, pdb
import importlib
import random
import numpy as np
import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import librosa
from torch.utils.data import Dataset, DataLoader
from data.voxceleb1 import Dataset_Voxceleb1
from data.esc50 import Dataset_ESC50
from data.crema_d import Dataset_CREMAD
from data.fma import Dataset_FMA

def set_random_seed(random_seed, args=None):
    """ set_random_seed(random_seed, args=None)
    Set the random_seed for numpy, python, and cudnn

    input
    -----
      random_seed: integer random seed
      args: argue parser
    """

    # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    # For torch.backends.cudnn.deterministic
    # Note: this default configuration may result in RuntimeError
    # see https://pytorch.org/docs/stable/notes/randomness.html
    if args is None:
        cudnn_deterministic = True
        cudnn_benchmark = False
    else:
        cudnn_deterministic = args.cudnn_deterministic_toggle
        cudnn_benchmark = args.cudnn_benchmark_toggle

        if not cudnn_deterministic:
            print("cudnn_deterministic set to False")
        if cudnn_benchmark:
            print("cudnn_benchmark set to True")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return np.pad(x[:max_len], (149,0), 'constant', constant_values=(0,0))
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    padded_x = np.pad(padded_x, (149,0), 'constant', constant_values=(0,0))
    return padded_x	

def pad_(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    padded_x = np.pad(x, (0, max_len-x_len), 'constant', constant_values=(0, 0))
    return padded_x

def batch_file(path, file_name):
	file_list = []
	for file in glob.glob(os.path.join(path, file_name)):
		file_list.append(file)
	return file_list

def train_data(args):
    if args.dataset == 'VoxCeleb1':
        train_set = Dataset_Voxceleb1('/media/mengh/SharedData/hanyu/Adaptive_PCEN/VoxCeleb1', 'train', front_end=args.frontend, dynamic=args.dynamic, data_type=args.data_type)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    if args.dataset == 'ESC50':
        train_set = Dataset_ESC50('/media/mengh/SharedData/hanyu/Adaptive_PCEN/ESC50', fold=args.fold, subset='train', front_end=args.frontend, dynamic=args.dynamic, data_type=args.data_type)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    if args.dataset == 'CREMAD':
        train_set = Dataset_CREMAD('/media/mengh/SharedData/hanyu/Adaptive_PCEN/crema_d', subset='train', front_end=args.frontend, dynamic=args.dynamic, data_type=args.data_type)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    if args.dataset == 'FMA_Small':
        train_set = Dataset_FMA('/media/mengh/SharedData/hanyu/Adaptive_PCEN/fma', subset='small', split='training', front_end=args.frontend, dynamic=args.dynamic, data_type=args.data_type)
        # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    del train_set
    return train_loader

def val_data(args):
    if args.dataset == 'VoxCeleb1':
        test_set = Dataset_Voxceleb1('/media/mengh/SharedData/hanyu/Adaptive_PCEN/VoxCeleb1', 'test', front_end=args.frontend, dynamic=args.dynamic, data_type=args.data_type)
        dev_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)
        del test_set
    if args.dataset == 'ESC50':
        test_set = Dataset_ESC50('/media/mengh/SharedData/hanyu/Adaptive_PCEN/ESC50', fold=args.fold, subset='test', front_end=args.frontend, dynamic=args.dynamic, data_type=args.data_type)
        dev_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)
    if args.dataset == 'CREMAD':
        test_set = Dataset_CREMAD('/media/mengh/SharedData/hanyu/Adaptive_PCEN/crema_d', subset='test', front_end=args.frontend, dynamic=args.dynamic, data_type=args.data_type)
        dev_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)
    if args.dataset == 'FMA_Small':
        test_set = Dataset_FMA('/media/mengh/SharedData/hanyu/Adaptive_PCEN/fma', subset='small', split='test', front_end=args.frontend, dynamic=args.dynamic, data_type=args.data_type)
        dev_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)
        del test_set
    return dev_loader
     
                
                



