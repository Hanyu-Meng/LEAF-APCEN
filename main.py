"""
Training and testing script for LEAF, Simplified LEAF, Simplify-APCEN, and Fixed LEAF
Author: Hanyu Meng 
Time: May 2025
"""
import argparse
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
import hydra
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm
from data_utils import train_data, val_data, set_random_seed
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet
from model import AudioClassifier, Ada_AudioClassifier
from model.leaf import *
from model.adaleaf import *
from utils import optimizer_to, scheduler_to
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='hydra')
# from args import get_args
# Default LEAF parameters
n_filters = 40
sample_rate = 16000
window_len = 25.0
window_stride = 10.0
min_freq = 60.0
max_freq = 7800.0


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def set_frozen_layers(network, args):
    if args.frontend_lr_factor == 1:
            params = network.parameters()
            if args.fixed_filter:
                for p in network._frontend.filterbank.parameters(): p.requires_grad=False
            if args.fixed_PCEN:
                for name, p in network._frontend.compression.named_parameters():
                    if name in ('alpha','s','r'): p.requires_grad=False
            if args.fixed_all_PCEN:
                for p in network._frontend.compression.parameters(): p.requires_grad=False
            if args.fixed_backend:
                for p in network._encoder.parameters(): p.requires_grad=False
    else:
        frontend_params = list(network._frontend.parameters())
        frontend_ids = set(id(p) for p in frontend_params)
        params = [p for p in network.parameters() if id(p) not in frontend_ids]
        
    total, trainable = count_parameters(network)
    print(f"✅ Total parameters: {total:,}")
    print(f"✅ Trainable parameters: {trainable:,}")

    return network

    # if not args.resume or not os.path.exists(args.resume):
    #     criterion, optimizer, scheduler = criterion_and_optimizer(args, network)

    # ## load previous run
    # if args.resume and not os.path.exists(args.resume):
    #     print("resume file %s does not exist; ignoring" % args.resume)

    # if args.resume and os.path.exists(args.resume):
    #     saved_dict = torch.load(args.resume, map_location=torch.device('cpu'))
    #     network.load_state_dict(saved_dict['network'])
    #     criterion, optimizer, scheduler = criterion_and_optimizer(args, network)
    #     args.start_epoch = 1
    #     optimizer.load_state_dict(saved_dict['optimizer'])
    #     if args.scheduler is not None and saved_dict['scheduler'] is not None:
    #         scheduler.load_state_dict(saved_dict['scheduler'])
    #     del saved_dict

    # ## move network and optim to device
    # network.to(device)
    # torch.cuda.empty_cache()
    # optimizer_to(optimizer, device)
    # scheduler_to(scheduler, device)

    # # return network if wanted
    # if args.ret_network:
    #     if args.data_set != "None":
    #         return network, train_loader, val_loader, test_loader
    #     else:
    #         return network

def infer(args, network, device):
    num_correct = 0.0
    num_correct_top5 = 0.0
    num_total = 0.0
    dev_loader = val_data(args)
    if args.dataset in ['ESC50', 'IEMOCAP']:
        model_tag = '{}_fold{}_{}_model_{}_{}_{}_Fixed_{}_FC_CrossEn_Seg1S_{}'.format(args.dataset, args.fold, args.frontend, args.batch_size, args.lr, args.decay, args.back_end, args.data_type)
    else: model_tag = '{}_{}_model_{}_{}_{}_original_0.9_{}_FC_CrossEn_Seg1S_{}'.format(args.dataset, args.frontend, args.batch_size, args.lr, args.decay, args.back_end, args.data_type)
    model_path = os.path.join(args.model_path, model_tag) + '/epoch_{}.pth'.format(args.epoch)
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()
    # pdb.set_trace()
    for batch_x, batch_y in tqdm(dev_loader):
        if args.frontend == 'Ada_FE': batch_x = batch_x.squeeze(0)
        batch_size = batch_x.size(0)
        num_frames = batch_x.size(1)
        # num_total += batch_size
        # num_total += 1
        if 'SpeechCOM' in args.dataset: num_total += batch_size # SpeechCOM
        elif args.dataset in ['GTZAN', 'VoxCeleb1', 'ESC50', 'IEMOCAP', 'CREMAD', 'FMA_Small', 'FMA_Medium']: num_total +=1 # GTZAN, VoCeleb1, ESC50
        batch_x = Variable(batch_x.to(device).contiguous())
        batch_y = Variable(batch_y.view(-1).to(device).contiguous())
        batch_out = network(batch_x) # SpeechCOM for Leaf, tdfbanks and SincNet
        if args.dataset in ['GTZAN', 'VoxCeleb1', 'ESC50', 'IEMOCAP', 'CREMAD', 'FMA_Small', 'FMA_Medium']: 
            batch_out = torch.mean(batch_out, dim=0, keepdim=True) 
        _, batch_pred = batch_out.data.max(dim=1)
        if args.dataset == 'IEMOCAP': k=2
        else: k=5
        _, batch_pred_top5 = batch_out.data.topk(k, dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        num_correct_top5 += (batch_pred_top5 == batch_y.view(-1,1)).sum().item()
    print(100 * (num_correct / num_total))
    print(100 * (num_correct_top5 / num_total))
    return 100 * (num_correct / num_total)

def evaluate_accuracy(dev_loader, network, device, args):
    num_correct = 0.0
    num_correct_top5 = 0.0
    num_total = 0.0
    network.eval() 
    # pdb.set_trace()
    for batch_x, batch_y in tqdm(dev_loader):
        if args.frontend == 'Ada_FE': batch_x = batch_x.squeeze(0)
        batch_size = batch_x.size(0)
        num_frames = batch_x.size(1)
        if 'SpeechCOM' in args.dataset: nsum_total += batch_size # SpeechCOM
        elif args.dataset in ['GTZAN', 'VoxCeleb1', 'ESC50', 'IEMOCAP', 'CREMAD', 'FMA_Small', 'FMA_Medium']: num_total +=1 # GTZAN, VoCeleb1, ESC50
        
        batch_x = Variable(batch_x.to(device).contiguous())
        batch_y = Variable(batch_y.view(-1).to(device).contiguous())
        if args.frontend == 'Simplify_AdaLeaf' and args.dataset == 'VoxCeleb1':
            batch_x = batch_x
        else:
            batch_x = batch_x.squeeze(0)

        if 'SpeechCOM' in args.dataset:
            batch_out = network(batch_x)
        else:
            batch_out = network(batch_x) if args.frontend == 'Simplify_AdaLeaf' else network(batch_x.permute(1,0,2)) # SpeechCOM for Leaf, tdfbanks and SincNet
        # ----------- average logits -------------- #
        # GTZAN, Voxceleb1, ESC50, IEMOCAP, CREMAD
        if args.dataset in ['GTZAN', 'VoxCeleb1', 'ESC50', 'IEMOCAP', 'CREMAD', \
        'FMA_Small', 'FMA_Medium']: batch_out = torch.mean(batch_out, dim=0, keepdim=True) 
        _, batch_pred = batch_out.data.max(dim=1)
        if args.dataset == 'IEMOCAP': k=2
        else: k=5
        _, batch_pred_top5 = batch_out.data.topk(k, dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        num_correct_top5 += (batch_pred_top5 == batch_y.view(-1,1)).sum().item()
    return 100 * (num_correct/num_total), 100 * (num_correct_top5/num_total)

def train_epoch(args, train_loader, network, lr, optim, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    i = 0
    network.train() # front-end
    # ------ set Loss functions ----- #
    criterion = nn.CrossEntropyLoss()
    for batch_x, batch_y in tqdm(train_loader):
        batch_size = batch_x.size(0) # batch_x --> [B,nframe,frame_len]
        num_frames = batch_x.size(1)
        wlen = batch_x.size(2) 
        filtered = torch.zeros((batch_size, n_filters, num_frames*batch_x.size(2))).to(device)
        num_total += batch_size
        i += 1
        batch_x = Variable(batch_x.to(device).contiguous()) # no need for new version of pytorch
        # batch_y --> [B]--> for label
        batch_y = Variable(batch_y.view(-1).to(device).contiguous())
        batch_out = network(batch_x) if args.frontend == 'Simplify_AdaLeaf' else network(batch_x) # Leaf, SincNet, tdfbanks, AdaLeaf, SimplifyLeaf
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if i % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format((num_correct / num_total) * 100))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    # print(train_accuracy)
    return running_loss, train_accuracy

def train(args, network):
    device = torch.device(args.device)
    # model path
    if not os.path.exists(args.model_path): os.mkdir(args.model_path)
    model_path = os.path.join(args.model_path, args.frontend)
    if not os.path.exists(model_path): os.mkdir(model_path)
    # make experiment reproducible
    set_random_seed(args.seed, args)
    # define model saving path by the dataset
    if args.dataset in ['IEMOCAP', 'ESC50']:
        model_tag = '{}_fold{}_{}_model_{}_{}_{}_original_{}_FC_CrossEn_Seg1S_{}'.format(args.dataset, args.fold, args.frontend, args.batch_size, args.lr, str(args.decay), args.back_end, args.data_type)
    else: model_tag = '{}_{}_model_{}_{}_{}_original_{}_FC_CrossEn_Seg1S_{}'.format(args.dataset, args.frontend, args.batch_size, args.lr, str(args.decay), args.back_end, args.data_type)
    # ------------- Ablation Studies ---------- #
    # if args.dataset in ['IEMOCAP', 'ESC50']:
    #     model_tag = '{}_fold{}_{}_model_{}_{}_{}_original_0.9_{}_FC_CrossEn_Seg1S'.format(args.dataset, args.fold, args.front_end, args.batch_size, args.lr, str(args.decay), args.back_end)
    # else: model_tag = '{}_{}_model_{}_{}_{}_original_0.9_{}_FC_CrossEn_Seg1S'.format(args.dataset, args.front_end, args.batch_size, args.lr, str(args.decay), args.back_end)

    if args.comment: model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join(model_path, model_tag)
    if not os.path.exists(model_save_path): os.mkdir(model_save_path)
    print(model_tag)
    if args.cont:
        epoch_start = args.epoch
        model_path = model_save_path + '/epoch_{}.pth'.format(str(args.epoch))
        network.load_state_dict(torch.load(model_path, map_location=device))
    else: epoch_start = 0

    with open(os.path.join(model_save_path, 'args.txt'), 'w') as f:
        f.writelines('%s=%s\n' % (k, v) for k, v in vars(args).items())

     # ------------- Save Log Files ---------- #
    if not os.path.exists(args.log_path): os.mkdir(args.log_path)
    with open(args.log_path + '/' + model_tag + ".csv", "a") as results:
        results.write("'"'Validation Acc'"','"'Val_Top5 Acc'"','"'Training Acc'"', '"'Epoch'"', '"'D/T'"'\n")
    # set model save directory
    if not os.path.exists(model_save_path): os.mkdir(model_save_path)
    # set Adam optimizer --> decay learning rate or not
    if args.decay:
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if not args.decay:
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    
    # load the dataset for train and validation
    train_loader = train_data(args)
    dev_loader = val_data(args)

    # Training and validation
    best_acc = 35

    # set frozen layers
    if args.frontend == 'Fixed_Leaf' or args.frontend == 'Fixed_SimplifyLeaf':
        args.fixed_filter = True
        args.fixed_all_PCEN = True

    network = set_frozen_layers(network, args)

    for epoch in range(epoch_start, args.num_epochs):
        running_loss, train_accuracy = train_epoch(args, train_loader, network, args.lr, optimizer, device)
        valid_accuracy, valid_top5 = evaluate_accuracy(dev_loader, network, device, args)
        with open(args.log_path + '/' + model_tag + ".csv", "a") as results:
            results.write("%g, %g, %g, %d, %s\n" % (valid_accuracy, valid_top5, train_accuracy, epoch+1, datetime.now().strftime('%Y-%m-%d/%H:%M:%S')))
        print('\n Epoch {} - loss {} - train_acc {:.2f} - valid_acc {:.2f} - ver {}'.format(epoch+1, running_loss, train_accuracy, valid_accuracy, model_tag))
        if valid_accuracy > best_acc:
            print('best model find at epoch', epoch+1)
            torch.save(network.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch+1)))
        best_acc = max(valid_accuracy, best_acc)

@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(args):
    device = torch.device(args.device)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cudnn.benchmark = args.cudnn_benchmark

    # -------- audio calssification task -------- #
    if args.dataset == 'ESC50': num_classes=50
    elif args.dataset == 'SpeechCOM_V1_30': num_classes=30
    elif args.dataset == 'SpeechCOM_V2': num_classes=35
    elif args.dataset == 'VoxCeleb1': num_classes=1251
    elif args.dataset == 'GTZAN': num_classes=10
    elif args.dataset == 'CREMAD': num_classes=6
    elif args.dataset == 'IEMOCAP': num_classes=4
    elif args.dataset == 'FMA_Small': num_classes=8
    elif args.dataset == 'FMA_Medium': num_classes=16

    # ------ back end ----- #
    # load the encoder --> efficientnet-b0
    frontend_channels = 1
    encoder = EfficientNet.from_name("efficientnet-b0", num_classes=num_classes, include_top=False,
                                     in_channels=frontend_channels)
    encoder._avg_pooling = torch.nn.Identity()

    # ------ front end ------ #
    if args.frontend == 'Leaf' or args.frontend == 'Fixed_Leaf':
        compression_fn = PCEN(num_bands=n_filters,
                        s=0.04,
                        alpha=0.96,
                        delta=2.0,
                        r=0.5,
                        eps=1e-12,
                        learn_logs=args.pcen_learn_logs,
                        clamp=1e-5)
                
        frontend = Leaf(n_filters=n_filters,
                        min_freq=min_freq,
                        max_freq=max_freq,
                        sample_rate=sample_rate,
                        window_len=window_len,
                        window_stride=window_stride,
                        compression=compression_fn)

    elif args.frontend == 'AdaLeaf':
        compression_fn = AdaptivePCEN()     
        frontend = AdaptiveLeaf(n_filters=n_filters,
                        min_freq=min_freq,
                        max_freq=max_freq,
                        sample_rate=sample_rate,
                        window_len=window_len,
                        window_stride=window_stride,
                        compression=compression_fn)
        
    elif args.frontend == 'SimplifyLeaf' or args.frontend == 'Fixed_SimplifyLeaf':
        compression_fn = Simplify_PCEN(num_bands=n_filters,
                s=0.04,
                alpha=0.48,
                delta=2.0,
                r=0.5,
                eps=1e-12,
                learn_logs=args.pcen_learn_logs,
                clamp=1e-5)
        
        frontend = Leaf(n_filters=n_filters,
                        min_freq=min_freq,
                        max_freq=max_freq,
                        sample_rate=sample_rate,
                        window_len=window_len,
                        window_stride=window_stride,
                        compression=compression_fn)
        
    elif args.frontend == 'Simplify_AdaLeaf':
        compression_fn = Simplify_AdaptivePCEN(init_s=0.04,
                    init_alpha=0.48,
                    init_r=0.5,
                    eps=1e-12,
                    clamp=1e-5)
        
        frontend = AdaptiveLeaf(n_filters=n_filters,
                        min_freq=min_freq,
                        max_freq=max_freq,
                        sample_rate=sample_rate,
                        window_len=window_len,
                        window_stride=window_stride,
                        compression=compression_fn)
        
    # ------ combine front-end and backend ------ #
    if args.frontend == 'Simplify_AdaLeaf':
        network = Ada_AudioClassifier(
            num_outputs=num_classes,
            n_filters=n_filters,
            frontend=frontend,
            encoder=encoder).to(device)
    else:
        network = AudioClassifier(
            num_outputs=num_classes,
            frontend=frontend,
            encoder=encoder).to(device)
    # ------ train or inference ------ #
    if args.train: train(args, network)
    if args.eval: infer(args, network)

if __name__ == '__main__':
    main()

