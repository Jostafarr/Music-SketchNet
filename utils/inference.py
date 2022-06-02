import pretty_midi as pyd
from loader.dataloader import MIDI_Render
import random
import numpy as np
import torch
import os
from torch import optim
from torch.nn import functional as F
from SketchVAE.sketchvae import SketchVAE
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.distributions import Normal
from SketchNet.sketchnet import SketchNet
from utils.helpers import *
import time

s_dir = ""
zp_dims = 128
zr_dims = 128
pf_dims = 512
gen_dims = 1024
combine_dims = 512
combine_head = 4
combine_num = 4
pf_num = 2
inpaint_len = 4
seq_len = 16
total_len = 16
batch_size = 1
n_epochs = 15
data_path = [
    s_dir + "data/irish-dis-measure-vae-train-whole.npy",
    s_dir + "data/irish-dis-measure-vae-validate-whole.npy",
    s_dir + "data/irish-dis-measure-vae-validate-repetition.npy",
    s_dir + "data/irish-dis-measure-vae-validate-non-repetition.npy"
]
lr = 1e-4
decay = 0.9999
##############################
##############  for vae init ##############
vae_hidden_dims = 1024
vae_zp_dims = 128
vae_zr_dims = 128
vae_beta = 0.1
vae_input_dims = 130
vae_pitch_dims = 129
vae_rhythm_dims = 3
vae_seq_len = 6 * 4
vae_beat_num = 4
vae_tick_num = 6
############################

def process_raw_x(raw_x, n_past, n_inpaint, n_future):
    raw_px, raw_rx, raw_len_x, raw_nrx, raw_gd = raw_x
    past_px = raw_px[:,:n_past,:]
    inpaint_px = raw_px[:,n_past:n_past + n_inpaint,:]
    future_px = raw_px[:,n_future:,:]
    past_rx = raw_rx[:,:n_past,:]
    inpaint_rx = raw_rx[:,n_past:n_past + n_inpaint,:]
    future_rx = raw_rx[:,n_future:,:]
    past_len_x = raw_len_x[:,:n_past]
    inpaint_len_x = raw_len_x[:,n_past:n_past + n_inpaint]
    future_len_x = raw_len_x[:,n_future:]
    past_nrx = raw_nrx[:,:n_past,:]
    inpaint_nrx = raw_nrx[:,n_past:n_past + n_inpaint,:]
    future_nrx = raw_nrx[:,n_future:,:]
    past_gd = raw_gd[:,:n_past,:]
    inpaint_gd = raw_gd[:,n_past:n_past + n_inpaint,:]
    future_gd = raw_gd[:,n_future:,:]
    re = [
        past_px, past_rx, past_len_x, past_nrx, past_gd,
        inpaint_px, inpaint_rx, inpaint_len_x, inpaint_nrx, inpaint_gd,
        future_px, future_rx, future_len_x, future_nrx, future_gd,
    ]
    return re

def process_raw_influence(raw_x, n_past):
    raw_px, raw_rx, raw_len_x, raw_nrx, raw_gd = raw_x
    past_px = raw_px[:,:n_past,:]
    past_rx = raw_rx[:,:n_past,:]
    past_len_x = raw_len_x[:,:n_past]
    past_nrx = raw_nrx[:,:n_past,:]
    past_gd = raw_gd[:,:n_past,:]
    re = [
        past_px, past_rx, past_len_x, past_nrx, past_gd
    ]
    return re

def get_acc(recon, gd):
    recon = recon.cpu().detach().numpy()
    gd = gd.cpu().detach().numpy()
    return np.sum(recon == gd) / recon.size

def gen_sketch(index, px, len_x, nrx, total):
    # pitch_fisrt, rhythm_last
    sketch_cond = []
    for i in index:
        if i < total:
            sketch_cond.append([px[:, i], len_x[:, i]])
        else:
            sketch_cond.append(nrx[:, i - total])
    return index, sketch_cond


def inference(past, future, influence):
        # load VAE model
    vae_model = SketchVAE(
        vae_input_dims, vae_pitch_dims, vae_rhythm_dims, vae_hidden_dims, 
        vae_zp_dims, vae_zr_dims, vae_seq_len, vae_beat_num, vae_tick_num, 4000)
    dic = torch.load("model_backup/sketchvae-param.pt")

    for name in list(dic.keys()):
        dic[name.replace('module.', '')] = dic.pop(name)
    vae_model.load_state_dict(dic)

    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        vae_model.cuda()
    else:
        print('Using: CPU')
    vae_model.eval()

        # import model
    save_path = s_dir +"model_backup/"
    save_period = 5

    # think about traning with mse
    model = SketchNet(
        zp_dims, zr_dims, 
        pf_dims, gen_dims, combine_dims,
        pf_num, combine_num, combine_head,
        inpaint_len, total_len, 
        vae_model, True
    )
    dic = torch.load(save_path + "sketchnet-param.pt")
    for name in list(dic.keys()):
        dic[name.replace('module.', '')] = dic.pop(name)
    model.load_state_dict(dic)
    model.set_stage("sketch")

    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('Using: CPU')
    print(model)
    model.eval()

    model.set_stage("sketch")
    device = torch.device(torch.cuda.current_device())
    save_period = 5
    losses = []
    step = 0
    n_past = 6
    n_future = 10
    n_inpaint = 4
    iteration = 0
    output = []
    v_mean_loss = 0.0
    v_mean_acc = 0.0
    total = 0
    print(vae_model.training)
    
    v_raw_x = process_raw_x(past, n_past, n_inpaint, n_future)
    # choose another song any number you like
    t_raw_x = process_raw_influence(influence, n_past)
    for k in range(len(v_raw_x)):
        v_raw_x[k] = v_raw_x[k].to(device = device,non_blocking = True)

    for k in range(len(t_raw_x)):   
        t_raw_x[k] = t_raw_x[k].to(device = device,non_blocking = True)
    v_past_px, v_past_rx, v_past_len_x, v_past_nrx, v_past_gd, \
    v_inpaint_px, v_inpaint_rx, v_inpaint_len_x, v_inpaint_nrx, v_inpaint_gd,\
    v_future_px, v_future_rx, v_future_len_x, v_future_nrx, v_future_gd = v_raw_x
    v_inpaint_gd_whole = v_inpaint_gd.contiguous().view(-1)
    v_past_x = [v_past_px, v_past_rx, v_past_len_x, v_past_nrx, v_past_gd]
    v_inpaint_x = [v_inpaint_px, v_inpaint_rx, v_inpaint_len_x, v_inpaint_nrx, v_inpaint_gd]
    v_future_x = [v_future_px, v_future_rx, v_future_len_x, v_future_nrx, v_future_gd]
    
    
    t_inpaint_px, t_inpaint_rx, t_inpaint_len_x, t_inpaint_nrx, t_inpaint_gd = t_raw_x
    
    v_sketch_index, v_sketch_cond = gen_sketch([0, 5, 7], t_inpaint_px, t_inpaint_len_x, t_inpaint_nrx, 4)
    
    model.eval()
    with torch.no_grad():
        v_recon_x = model.sketch_generation(v_past_x, v_future_x, v_inpaint_x, v_sketch_index, v_sketch_cond)
        v_result = v_recon_x.argmax(-1)
    total += 1
    output.append(
        {
        "past": v_past_gd.cpu().detach().numpy(),
        "inpaint":v_result.cpu().detach().numpy(),
        "future":v_future_gd.cpu().detach().numpy(),
        "influence":t_inpaint_gd.cpu().detach().numpy(),
        }
    )