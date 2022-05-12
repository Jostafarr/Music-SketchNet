import torch
import os
import numpy as np
from torch import optim
from torch.nn import functional as F
from InpaintRNN.inpaintrnn_dis import InpaintingNet
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.distributions import Normal
from SketchVAE.sketchvae import SketchVAE
from utils.helpers import *
import time

s_dir = ""
input_dims = 256
pf_hidden_dims = 512
g_h_dims = 1024
pf_num = 2
inpaint_len = 4
seq_len = 16
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

def processed_data_tensor(data):
    print("processed data:")
    gd = []
    px = []
    rx = []
    len_x = []
    nrx = []
    total = 0
    for i, d in enumerate(data):
        gd.append([list(dd[0]) for dd in d])
        px.append([list(dd[1]) for dd in d])
        rx.append([list(dd[2]) for dd in d])
        len_x.append([dd[3] for dd in d])
        if len(gd[-1][-1]) != vae_seq_len:
            gd[-1][-1].extend([128] * (vae_seq_len - len(gd[-1][-1])))
            px[-1][-1].extend([128] * (vae_seq_len - len(px[-1][-1])))
            rx[-1][-1].extend([2] * (vae_seq_len - len(rx[-1][-1])))
    for i,d in enumerate(len_x):
        for j,dd in enumerate(d):
            if len_x[i][j] == 0:
                gd[i][j][0] = 60
                px[i][j][0] = 60
                rx[i][j][0] = 1
                len_x[i][j] = 1
                total += 1
    gd = np.array(gd)
    px = np.array(px)
    rx = np.array(rx)
    len_x = np.array(len_x)
    for d in rx:
        nnrx = []
        for dd in d:
            temp = np.zeros((vae_seq_len, vae_rhythm_dims))
            lins = np.arange(0, len(dd))
            temp[lins, dd - 1] = 1
            nnrx.append(temp)
        nrx.append(nnrx)
    nrx = np.array(nrx)
    gd = torch.from_numpy(gd).long()
    px = torch.from_numpy(px).long()
    rx = torch.from_numpy(rx).float()
    len_x = torch.from_numpy(len_x).long()
    nrx = torch.from_numpy(nrx).float()
    print("processed finish! zeros:", total)
    print(gd.size(),px.size(),rx.size(),len_x.size(),nrx.size())
    return TensorDataset(px, rx, len_x, nrx, gd)

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
def get_acc(recon, gd):
    recon = recon.cpu().detach().numpy()
    gd = gd.cpu().detach().numpy()
    return np.sum(recon == gd) / recon.size

def evaluata_SketchVAE_InpaintRNN():
    validate_set = np.load(data_path[3], allow_pickle=True)
    validate_loader = DataLoader(
        dataset=processed_data_tensor(validate_set),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    validate_data = []
    for i, d in enumerate(validate_loader):
        validate_data.append(d)
    print(len(validate_data))

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
    save_path = s_dir + "model_backup/"
    save_period = 5

    model = InpaintingNet(input_dims, pf_hidden_dims, g_h_dims, pf_num, inpaint_len, vae_model, False, 2000, True)
    dic = torch.load("model_backup/sketchvae-inpaintrnn-param.pt")
    for name in list(dic.keys()):
        dic[name.replace('module.', '')] = dic.pop(name)
    model.load_state_dict(dic)

    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('Using: CPU')

    model.eval()

    device = torch.device(torch.cuda.current_device())
    save_period = 2
    losses = []
    step = 0
    n_past = 6
    n_future = 10
    n_inpaint = 4
    iteration = 0
    model.eval()
    v_mean_loss = 0.0
    v_mean_acc = 0.0
    total = 0
    output = []
    # save_period = 200
    for i in range(len(validate_data)):
        v_raw_x = process_raw_x(validate_data[i], n_past, n_inpaint, n_future)
        for k in range(len(v_raw_x)):
            v_raw_x[k] = v_raw_x[k].to(device=device, non_blocking=True)
        v_past_px, v_past_rx, v_past_len_x, v_past_nrx, v_past_gd, \
        v_inpaint_px, v_inpaint_rx, v_inpaint_len_x, v_inpaint_nrx, v_inpaint_gd, \
        v_future_px, v_future_rx, v_future_len_x, v_future_nrx, v_future_gd = v_raw_x
        v_inpaint_gd_whole = v_inpaint_gd.contiguous().view(-1)
        v_past_x = [v_past_px, v_past_rx, v_past_len_x, v_past_nrx, v_past_gd]
        v_inpaint_x = [v_inpaint_px, v_inpaint_rx, v_inpaint_len_x, v_inpaint_nrx, v_inpaint_gd]
        v_future_x = [v_future_px, v_future_rx, v_future_len_x, v_future_nrx, v_future_gd]

        model.eval()
        with torch.no_grad():
            v_recon_x, _, _ = model(v_past_x, v_future_x, v_inpaint_x)
            v_loss = F.cross_entropy(v_recon_x.view(-1, v_recon_x.size(-1)), v_inpaint_gd_whole, reduction="mean")
            v_acc = get_acc(v_recon_x.view(-1, v_recon_x.size(-1)).argmax(-1), v_inpaint_gd_whole)
            v_mean_loss += v_loss.item()
            v_mean_acc += v_acc
            v_result = v_recon_x.argmax(-1)
        #         print("validate finished", time.process_time())
        total += 1
        output.append(
            {
                "past": v_past_gd.cpu().detach().numpy(),
                "future": v_future_gd.cpu().detach().numpy(),
                "inpaint": v_result.cpu().detach().numpy(),
                "gd": v_inpaint_gd.cpu().detach().numpy(),
                "acc": v_acc,
                "nll": v_loss.item()
            }
        )
        print(i)

    np.save("nonre-res-test-irish-inpaintNet-dismeasure-ce-1-loss_0.20945583218566277_200_248600.npy", output)
    # np.save("irish-inpaint-ce-generate-train.npy", i_out)