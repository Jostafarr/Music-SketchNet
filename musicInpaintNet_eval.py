import torch
import os
import numpy as np
from torch import optim
from torch.nn import functional as F
from InpaintRNN.inpaintrnn import InpaintingNet
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.distributions import Normal
from MeasureVAE.measure_vae import MeasureVAE
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
whole_data_path = [
    s_dir + "data/irish-dis-measure-vae-train-whole.npy",
    s_dir + "data/irish-dis-measure-vae-validate-whole.npy",
    s_dir + "data/irish-dis-measure-vae-validate-repetition.npy",
    s_dir + "data/irish-dis-measure-vae-validate-non-repetition.npy"
]
lr = 1e-4
decay = 0.9999

def generate_batch(index_file):
    zs = []
    for d in index_file:
        idx,sta,end = d
        zs.append([train_x[idx,0][sta:end + 1], train_x[idx,1][sta:end + 1]])
    return np.array(zs)

def processed_data_tensor(data):
    print("processed data:")
    gd = []
    len_x = []
    total = 0
    for i, d in enumerate(data):
        gd.append([list(dd[0]) for dd in d])
        len_x.append([dd[3] for dd in d])
        if len(gd[-1][-1]) != 24:
            gd[-1][-1].extend([128] * (24 - len(gd[-1][-1])))
    for i,d in enumerate(len_x):
        for j,dd in enumerate(d):
            if len_x[i][j] == 0:
                gd[i][j][0] = 60
                len_x[i][j] = 1
                total += 1
    gd = np.array(gd)
    len_x = np.array(len_x)
    gd = torch.from_numpy(gd).long()
    return gd

def evaluate_musicInpaintNet():
    validate_set = np.load(whole_data_path[3], allow_pickle=True)
    validate_gd = processed_data_tensor(validate_set)
    validate_set = TensorDataset(validate_gd, validate_gd)

    validate_loader = DataLoader(dataset=validate_set, batch_size=batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    validate_x = []
    validate_gd = []
    for x, y in validate_loader:
        validate_x.append(x)
        validate_gd.append(y)

    validate_loader = []

    vae_num_notes = 130
    vae_note_embedding_dim = 10
    vae_metadata_embedding_dim = 2
    vae_num_encoder_layers = 2
    vae_encoder_hidden_size = 512
    vae_encoder_dropout_prob = 0.5
    vae_has_metadata = False
    vae_latent_space_dim = 256
    vae_num_decoder_layers = 2
    vae_decoder_hidden_size = 512
    vae_decoder_dropout_prob = 0.5
    vae_batch_size = 256
    vae_num_epochs = 30
    vae_train = False
    vae_plot = False
    vae_log = True
    vae_lr = 1e-4
    vae_seq_len = 6 * 4

    vae_model = MeasureVAE(
        num_notes=vae_num_notes,
        note_embedding_dim=vae_note_embedding_dim,
        metadata_embedding_dim=vae_metadata_embedding_dim,
        num_encoder_layers=vae_num_encoder_layers,
        encoder_hidden_size=vae_encoder_hidden_size,
        encoder_dropout_prob=vae_encoder_dropout_prob,
        latent_space_dim=vae_latent_space_dim,
        num_decoder_layers=vae_num_decoder_layers,
        decoder_hidden_size=vae_decoder_hidden_size,
        decoder_dropout_prob=vae_decoder_dropout_prob,
        has_metadata=vae_has_metadata
    )
    dic = torch.load(s_dir + "model_backup/measure-vae-param.pt")
    for name in list(dic.keys()):
        dic[name.replace('module.', '')] = dic.pop(name)
    vae_model.load_state_dict(dic)

    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        vae_model.cuda()
    else:
        print('Using: CPU')
    vae_model.eval()

    model = InpaintingNet(input_dims, pf_hidden_dims, g_h_dims, pf_num, inpaint_len, vae_model, False, 2000, True)

    dic = torch.load("model_backup/musicinpaintnet-param.pt")
    for name in list(dic.keys()):
        dic[name.replace('module.', '')] = dic.pop(name)
    model.load_state_dict(dic)

    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('Using: CPU')
    model.eval()

    model.eval()
    device = torch.device(torch.cuda.current_device())
    n_past = 6
    n_future = 10
    n_inpaint = 4
    iteration = 0
    # save_period = 200
    output = []
    for i in range(len(validate_x)):
        v_mean_loss = 0.0
        v_acc = 0.0
        v_raw_x = validate_x[i]
        v_raw_gd = validate_gd[i]

        v_past_x = v_raw_x[:, :n_past, :]
        v_inpaint_x = v_raw_x[:, n_past:n_past + n_inpaint, :]
        v_future_x = v_raw_x[:, n_future:, :]
        v_inpaint_gd = v_raw_gd[:, n_past:n_past + n_inpaint, :]
        v_inpaint_gd = v_inpaint_gd.contiguous().view(-1)

        v_past_x = v_past_x.to(device=device, non_blocking=True)
        v_inpaint_x = v_inpaint_x.to(device=device, non_blocking=True)
        v_future_x = v_future_x.to(device=device, non_blocking=True)
        v_inpaint_gd = v_inpaint_gd.to(device=device, non_blocking=True)

        model.eval()
        with torch.no_grad():
            v_recon_x, _ = model(v_past_x, v_future_x, v_inpaint_x)
            v_loss = F.cross_entropy(v_recon_x.view(-1, v_recon_x.size(-1)), v_inpaint_gd, reduction="mean")
            v_mean_loss = v_loss.item()
            v_recon_x_note = v_recon_x.max(-1)[-1]
            correct = v_recon_x_note == v_inpaint_x
            v_acc = torch.sum(correct.float()) / (v_recon_x.view(-1, v_recon_x.size(-1)).size(0))
            v_acc = v_acc.item()
        output.append(
            {
                "past": v_past_x.cpu().detach().numpy(),
                "future": v_future_x.cpu().detach().numpy(),
                "inpaint": v_recon_x_note.cpu().detach().numpy(),
                "gd": v_inpaint_x.cpu().detach().numpy(),
                "acc": v_acc,
                "nll": v_mean_loss
            }
        )
        #print(i)

    # np.save("irish-inpaint-ce-generate-train.npy", i_out)
    np.save("nonre-res-test-irish-musicinpaintnet.npy", output)