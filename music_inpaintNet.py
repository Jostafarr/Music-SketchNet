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

input_dims = 256
pf_hidden_dims = 512
g_h_dims = 1024
pf_num = 2
inpaint_len = 4
seq_len = 16
batch_size = 32
n_epochs = 100

whole_data_path = [
    "data/irish-measure-vae-train-whole.npy",
    "data/irish-measure-vae-validate-whole.npy",
    "data/irish-measure-vae-test-whole.npy"
]
lr = 1e-4
save_period = 5
decay = 0.9999

def train_music_inpaintNet():
    train_set = np.load(whole_data_path[0], allow_pickle=True)
    validate_set = np.load(whole_data_path[1], allow_pickle=True)

    train_x = torch.from_numpy(train_set.item()["data"]).float()
    train_gd = torch.from_numpy(train_set.item()["gd"]).long()
    validate_x = torch.from_numpy(validate_set.item()["data"]).float()
    validate_gd = torch.from_numpy(validate_set.item()["gd"]).long()

    print(train_x.size())
    print(train_gd.size())
    print(validate_x.size())
    print(validate_gd.size())

    # You can see here we only use the "gd" without "data"
    train_set = TensorDataset(train_gd, train_gd)
    validate_set = TensorDataset(validate_gd, validate_gd)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
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
    dic = torch.load("model_backup/measure-vae-param.pt")
    for name in list(dic.keys()):
        dic[name.replace('module.', '')] = dic.pop(name)
    vae_model.load_state_dict(dic)

    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        vae_model.cuda()
    else:
        print('Using: CPU')
    vae_model.eval()

    save_path = "model_backup/"

    model = InpaintingNet(input_dims, pf_hidden_dims, g_h_dims, pf_num, inpaint_len, vae_model, False, 2000, True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('Using: CPU')

    print(model.parameters())
    return
    device = torch.device(torch.cuda.current_device())
    save_period = 2
    losses = []
    step = 0
    n_past = 6
    n_future = 10
    n_inpaint = 4
    iteration = 0
    for epoch in range(n_epochs):
        model.train()
        print("epoch: %d\n__________________________________________" % (epoch), flush=True)
        mean_loss = 0.0
        v_mean_loss = 0.0
        total = 0
        for i, tr_data in enumerate(train_loader):
            model.train()
            j = i % len(validate_x)
            raw_x, raw_gd = tr_data
            past_x = raw_x[:, :n_past, :]
            inpaint_x = raw_x[:, n_past:n_past + n_inpaint, :]
            future_x = raw_x[:, n_future:, :]

            inpaint_gd = raw_gd[:, n_past:n_past + n_inpaint, :]
            inpaint_gd = inpaint_gd.contiguous().view(-1)

            past_x = past_x.to(device=device, non_blocking=True)
            inpaint_x = inpaint_x.to(device=device, non_blocking=True)
            future_x = future_x.to(device=device, non_blocking=True)
            inpaint_gd = inpaint_gd.to(device=device, non_blocking=True)

            # validate
            v_raw_x = validate_x[j]
            v_raw_gd = validate_gd[j]
            v_past_x = v_raw_x[:, :n_past, :]
            v_inpaint_x = v_raw_x[:, n_past:n_past + n_inpaint, :]
            v_future_x = v_raw_x[:, n_future:, :]
            v_inpaint_gd = v_raw_gd[:, n_past:n_past + n_inpaint, :]
            v_inpaint_gd = v_inpaint_gd.contiguous().view(-1)

            v_past_x = v_past_x.to(device=device, non_blocking=True)
            v_inpaint_x = v_inpaint_x.to(device=device, non_blocking=True)
            v_future_x = v_future_x.to(device=device, non_blocking=True)
            v_inpaint_gd = v_inpaint_gd.to(device=device, non_blocking=True)

            optimizer.zero_grad()

            recon_x, iteration = model(past_x, future_x, inpaint_x)
            loss = F.cross_entropy(recon_x.view(-1, recon_x.size(-1)), inpaint_gd, reduction="mean")
            loss.backward()
            optimizer.step()
            total += 1
            mean_loss += loss.item()
            model.eval()
            with torch.no_grad():
                v_recon_x, _ = model(v_past_x, v_future_x, v_inpaint_x)
                v_loss = F.cross_entropy(v_recon_x.view(-1, v_recon_x.size(-1)), v_inpaint_gd, reduction="mean")
                v_mean_loss += v_loss.item()
            print("batch %d loss: %.5f | v_loss: %.5f |  iteration: %d" % (i, loss.item(), v_loss.item(), iteration),
                  flush=True)
        mean_loss /= total
        v_mean_loss /= total
        print("epoch %d loss: %.5f | v_loss: %.5f " % (epoch, mean_loss, v_mean_loss), flush=True)
        losses.append([mean_loss, v_mean_loss])
        if (epoch + 1) % save_period == 0:
            filename = "inpaintNet-" + 'loss_' + str(mean_loss) + "_" + str(epoch + 1) + '_' + str(iteration) + ".pt"
            torch.save(model.cpu().state_dict(), save_path + filename)
            model.cuda()
        np.save("inpaintNet_measure_log.npy", losses)