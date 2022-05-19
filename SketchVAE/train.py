import torch
import os
import numpy as np
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from SketchVAE.sketchvae import SketchVAE
from torch.utils.data import Dataset, DataLoader, TensorDataset

class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N

def loss_function(recon, target, p_dis, r_dis, beta):
    CE = F.cross_entropy(recon.view(-1, recon.size(-1)), target, reduction = "mean")
    normal1 = std_normal(p_dis.mean.size())
    normal2=  std_normal(r_dis.mean.size())
    KLD1 = kl_divergence(p_dis, normal1).mean()
    KLD2 = kl_divergence(r_dis, normal2).mean()
    max_indices = recon.view(-1, recon.size(-1)).max(-1)[-1]
    correct = max_indices == target
    acc = torch.sum(correct.float()) / target.size(0)
    return acc, CE + beta * (KLD1 + KLD2)

s_dir = "" # folder_address
batch_size = 64
n_epochs = 100
# data_path = ["data/irish_train_chord_rhythm.npy",
#              "data/irish_validate_chord_rhythm.npy",
#              "data/irish_test_chord_rhythm.npy"]
save_path = "model_backup" # save_model_address
lr = 1e-4
decay = 0.9999
hidden_dims = 1024
zp_dims = 128
zr_dims = 128
vae_beta = 0.1
input_dims = 130
pitch_dims = 129
rhythm_dims = 3
seq_len = 4 * 6
beat_num = 4
tick_num = 6
# set here to config your save_period (2 i.e. save the model every 2 epochs)
save_period = 2

def processed_data_tensor(data):
    print("processed data:")
    gd = np.array([d[0] for d in data])
    px = np.array([d[1] for d in data])
    rx = np.array([d[2] for d in data])
    len_x = np.array([d[3] for d in data])
    nrx = []
    for i,r in enumerate(rx):
        temp = np.zeros((seq_len, rhythm_dims))
        lins = np.arange(0, len(r))
        temp[lins, r - 1] = 1
        nrx.append(temp)
    nrx = np.array(nrx)
    gd = torch.from_numpy(gd).long()
    px = torch.from_numpy(px).long()
    rx = torch.from_numpy(rx).float()
    len_x = torch.from_numpy(len_x).long()
    nrx = torch.from_numpy(nrx).float()
    print("processed finish!")
    return TensorDataset(px, rx, len_x, nrx, gd)

def train_sketch_vae(datapath):
    train_set = np.load(os.path.join(s_dir, data_path[0]), allow_pickle=True)
    validate_set = np.load(os.path.join(s_dir, data_path[1]), allow_pickle=True)
    train_set = DataLoader(
        dataset=processed_data_tensor(train_set),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    validate_set = DataLoader(
        dataset=processed_data_tensor(validate_set),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    model = SketchVAE(input_dims, pitch_dims, rhythm_dims, hidden_dims, zp_dims, zr_dims, seq_len, beat_num, tick_num,
                      4000)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if decay > 0:
        scheduler = MinExponentialLR(optimizer, gamma=decay, minimum=1e-5)
    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('Using: CPU')

    validate_data = []
    for i, d in enumerate(validate_set):
        validate_data.append(d)
    print(len(validate_data))

    logs = []
    device = torch.device(torch.cuda.current_device())
    iteration = 0
    step = 0
    for epoch in range(n_epochs):
        print("epoch: %d\n__________________________________________" % (epoch), flush=True)
        mean_loss = 0.0
        mean_acc = 0.0
        v_mean_loss = 0.0
        v_mean_acc = 0.0
        total = 0
        for i, d in (enumerate(train_set)):
            # validate display
            model.train()
            j = i % len(validate_data)
            px, rx, len_x, nrx, gd = d
            v_px, v_rx, v_len_x, v_nrx, v_gd = validate_data[j]

            px = px.to(device = device,non_blocking = True)
            len_x = len_x.to(device = device,non_blocking = True)
            nrx = nrx.to(device = device,non_blocking = True)
            gd = gd.to(device = device,non_blocking = True)

            v_px = v_px.to(device = device,non_blocking = True)
            v_len_x = v_len_x.to(device = device,non_blocking = True)
            v_nrx = v_nrx.to(device = device,non_blocking = True)
            v_gd = v_gd.to(device = device,non_blocking = True)

            optimizer.zero_grad()
            recon, p_dis, r_dis, iteration = model(px, nrx, len_x, gd)
            acc, loss = loss_function(recon, gd.view(-1), p_dis, r_dis, vae_beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            mean_loss += loss.item()
            mean_acc += acc.item()

            model.eval()
            with torch.no_grad():
                v_recon, v_p_dis, v_r_dis, _ = model(v_px, v_nrx, v_len_x, v_gd)
                v_acc, v_loss = loss_function(v_recon, v_gd.view(-1), v_p_dis, v_r_dis, vae_beta)
                v_mean_loss += v_loss.item()
                v_mean_acc += v_acc.item()
            step += 1
            total += 1
            if decay > 0:
                scheduler.step()
        mean_loss /= total
        mean_acc /= total
        v_mean_loss /= total
        v_mean_acc /= total
        print("epoch %d loss: %.5f acc: %.5f | val loss %.5f acc: %.5f iteration: %d"
              % (epoch, mean_loss, mean_acc, v_mean_loss, v_mean_acc, iteration), flush=True)
        logs.append([mean_loss, mean_acc, v_mean_loss, v_mean_acc, iteration])
        if (epoch + 1) % save_period == 0:
            filename = "sketchvae-" + 'loss_' + str(v_mean_loss) + "_acc_" + str(v_mean_acc) + "_epoch_" + str(
                epoch + 1) + "_it_" + str(iteration) + ".pt"
            torch.save(model.cpu().state_dict(), os.path.join(s_dir, save_path, filename))
            model.cuda()
        np.save(os.path.join(s_dir, "sketchvae-log.npy"), logs)
