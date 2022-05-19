import torch
import os
import numpy as np
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from model_dis import RECVAE
from loader.dataloader import DataLoader

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

def loss_function(recon, recon_rhythm, target, target_rhythm, dis1, dis2, beta):
    CE = F.nll_loss(recon.view(-1, recon.size(-1)), target, reduction = "mean")
    rhy_CE = F.nll_loss(recon_rhythm.view(-1, recon_rhythm.size(-1)), target_rhythm, reduction = "mean")
    normal1 = std_normal(dis1.mean.size())
    normal2=  std_normal(dis2.mean.size())
    KLD1 = kl_divergence(dis1, normal1).mean()
    KLD2 = kl_divergence(dis2, normal2).mean()
    return CE, rhy_CE, CE + rhy_CE + beta * (KLD1 + KLD2)



batch_size = 128
n_epochs = 100
save_path = "model_backup"  # model_save_path
save_period = 2  # save every 2 epoches
data_path = ["data/irish_train.npy",
             "data/irish_validate.npy",
             "data/irish_test.npy"]
lr = 1e-4
decay = 0.9999
if_parallel = False
hidden_dims = 2048
z1_dims = 128
z2_dims = 128
vae_beta = 0.1
input_dims = 130
rhythm_dims = 3
seq_len = 6 * 4

def train_ec2vae():

    train_x = np.load(data_path[0], allow_pickle=True)
    validate_x = np.load(data_path[1], allow_pickle=True)
    test_x = np.load(data_path[2], allow_pickle=True)
    dl = DataLoader(train=train_x, validate=validate_x, test=test_x)
    # each measure is 24-token, we split it here
    dl.process_split(split_size=seq_len)
    print(len(dl.train_set), len(dl.validate_set), len(dl.test_set))

    model = RECVAE(input_dims, hidden_dims, rhythm_dims, z1_dims, z2_dims, seq_len, 3000)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if decay > 0:
        scheduler = MinExponentialLR(optimizer, gamma=decay, minimum=1e-5)
    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('Using: CPU')

    model.train()
    step = 0
    for epoch in range(n_epochs):
        print("epoch: %d\n__________________________________________" % (epoch), flush=True)
        train_batches, validate_batches = dl.start_new_epoch(batch_size=batch_size)
        for i in range(len(train_batches)):
            # validate display
            j = i % len(validate_batches)
            raw_x = dl.convert_onehot(train_batches[i])
            raw_vx = dl.convert_onehot(validate_batches[j])

            x = torch.from_numpy(raw_x).float()
            target_rhythm = np.expand_dims(raw_x[:, :, :-2].sum(-1), -1)
            target_rhythm = np.concatenate((target_rhythm, raw_x[:, :, -2:]), -1)
            target_rhythm = torch.from_numpy(target_rhythm).float()
            target_rhythm = target_rhythm.view(-1, target_rhythm.size(-1)).max(-1)[1]
            target = x.view(-1, x.size(-1)).max(-1)[1]

            vx = torch.from_numpy(raw_vx).float()
            target_rhythm_vx = np.expand_dims(raw_vx[:, :, :-2].sum(-1), -1)
            target_rhythm_vx = np.concatenate((target_rhythm_vx, raw_vx[:, :, -2:]), -1)
            target_rhythm_vx = torch.from_numpy(target_rhythm_vx).float()
            target_rhythm_vx = target_rhythm_vx.view(-1, target_rhythm_vx.size(-1)).max(-1)[1]
            target_vx = vx.view(-1, vx.size(-1)).max(-1)[1]

            if torch.cuda.is_available():
                x = x.cuda()
                target = target.cuda()
                target_rhythm = target_rhythm.cuda()
                vx = vx.cuda()
                target_vx = target_vx.cuda()
                target_rhythm_vx = target_rhythm_vx.cuda()

            optimizer.zero_grad()
            recon_x, recon_rhythm, dis1, dis2 = model(x)
            dis1 = Normal(dis1.mean, dis1.stddev)
            dis2 = Normal(dis2.mean, dis2.stddev)
            recon_loss, rhythm_loss, total_loss = loss_function(recon_x, recon_rhythm, target, target_rhythm, dis1,
                                                                dis2, vae_beta)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            v_total_loss = 0.0
            v_recon_loss = 0.0
            v_rhythm_loss = 0.0
            with torch.no_grad():
                vrecon_x, vrecon_rhythm, vdis1, vdis2 = model(vx)
                vdis1 = Normal(vdis1.mean, vdis1.stddev)
                vdis2 = Normal(vdis2.mean, vdis2.stddev)
                v_recon_loss, v_rhythm_loss, v_total_loss = loss_function(vrecon_x, vrecon_rhythm, target_vx,
                                                                          target_rhythm_vx, vdis1, vdis2, vae_beta)
            step += 1
            if decay > 0:
                scheduler.step()
            print("batch %d (total_loss, recon_loss, rhythm_loss) = (%.5f, %.5f, %.5f) | validate = (%.5f, %.5f, %.5f)"
                  % (
                  i, total_loss.item(), recon_loss.item(), rhythm_loss.item(), v_total_loss.item(), v_recon_loss.item(),
                  v_rhythm_loss.item()), flush=True)
        if (epoch + 1) % save_period == 0:
            filename = "ec2vae-" + 'loss_' + str(v_total_loss.item()) + "_epoch_" + str(epoch + 1) + ".pt"
            torch.save(model.cpu().state_dict(), os.path.join(save_path, filename))
            model.cuda()