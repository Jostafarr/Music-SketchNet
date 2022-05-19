from MeasureVAE.measure_vae import MeasureVAE
from utils.helpers import *
from loader.dataloader import DataLoader
import numpy as np
from torch import optim

data_path = [
    "data/irish_train.npy",
    "data/irish_validate.npy",
    "data/irish_test.npy"
]
# paramters initialization
num_notes = 130
note_embedding_dim=10
metadata_embedding_dim=2
num_encoder_layers=2
encoder_hidden_size=512
encoder_dropout_prob=0.5
has_metadata=False
latent_space_dim=256
num_decoder_layers=2
decoder_hidden_size=512
decoder_dropout_prob=0.5
batch_size=256
num_epochs=30
train=True
plot=False
log=True
lr=1e-4
seq_len = 6 * 4
n_epochs = 50
save_period = 2
save_path = "model_backup"

def compute_kld_loss(z_dist, prior_dist, beta=0.001):
    """

    :param z_dist: torch.nn.distributions object
    :param prior_dist: torch.nn.distributions
    :param beta:
    :return: kl divergence loss
    """
    kld = torch.distributions.kl.kl_divergence(z_dist, prior_dist)
    kld = beta * kld.sum(1).mean()
    return kld

def mean_crossentropy_loss(weights, targets):
    """
    Evaluates the cross entropy loss
    :param weights: torch Variable,
            (batch_size, seq_len, num_notes)
    :param targets: torch Variable,
            (batch_size, seq_len)
    :return: float, loss
    """
    criteria = torch.nn.CrossEntropyLoss(reduction='mean')
    batch_size, seq_len, num_notes = weights.size()
    weights = weights.contiguous().view(-1, num_notes)
    targets = targets.contiguous().view(-1)
    loss = criteria(weights, targets)
    return loss

def mean_accuracy(weights, targets):
    """
    Evaluates the mean accuracy in prediction
    :param weights: torch Variable,
            (batch_size, seq_len, num_notes)
    :param targets: torch Variable,
            (batch_size, seq_len)
    :return float, accuracy
    """
    _, _, num_notes = weights.size()
    weights = weights.contiguous().view(-1, num_notes)
    targets = targets.contiguous().view(-1)

    _, max_indices = weights.max(1)
    correct = max_indices == targets
    return torch.sum(correct.float()) / targets.size(0)


def train_measure_vae():
    data_path = [
        "data/irish_train.npy",
        "data/irish_validate.npy",
        "data/irish_test.npy"
    ]

    train_x = np.load(data_path[0], allow_pickle=True)
    validate_x = np.load(data_path[1], allow_pickle=True)
    test_x = np.load(data_path[2], allow_pickle=True)
    dl = DataLoader(train=train_x, validate=validate_x, test=test_x)
    dl.process_split(split_size=seq_len)
    print(len(dl.train_set), len(dl.validate_set), len(dl.test_set))

    model = MeasureVAE(
        num_notes=num_notes,
        note_embedding_dim=note_embedding_dim,
        metadata_embedding_dim=metadata_embedding_dim,
        num_encoder_layers=num_encoder_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_dropout_prob=encoder_dropout_prob,
        latent_space_dim=latent_space_dim,
        num_decoder_layers=num_decoder_layers,
        decoder_hidden_size=decoder_hidden_size,
        decoder_dropout_prob=decoder_dropout_prob,
        has_metadata=has_metadata
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('Using: CPU')

    print("MeasureVAE parameters: " + model.parameters())
    return
    model.train()
    step = 0
    for epoch in range(n_epochs):
        print("epoch: %d\n__________________________________________" % (epoch), flush=True)
        train_batches, validate_batches = dl.start_new_epoch(batch_size=batch_size)
        for i in range(len(train_batches)):
            model.train()
            # validate display
            j = i % len(validate_batches)
            raw_x = train_batches[i]
            raw_vx = validate_batches[j]
            x = torch.from_numpy(raw_x).long()
            target = x.view(-1)
            v_x = torch.from_numpy(raw_vx).long()
            v_target = v_x.view(-1)

            if torch.cuda.is_available():
                x = x.cuda()
                target = target.cuda()
                v_x = v_x.cuda()
                v_target = v_target.cuda()

            optimizer.zero_grad()
            weights, samples, z_dist, prior_dist, z_tilde, z_prior = model(measure_score_tensor=x, train=True)
            recons_loss = mean_crossentropy_loss(weights=weights, targets=target)
            dist_loss = compute_kld_loss(z_dist, prior_dist)
            loss = recons_loss + dist_loss
            accuracy = mean_accuracy(weights=weights, targets=target)

            loss.backward()
            optimizer.step()

            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                v_weights, v_samples, v_z_dist, v_prior_dist, v_z_tilde, v_z_prior = model(measure_score_tensor=v_x,
                                                                                           train=False)
                v_recons_loss = mean_crossentropy_loss(weights=v_weights, targets=v_target)
                v_dist_loss = compute_kld_loss(v_z_dist, v_prior_dist)
                v_loss = v_recons_loss + v_dist_loss
                v_accuracy = mean_accuracy(weights=v_weights, targets=v_target)
            print("batch %d loss: %.5f acc: %.5f| val loss %.5f acc: %.5f "
                  % (i, loss.item(), accuracy.item(), v_loss.item(), v_accuracy.item()), flush=True)
        if (epoch + 1) % save_period == 0:
            filename = "measure-vae-" + 'loss_' + str(v_loss.item()) + "_acc_" + str(
                v_accuracy.item()) + "_epoch_" + str(epoch + 1) + ".pt"
            torch.save(model.cpu().state_dict(), os.path.join(save_path, filename))
            model.cuda()