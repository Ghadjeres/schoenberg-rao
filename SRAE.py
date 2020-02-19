import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from SR import SR_mixtures_optimized, SR_mixture_discrete_optimized, \
    SR_discrete_optimized
import os

if not os.path.exists('results'):
    os.mkdir('results')

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--sample_prior', action='store_true',
                    help=f'If we regularize using samples from the prior or using a closed-form '
                         f'formula')
parser.add_argument('--prior_shape', type=str, default='unit',
                    choices=['unit', 'mog'],
                    help=f'prior can be either a unit Gaussian or a mixture of Gaussian '
                         f'distributions')
parser.add_argument('--encoder_type', type=str, default='deterministic',
                    choices=['deterministic', 'stochastic'],
                    help=f'If we use deterministic or stochastic encoders')
parser.add_argument('--loss', type=str, default='bce',
                    choices=['bce', 'gaussian'],
                    help=f'Which loss on the image space to use')
parser.add_argument('--z_dim', type=int, default=8,
                    help=f'Dimension of the latent space')
parser.add_argument('--lr', type=float, default=1e-4,
                    help=f'Learning rate')
parser.add_argument('--beta', type=float, default=100,
                    help=f'Regularization coefficient in front of the SR term')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

STOCHASTIC_ENCODER = args.encoder_type == 'stochastic'
LOSS_FUNCTION = args.loss
z_dim = args.z_dim
lr = args.lr
PRIOR_SHAPE = args.prior_shape
SAMPLE_PRIOR = args.sample_prior
BETA = args.beta


class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 1024)
        self.fc21 = nn.Linear(1024, z_dim)
        self.fc22 = nn.Linear(1024, z_dim)
        self.fc3 = nn.Linear(z_dim, 1024)
        self.fc4 = nn.Linear(1024, 784)
        self.precision = nn.Parameter(torch.ones(1) * -5)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        # change logvar init
        return self.fc21(h1), self.fc22(h1) - 5

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))

        # stochastic
        if STOCHASTIC_ENCODER:
            z = self.reparameterize(mu, logvar)
        else:
            # deterministic
            z = mu
        return self.decode(z), self.precision, mu, logvar


model = VAE(z_dim=z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


# Sliced Schoenberg-Rao
def loss_function_SSR(recon_x, x, mu, logvar, precision):
    if LOSS_FUNCTION == 'bce':
        RE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    else:
        RE = ((recon_x - x.view(-1, 784)) ** 2 / 2 / torch.exp(precision) + precision / 2).sum()

    # sample b
    b = torch.randn(mu.size(-1))
    b /= torch.norm(b, p=2)
    b = b.to(device)

    mu = (mu * b.unsqueeze(0)).sum(1)
    if STOCHASTIC_ENCODER:
        sigma = torch.sqrt((torch.exp(logvar) * (b * b).unsqueeze(0)).sum(1))

        m = torch.cat([
            torch.ones_like(mu).unsqueeze(1) / mu.size(0),
            mu.unsqueeze(1),
            sigma.unsqueeze(1)], dim=1

        )
    else:
        discrete = torch.cat([
            torch.ones_like(mu).unsqueeze(1) / mu.size(0),
            mu.unsqueeze(1)
        ], dim=1
        )

    # prior
    if PRIOR_SHAPE == 'unit':
        m_prior = torch.Tensor([[
            1, 0, 1
        ]
        ]
        )
    else:
        diag = torch.zeros_like(logvar[:1])
        diag[0, 0] = 1
        diag[0, 1] = 1
        mu_prior_diag = (diag * b.unsqueeze(0)).sum(1)
        anti_diag = torch.zeros_like(logvar[:1])
        anti_diag[0, 0] = 1
        anti_diag[0, 1] = -1
        mu_prior_anti_diag = (anti_diag * b.unsqueeze(0)).sum(1)
        sigma_prior = torch.sqrt((torch.ones_like(logvar[:1]) * 0.4 * 0.4 * (b * b).unsqueeze(
            0)).sum(1))

        # m_prior = torch.Tensor([
        #     [
        #         1 / 2, mu_prior, sigma_prior
        #     ],
        #     [
        #         1 / 2, -mu_prior, sigma_prior
        #     ]
        # ]
        # )

        m_prior = torch.Tensor([
            [
                1 / 4, mu_prior_diag, sigma_prior
            ],
            [
                1 / 4, -mu_prior_diag, sigma_prior
            ],
            [
                1 / 4, mu_prior_anti_diag, sigma_prior
            ],
            [
                1 / 4, -mu_prior_anti_diag, sigma_prior
            ]
        ]
        )

    m_prior = m_prior.to(device)

    if STOCHASTIC_ENCODER:
        SR = SR_mixtures_optimized(m, m_prior)
    else:
        SR = SR_mixture_discrete_optimized(m_prior, discrete).sum()
    return RE + 128 * 784 * BETA * SR, RE.item(), SR.item()


# Schoenberg-Rao on samples
def loss_function_SR(recon_x, x, mu, logvar, precision):
    if LOSS_FUNCTION == 'bce':
        RE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    else:
        RE = ((recon_x - x.view(-1, 784)) ** 2 / 2 / torch.exp(precision) + precision / 2).sum()

    # prior
    if PRIOR_SHAPE == 'unit':
        prior = torch.randn_like(mu)
    else:
        # mog
        prior = torch.randn_like(mu) * 0.4

        offset = (2 * torch.randint(2, (mu.size(0),)) - 1).to(device)
        prior[:, 0] += offset
        offset = (2 * torch.randint(2, (mu.size(0),)) - 1).to(device)
        prior[:, 1] += offset

    # using SR_discrete
    emp_p = torch.ones_like(mu[:, 0]) / mu.size(0)

    prior = prior.to(device)
    if STOCHASTIC_ENCODER:
        samples = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
    else:
        samples = mu
    SR = SR_discrete_optimized(emp_p, prior, emp_p, samples)
    return RE + 128 * 784 * BETA * SR, RE.item(), SR.item()


if SAMPLE_PRIOR:
    loss_function = loss_function_SR
else:
    loss_function = loss_function_SSR


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, precision, mu, logvar = model(data)
        loss, bce, sr = loss_function(recon_batch, data, mu, logvar, precision)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))
            print(f'RE: {bce} SR: {sr}')

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, precision, mu, logvar = model(data)
            loss, _, _ = loss_function(recon_batch, data, mu, logvar, precision)
            test_loss += loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():

            # plot samples
            sample = torch.randn(64, z_dim).to(device)
            if not PRIOR_SHAPE == 'unit':
                sample *= 0.4
                offset = (2 * torch.randint(2, (64,)) - 1).to(device)
                sample[:, 0] += offset
                offset = (2 * torch.randint(2, (64,)) - 1).to(device)
                sample[:, 1] += offset
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

            # plot histograms
            import seaborn as sns
            import pandas as pd

            mus = []
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon_batch, precision, mu, logvar = model(data)

                # stochastic
                if STOCHASTIC_ENCODER:
                    samples = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
                else:
                    samples = mu
                mus.append(samples)
                if i > 32:
                    break

            mu = torch.cat(mus, dim=0)

            mu = mu.detach().cpu()
            df = {'x': mu[:, 0], 'y': mu[:, 1]}
            df = pd.DataFrame(data=df)
            g = sns.jointplot(x='x', y='y', data=df, kind='kde', xlim=(-3., 3),
                              ylim=(-3, 3))
            g.savefig(f'results/jointplot_{epoch}.png')
