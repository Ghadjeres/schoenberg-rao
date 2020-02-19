import numpy as np
import torch



def rao_sim_entropy(D, p):
    """
    Compute similarity sensitive Rao's quadratic entropy of a (batch of) distribution(s) p
    over an alphabet of n elements.

    Inputs:
        D [n x n tensor] : Distance matrix
        p [batch_size x n tensor] : Probability distributions over n elements

    Output:
        [batch_size x 1 tensor] of entropy for each distribution
    """
    return rao_quad_entropy(D, p, p)


def rao_quad_entropy(D, p, q):
    """
    Compute similarity sensitive  Rao's quadratic entropy of a (batch of) distribution(s) p
    over an alphabet of n elements.

    Inputs:
        D [n x n tensor] : Distance matrix
        p [batch_size x n tensor] : Probability distributions over n elements

    Output:
        [batch_size x 1 tensor] of entropy for each distribution
    """
    ent = torch.einsum('bi,bj,ij->b', p, q, D)
    return ent


def mixtures_Jgaussiansl2(m1, m2):
    """

    :param m1: list of [pi, mu, sigma]
    :param m2:
    :return:
    """
    s = 0
    # print('m1')
    # print(m1)
    # print('m2')
    # print(m2)
    #     f = lambda d: d.pow(0.1)
    # f = lambda d: torch.log(1 + d)
    # f = lambda d: d
    f = lambda d: d.pow(0.5)

    for pi1, mu1, sigma1 in m1:
        for pi2, mu2, sigma2 in m2:
            s += pi1 * pi2 * f((mu1 - mu2) ** 2 + (sigma1 - sigma2) ** 2)

    for pi1, mu1, sigma1 in m1:
        for pi2, mu2, sigma2 in m1:
            s -= 1 / 2 * pi1 * pi2 * f((mu1 - mu2) ** 2 + (sigma1 - sigma2) ** 2)

    for pi1, mu1, sigma1 in m2:
        for pi2, mu2, sigma2 in m2:
            s -= 1 / 2 * pi1 * pi2 * f((mu1 - mu2) ** 2 + (sigma1 - sigma2) ** 2)

    return s


def generate_mixture(num_components):
    mixture = []
    pis = torch.softmax(torch.randn(num_components), dim=0)
    for pi in pis:
        mu = torch.randn(1)
        sigma = torch.rand(1)
        mixture.append([pi, mu, sigma])
    return mixture


def gaussian(x, mu, sigma):
    return torch.exp(- (x - mu) ** 2 / 2 / sigma ** 2) / np.sqrt(2 * np.pi) / sigma


def discret_continu(p, y, mu, sigma):
    N_y = torch.cat([gaussian(yi, mu, sigma) for yi in y], dim=0)
    # print(N_y)
    D = utils.batch_pdist(y.unsqueeze(1), y.unsqueeze(1), p=2)
    return (
            (p * (2 * sigma ** 2 * N_y + (y - mu) * torch.erf((y - mu) / sigma / np.sqrt(2)))).sum()
            - 1 / 2 * torch.einsum('i,j,ij', p, p, D) - sigma / np.sqrt(np.pi)
    )


def SR_gaussian(mu1, sigma1, mu2, sigma2):
    res = (
                  (mu2 - mu1) * torch.erf(
              (mu2 - mu1) / np.sqrt(2) / torch.sqrt(sigma1 ** 2 + sigma2 ** 2))
                  +
                  np.sqrt(2 / np.pi) * torch.sqrt(sigma1 ** 2 + sigma2 ** 2) *
                  torch.exp(- 1 / 2 * (mu1 - mu2) ** 2 / (sigma1 ** 2 + sigma2 ** 2))
          ) - (sigma1 + sigma2) / np.sqrt(np.pi)
    # print(res)
    return res


def SR_gaussian_optimized(mu1, sigma1, mu2, sigma2):
    diff_mu = (mu1.unsqueeze(1) - mu2.unsqueeze(0))
    sum_sigma = (sigma1.unsqueeze(1) + sigma2.unsqueeze(0))
    sum_sigma_square = (sigma1.unsqueeze(1) ** 2 + sigma2.unsqueeze(0) ** 2)

    res = (
            diff_mu * torch.erf(diff_mu / np.sqrt(2) / torch.sqrt(sum_sigma_square))
            + np.sqrt(2 / np.pi) * torch.sqrt(sum_sigma_square) *
            torch.exp(-1 / 2 * diff_mu ** 2 / sum_sigma_square)
            - sum_sigma / np.sqrt(np.pi)
    )
    return res


def SR_gaussian_discrete_optimized(mu1, sigma1, x):
    mu_moins_x = (mu1.unsqueeze(1) - x.unsqueeze(0))

    res = (
            mu_moins_x * torch.erf(mu_moins_x / np.sqrt(2) / sigma1.unsqueeze(1)) +
            sigma1.unsqueeze(1) / np.sqrt(np.pi) * (
                    np.sqrt(2) * torch.exp(
                - (mu_moins_x) ** 2 / 2 / sigma1.unsqueeze(1) ** 2
            ) - 1
            )
    )
    return res


def SR_mixture_discrete_optimized(m1, discrete):
    pi1 = m1[:, 0]
    mu1 = m1[:, 1]
    sigma1 = m1[:, 2]

    pi2 = discrete[:, 0]
    x = discrete[:, 1]

    SR_12 = SR_gaussian_discrete_optimized(mu1, sigma1, x)
    # (batch_size m1, batch_size_m2, num_dim)
    sum = 0
    sum += torch.einsum(
        'mn, m, n->',
        SR_12, pi1, pi2
    )

    SR_11 = SR_gaussian_optimized(mu1, sigma1, mu1, sigma1)
    # (batch_size m1, batch_size_m1, num_dim)
    sum -= 1 / 2 * torch.einsum(
        'mn, m, n->',
        SR_11, pi1, pi1
    )

    SR_22 = torch.abs(
        x.unsqueeze(0) - x.unsqueeze(1)
    )
    # (batch_size m1, batch_size_m1, num_dim)
    sum -= 1 / 2 * torch.einsum(
        'mn, m, n->',
        SR_22, pi2, pi2
    )

    return sum


def SR_mixtures_optimized(m1, m2):
    pi1 = m1[:, 0]
    pi2 = m2[:, 0]
    mu1 = m1[:, 1]
    mu2 = m2[:, 1]
    sigma1 = m1[:, 2]
    sigma2 = m2[:, 2]

    SR_12 = SR_gaussian_optimized(mu1, sigma1, mu2, sigma2)
    # (batch_size m1, batch_size_m2, num_dim)
    sum = 0

    sum += torch.einsum(
        'mn, m, n->',
        SR_12, pi1, pi2
    )

    SR_11 = SR_gaussian_optimized(mu1, sigma1, mu1, sigma1)
    # (batch_size m1, batch_size_m1, num_dim)
    sum -= 1 / 2 * torch.einsum(
        'mn, m, n->',
        SR_11, pi1, pi1
    )
    # DO NOT COMPUTE PRIOR?
    SR_22 = SR_gaussian_optimized(mu2, sigma2, mu2, sigma2)
    # (batch_size m1, batch_size_m1, num_dim)
    sum -= 1 / 2 * torch.einsum(
        'mn, m, n->',
        SR_22, pi2, pi2
    )

    return sum


def SR_discrete_optimized(pi1, x1, pi2, x2):
    # f = lambda x: torch.log(1 + x)
    f = lambda x: torch.pow(x, 0.1)
    # f = lambda x: torch.sqrt(x)
    # f = lambda x: torch.log(1 + torch.sqrt(x))
    # f = lambda x: x
    SR_12 = (((x1.unsqueeze(0) - x2.unsqueeze(1)) ** 2).sum(-1) + 1e-10)
    SR_12 = f(SR_12)

    sum = 0
    sum += torch.einsum(
        'mn, m, n->',
        SR_12, pi1, pi2
    )

    SR_11 = (((x1.unsqueeze(0) - x1.unsqueeze(1)) ** 2).sum(-1) + 1e-10)
    # mask = torch.eye(SR_11.size(0))
    SR_11 = f(SR_11)
    # SR_11 = SR_11 * (1 - mask)

    # (batch_size m1, batch_size_m1, num_dim)
    sum -= 1 / 2 * torch.einsum(
        'mn, m, n->',
        SR_11, pi1, pi1
    )

    SR_22 = (((x2.unsqueeze(0) - x2.unsqueeze(1)) ** 2).sum(-1) + 1e-10)
    SR_22 = f(SR_22)
    # mask = torch.eye(SR_22.size(0))
    # SR_22 = SR_22 * (1 - mask)

    sum -= 1 / 2 * torch.einsum(
        'mn, m, n->',
        SR_22, pi2, pi2
    )
    return sum


def SR_discrete_optimized_batched(pi1, x1, pi2, x2):
    f = lambda x: torch.sqrt(x)
    # f = lambda x: torch.log(1 + torch.sqrt(x))

    SR_12 = (((x1.unsqueeze(1) - x2.unsqueeze(0)) ** 2).sum(-1) + 1e-10)
    SR_12 = f(SR_12)

    sum = 0
    sum += torch.einsum(
        'mn, bm, bn->b',
        SR_12, pi1, pi2
    )

    SR_11 = (((x1.unsqueeze(1) - x1.unsqueeze(0)) ** 2).sum(-1) + 1e-10)
    # mask = torch.eye(SR_11.size(0))
    SR_11 = f(SR_11)
    # SR_11 = SR_11 * (1 - mask)

    # (batch_size m1, batch_size_m1, num_dim)
    sum -= 1 / 2 * torch.einsum(
        'mn, bm, bn->b',
        SR_11, pi1, pi1
    )

    SR_22 = (((x2.unsqueeze(1) - x2.unsqueeze(0)) ** 2).sum(-1) + 1e-10)
    SR_22 = f(SR_22)
    # mask = torch.eye(SR_22.size(0))
    # SR_22 = SR_22 * (1 - mask)

    sum -= 1 / 2 * torch.einsum(
        'mn, bm, bn->b',
        SR_22, pi2, pi2
    )
    return sum


def SR_discrete_same_atoms(D, pi1, pi2):
    "batched"
    sum = 0
    sum += torch.einsum(
        'mn, bm, bn->b',
        D, pi1, pi2
    )

    sum -= 1 / 2 * torch.einsum(
        'mn, bm, bn->b',
        D, pi1, pi1
    )

    sum -= 1 / 2 * torch.einsum(
        'mn, bm, bn->b',
        D, pi2, pi2
    )
    return sum


def SR_mixtures(m1, m2, f=None):
    """

    :param m1: list of [pi, mu, sigma]
    :param m2:
    :param f: additional function to be applied on SR
    :return:
    """
    sum = 0
    if f is None:
        f = lambda d: d

    for pi1, mu1, sigma1 in m1:
        for pi2, mu2, sigma2 in m2:
            sum += pi1 * pi2 * f(SR_gaussian(mu1, sigma1, mu2, sigma2))

    for pi1, mu1, sigma1 in m1:
        for pi2, mu2, sigma2 in m1:
            sum -= 1 / 2 * pi1 * pi2 * f(SR_gaussian(mu1, sigma1, mu2, sigma2))

    for pi1, mu1, sigma1 in m2:
        for pi2, mu2, sigma2 in m2:
            sum -= 1 / 2 * pi1 * pi2 * f(SR_gaussian(mu1, sigma1, mu2, sigma2))
    return sum

from skimage.transform import resize

def sample_and_resize(D, img_size, num_samples=1):
    res = []
    for _ in range(num_samples):
        ix = int(np.random.choice(len(D), 1))
        sample_img = D[ix, ...].data.numpy()
        #sample_img = tform(sample_img)[0, ...].data.numpy()
        if img_size != 28:
            sample_img = resize(sample_img, (img_size, img_size), mode='constant')
        sample_img = np.abs(sample_img) / np.abs(sample_img).sum()
        res.append(sample_img)
    return torch.tensor(res).double()

def batch_pdist(X, Y, p=2):
    return torch.norm(X[..., None, :] - Y[..., None, :, :], p=p, dim=-1)