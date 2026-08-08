import torch
import torch.nn as nn
from torch.autograd import grad as autograd_grad


def gaussian_log_prob(omega):
    mean = torch.tensor([0.,0.,0.])
    stddev = torch.tensor([.5,1.,2.])
    ll = torch.distributions.MultivariateNormal(mean, torch.diag(stddev**2)).log_prob(omega)
    return ll.sum()

def banana_log_prob(w, a = 1, b = 1, c = 1):
    ll = -(1/200) * torch.square(a * w[..., 0]) - .5 * torch.square(c*w[..., 1] + b * torch.square(a * w[..., 0]) - 100 * b)
    return ll.sum()

def high_dimensional_gaussian_log_prob(w, D):
    ll = torch.distributions.MultivariateNormal(torch.zeros(D), covariance_matrix=torch.diag(torch.ones(D))).log_prob(w)

    return ll.sum()

def normal_normal_conjugate(w):
    mu0 = 0.0
    tau = 1.5
    sigma = torch.exp(w[..., 1]) + .001
    ll = torch.distributions.Normal(mu0, tau).log_prob(w[..., 0])
    ll += torch.distributions.InverseGamma(2, 3).log_prob(sigma)
    ll += torch.distributions.Normal(1.7, sigma).log_prob(w[..., 0])
    return ll.sum()

def high_dimensional_warped_gaussian_log_prob(w, D, scales):
    mean = torch.zeros(D)
    cov = torch.diag(scales)
    ll = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).log_prob(w)
    return ll.sum()
    


def compute_reversibility_error(model, test_initial_conditions, t):
    D = test_initial_conditions.shape[-1] // 2
    _, forward_trajectories = model(test_initial_conditions, t)
    forward_trajectories = torch.swapaxes(forward_trajectories, 0, 1)
    end_positions = forward_trajectories[:,-1,:]
    backward_conditions = torch.matmul(end_positions, torch.block_diag(torch.eye(D), -1*torch.eye(D)))
    _, backward_trajectories = model(backward_conditions , t)
    backward_trajectories = torch.swapaxes(backward_trajectories, 0, 1)
    loss = nn.MSELoss()(backward_trajectories[:, -1, :D].detach(), test_initial_conditions[..., :D].detach())
    return loss, forward_trajectories[..., :D].detach(), backward_trajectories[..., :D].detach()

def compute_hamiltonian_error(model, test_initial_conditions, t, log_prob_func):
    D = test_initial_conditions.shape[-1] // 2
    hamiltonian = lambda x: -1*log_prob_func(x[..., :D]) + .5 * torch.sum(torch.square(x[..., D:]),dim=-1)
    initial_hamiltonian_values = hamiltonian(test_initial_conditions)
    _, trajectories = model(test_initial_conditions, t)
    forward_trajectories = torch.swapaxes(trajectories.detach(), 0, 1)
    batched_hamiltonian = torch.vmap(hamiltonian)
    try:
        forward_hamiltonians = batched_hamiltonian(forward_trajectories)
    except:
        hamiltonians_list = []
        bad_index = []
        for i in range(test_initial_conditions.shape[0]):
            try:
                hamiltonians_list.append(hamiltonian(trajectories[i]))
            except:
                bad_index.append(i)
        forward_hamiltonians = torch.stack(hamiltonians_list, axis = 0)

        index = torch.ones(test_initial_conditions.shape[0], dtype=bool)
        index[bad_index] = False
        initial_hamiltonian_values = initial_hamiltonian_values[index]

    delta_hamiltonian = torch.abs(forward_hamiltonians - initial_hamiltonian_values[:, None]) / (initial_hamiltonian_values[:, None])
    return torch.mean(delta_hamiltonian, dim = -1)


def params_grad(p, log_prob_func):
    p = p.detach().requires_grad_(True)
    return autograd_grad(log_prob_func(p), p, create_graph=False)[0]






def funnel_log_prob(w):
    """Neal's funnel (2-D): v ~ N(0, 9), x ~ N(0, e^v).

    The canonical target where position-dependent curvature defeats plain HMC
    and Riemannian methods shine."""
    v, x = w[..., 0], w[..., 1]
    ll = torch.distributions.Normal(0., 3.).log_prob(v)
    ll = ll + torch.distributions.Normal(0., torch.exp(v / 2)).log_prob(x)
    return ll.sum()


def make_gp_regression_log_prob(num_data=200):
    """Log posterior of GP regression hyperparameters (log lengthscale,
    log signal variance, log noise variance) with standard-normal priors.

    Each evaluation requires an O(N^3) Cholesky, so gradient calls are
    genuinely expensive — the regime surrogate HMC targets. The dataset is
    deterministic (pseudo-noise from a fixed formula) so every run and every
    device sees identical data without RNG/device coupling.
    """
    x = torch.linspace(-3., 3., num_data)
    true_f = torch.sin(2. * x) + 0.5 * torch.cos(5. * x)
    noise = 0.1 * torch.sin(123.456 * x * x)
    y = true_f + noise
    sq_dists = (x[:, None] - x[None, :]) ** 2

    def log_prob(w):
        log_l, log_s, log_n = w[..., 0], w[..., 1], w[..., 2]
        ls2 = torch.exp(2. * log_l)[..., None, None]
        sig = torch.exp(log_s)[..., None, None]
        noise_v = (torch.exp(log_n) + 1e-4)[..., None, None]
        K = sig * torch.exp(-sq_dists / (2. * ls2)) + noise_v * torch.eye(num_data)
        ll = torch.distributions.MultivariateNormal(
            torch.zeros(num_data), covariance_matrix=K).log_prob(y)
        prior = torch.distributions.Normal(0., 1.).log_prob(w).sum()
        return ll.sum() + prior

    return log_prob


def compute_rm_hamiltonian_error(model, test_initial_conditions, t, rm_hamiltonian_func):
    """Mean relative drift of a *Riemannian* Hamiltonian along model
    trajectories. rm_hamiltonian_func(q, p) evaluates the exact non-separable
    Hamiltonian for a single (unbatched) phase-space point; rows whose exact
    Hamiltonian cannot be evaluated (diverged trajectories) are skipped."""
    D = test_initial_conditions.shape[-1] // 2
    _, trajectories = model(test_initial_conditions, t)
    traj = torch.swapaxes(trajectories.detach(), 0, 1)  # (B, T, 2D)
    errors = []
    for b in range(traj.shape[0]):
        try:
            h = torch.stack([
                rm_hamiltonian_func(traj[b, i, :D], traj[b, i, D:]).reshape(())
                for i in range(traj.shape[1])
            ])
            errors.append(torch.mean(torch.abs((h - h[0]) / h[0])))
        except Exception:
            continue
    if not errors:
        return torch.tensor(float("nan"))
    return torch.stack(errors).mean()
