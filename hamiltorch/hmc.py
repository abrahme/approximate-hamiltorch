import torch
from abc import abstractmethod, ABC
from . import util
from typing import Union
from .models import train, train_ode, train_symplectic, NNgHMC, HNNEnergyDeriv, HNNEnergyExplicit, HNNODE, HNN, GSymplecticNeuralNetwork, SymplecticNeuralNetwork, create_training_set_symplectic, create_training_set_symplectic_with_gradients
from .models import RMHNN, RMHNNEnergyExplicit, RMHNNEnergyDeriv, RMHNNODE, NNODEgRMHMC
from .symplectic import TimeSymmetricSymplectic
from .ode import NonSeparableSynchronousLeapfrog
from .samplers import fisher, rm_hamiltonian, Sampler, Integrator, Metric

def collect_gradients(log_prob, params, pass_grad = None):
    """Returns the parameters and the corresponding gradients (params.grad).

    Parameters
    ----------
    log_prob : torch.tensor
        Tensor shape (1,) which is a function of params (Can also be a tuple where log_prob[0] is the value to be differentiated).
    params : torch.tensor
        Flat vector of model parameters: shape (D,), where D is the dimensionality of the parameters .
    pass_grad : None or torch.tensor or callable.
        If set to a torch.tensor, it is used as the gradient  shape: (D,), where D is the number of parameters of the model. If set
        to callable, it is a function to be called instead of evaluating the gradient directly using autograd. None is default and
        means autograd is used.

    Returns
    -------
    torch.tensor
        The params, which is returned has the gradient attribute attached, i.e. params.grad.

    """

    if isinstance(log_prob, tuple):
        log_prob[0].backward()
        params_list = list(log_prob[1])
        params = torch.cat([p.flatten() for p in params_list])
        params.grad = torch.cat([p.grad.flatten() for p in params_list])
    elif pass_grad is not None:
        if callable(pass_grad):
            params.grad = pass_grad(params)
        else:
            params.grad = pass_grad
    else:
        params.grad = torch.autograd.grad(log_prob,params)[0]
    return params


class HMCBase(ABC):
    def __init__(self, step_size: float, L: int, log_prob_func: callable, dim: int):
        self.step_size = step_size
        self.L = L
        self.log_prob_func = log_prob_func
        self.dim = dim

    @abstractmethod
    def gibbs(self, q=None):
        """Draw a fresh momentum. `q` is the current position, used by
        samplers whose momentum distribution is position-dependent (RMHMC)."""
        return torch.distributions.Normal(torch.zeros(self.dim), torch.ones(self.dim)).sample()
    
    @abstractmethod
    def hamiltonian(self, q, p):
        return -self.log_prob_func(q) + .5 * torch.square(p).sum()
    
    @classmethod
    def metropolis_accept_step(cls, hamiltonian_old, hamiltonian_new):
        rho = min(0., float((-hamiltonian_new + hamiltonian_old).detach()))
        if rho >= torch.log(torch.rand(1)):
            return True
        else:
            return False
    
    @abstractmethod
    def step(self, q, p, *args, **kwargs):
        """
        this method generates the trajectory starting at initial q, p
        
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample(self, *args, **kwargs):
        """
        this method samples 
        
        """
        raise NotImplementedError
    
    @abstractmethod
    def params_grad(self,*args, **kwargs):
        """
        this method computes gradient of log prob function wrt params
        """
        raise NotImplementedError
    


class HMC(HMCBase):
    def __init__(self, step_size: float, L: int,  log_prob_func: callable, dim: int):
        super().__init__(step_size, L, log_prob_func, dim)
    
    def step(self, q, p, grad_func = None):
        """Leapfrog integration starting from (q, p).

        Returns trajectories of L+1 points at times 0, eps, ..., L*eps. The
        initial state is included and all stored momenta are time-synchronized
        (the staggered half-step momentum is corrected at every step, not just
        the last), so trajectories are directly usable as surrogate training
        data. The final (q, p) pair is identical to the standard leapfrog
        proposal, so sampling behavior is unchanged.
        """
        p_grad = self.params_grad(q, grad_func)
        ret_params = [q.clone()]
        ret_momenta = [p.clone()]
        ret_grad = [p_grad.clone()]
        # half-kick: p is staggered (at time t + eps/2) inside the loop
        p = p + 0.5 * self.step_size * p_grad
        for n in range(self.L):
            q = q + self.step_size * p
            p_grad = self.params_grad(q, grad_func)
            ret_params.append(q.clone())
            # synchronized momentum at integer time = staggered p + half kick
            ret_momenta.append(p + 0.5 * self.step_size * p_grad)
            ret_grad.append(p_grad.clone())
            p = p + self.step_size * p_grad
        return torch.stack(ret_params,axis = 0),  torch.stack(ret_momenta,axis=0), torch.stack(ret_grad, axis = 0)
    
    def params_grad(self, q, pass_grad):
        q = q.detach().requires_grad_()
        log_prob = self.log_prob_func(q)
        q = collect_gradients(log_prob, q, pass_grad)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return q.grad
    
    def gibbs(self, q=None):
        return super().gibbs(q)
    
    def hamiltonian(self, q, p):
        return super().hamiltonian(q, p)

    def sample(self, q_init, grad_func = None, num_samples=1000):

        """
        returns all trajectories for parameter, momentum, gradient, as well as if sample was accepted
        
        """
        device = q_init.device
        params = q_init.clone().requires_grad_()
        param_burn_prev = q_init.clone()
        ret_params = [params.clone()]
        num_rejected = 0
        accepted = []
        param_trajectories = []
        gradient_trajectories = []
        momentum_trajectories = []
        util.progress_bar_init('Sampling ({}; {})'.format("HMC", "Leapfrog"), num_samples, 'Samples')
        for n in range(num_samples):
            util.progress_bar_update(n)
            try:
                momentum = self.gibbs(params)

                ham = self.hamiltonian(params, momentum)

                leapfrog_params, leapfrog_momenta, leapfrog_grad = self.step(params, momentum, grad_func)
                
                param_trajectories.append(leapfrog_params)
                gradient_trajectories.append(leapfrog_grad)
                momentum_trajectories.append(leapfrog_momenta)
                params = leapfrog_params[-1].to(device).detach().requires_grad_()
                momentum = leapfrog_momenta[-1].to(device)
                new_ham = self.hamiltonian(params, momentum)

                if self.metropolis_accept_step(ham, new_ham):
                    param_burn_prev = leapfrog_params[-1].to(device).clone()
                    accepted.append(1.0)
                else:
                    num_rejected += 1
                    params = param_burn_prev.clone()
                    accepted.append(0.0)
            except util.LogProbError:
                num_rejected += 1
                params = ret_params[-1].to(device)
                params = param_burn_prev.clone()

        util.progress_bar_end('Acceptance Rate {:.2f}'.format(1 - num_rejected/num_samples)) #need to adapt for burn

        return torch.stack(param_trajectories,axis=0), torch.stack(momentum_trajectories,axis=0), torch.stack(gradient_trajectories,axis=0), torch.Tensor(accepted)
    

class HMCGaussianAnalytic(HMC):
    def __init__(self, step_size: float, L: int, log_prob_func: callable, dim: int, a:torch.Tensor):
        super().__init__(step_size, L, log_prob_func, dim)
        self.a = 1. / a  ### this is basically the inverse of the diagonal covariance matrix


    def compute_analytical_hamiltonian_path_gaussian(self, hamiltonian: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        computes analytical hamiltonian solutions of the form p^2/a^2 + q^2/b^2 = 1. 
        """
        b = torch.ones(self.dim)
        t = torch.linspace(0, end=self.L*self.step_size, steps=self.L + 1)
        new_a = torch.sqrt(hamiltonian * a * 2)
        new_b = torch.sqrt(hamiltonian * b * 2)
        return torch.outer(torch.cos(t),new_a),  torch.outer(torch.sin(t), new_b)

    def compute_analytical_hamiltonian_gradient_gaussian(self, hamiltonian: torch.Tensor,  a: torch.Tensor) -> torch.Tensor:
        b = torch.ones(self.dim)
        t = torch.linspace(0, end=self.L*self.step_size, steps=self.L + 1)
        new_a = torch.sqrt(hamiltonian * a * 2)
        new_b = torch.sqrt(hamiltonian * b * 2)
        return -torch.outer(torch.sin(t),new_a),  torch.outer(torch.cos(t), new_b)

    def step(self, q, p, *args):
        if q.dim() == 1:
            ham = self.hamiltonian(q, p)
            leapfrog_params, leapfrog_momenta = self.compute_analytical_hamiltonian_path_gaussian(ham, self.a)
            _, gradient_momenta = self.compute_analytical_hamiltonian_gradient_gaussian(ham ,self.a)

            return leapfrog_params, leapfrog_momenta, -gradient_momenta

        # batched diagnostics path: per-row Hamiltonian, trajectories stacked
        # time-first as (T, B, D) to mirror the unbatched (T, D) convention
        hams = torch.stack([self.hamiltonian(q[i], p[i]) for i in range(q.shape[0])])
        t = torch.linspace(0, end=self.L * self.step_size, steps=self.L + 1)
        new_a = torch.sqrt(2. * hams[:, None] * self.a[None, :])
        new_b = torch.sqrt(2. * hams[:, None] * torch.ones(1, self.dim))
        cos_t, sin_t = torch.cos(t)[:, None, None], torch.sin(t)[:, None, None]
        leapfrog_params = cos_t * new_a[None, ...]
        leapfrog_momenta = sin_t * new_b[None, ...]
        gradient_momenta = cos_t * new_b[None, ...]
        return leapfrog_params, leapfrog_momenta, -gradient_momenta
    
    def sample(self, q_init, grad_func=None, num_samples=1000):
        return super().sample(q_init, grad_func, num_samples)

    def params_grad(self, q, pass_grad):
        raise NotImplementedError
    
    def hamiltonian(self, q, p):
        return super().hamiltonian(q, p)
    
    def gibbs(self, q=None):
        return super().gibbs(q)
    
    

class SurrogateHMCBase(HMC):
    def __init__(self, step_size: float, L: int, log_prob_func: callable, dim: int, base_sampler: Union[HMC, HMCGaussianAnalytic]):
        super().__init__(step_size, L, log_prob_func, dim)
        self.base_sampler = base_sampler
        self.model = None
        self.burn_state = None

    @abstractmethod
    def create_surrogate(self, *args, **kwargs):
        raise NotImplementedError
    
    def params_grad(self, q, pass_grad):
        return super().params_grad(q, pass_grad)
    
    def gibbs(self, q=None):
        return super().gibbs(q)
    
    def hamiltonian(self, q, p):
        return super().hamiltonian(q, p)
    
    def sample(self, q_init, grad_func=None, num_samples=1000):
        return super().sample(q_init, grad_func, num_samples)
    
    def step(self, q, p, grad_func=None):
        return super().step(q, p, grad_func)

class SurrogateGradientHMC(SurrogateHMCBase):
    def __init__(self, step_size: float, L: int, log_prob_func: callable, dim: int, base_sampler: Union[HMC, HMCGaussianAnalytic] ):
        super().__init__(step_size, L, log_prob_func, dim, base_sampler)
    
    def create_surrogate(self, q_init: torch.Tensor, burn:int, epochs: int):
        param_examples, _, grad_examples, _ = self.base_sampler.sample(q_init, num_samples=burn)
        model =  NNgHMC(input_dim = self.dim, output_dim = self.dim, hidden_dim =  100 * self.dim)
        
        self.model, _ = train(model, torch.flatten(param_examples, end_dim=1).detach(), 
                              torch.flatten(grad_examples, end_dim=1).detach(), epochs=epochs)
        self.burn_state = param_examples[-1, -1, :].detach()

    def step(self, q, p, grad_func):
        return super().step(q, p, grad_func)
    
    def sample(self, q_init = None, num_samples=1000):
        return super().sample(self.burn_state if q_init is None else q_init, self.model, num_samples)
    
    def params_grad(self, q, pass_grad):
        return super().params_grad(q, pass_grad)
    
    def hamiltonian(self, q, p):
        return super().hamiltonian(q, p)
    
    def gibbs(self, q=None):
        return super().gibbs(q)
    

class SurrogateNeuralODEHMC(SurrogateHMCBase):
    def __init__(self, step_size: float, L: int, log_prob_func: callable, dim: int, base_sampler: Union[HMC, HMCGaussianAnalytic], model_type:str ):
        super().__init__(step_size, L, log_prob_func, dim, base_sampler)
        self.model_type = model_type
    
    def create_surrogate(self, q_init: torch.Tensor, burn:int, epochs: int, solver:str, sensitivity: str):
        param_examples, momenta_examples, grad_examples, _ = self.base_sampler.sample(q_init, num_samples=burn)
        model = HNNODE(HNNEnergyDeriv(input_dim = self.dim, hidden_dim= 100 * self.dim) , solver = solver, sensitivity=sensitivity)
        if self.model_type == "explicit_hamiltonian":
            model = HNNODE(HNN(HNNEnergyExplicit(self.dim, self.dim * 100)), sensitivity=sensitivity, solver = solver)

        # Trajectories now include the initial state: L+1 points at times
        # 0, eps, ..., L*eps, so the integration grid spacing is exactly eps.
        self.model, _ = train_ode(model,
                                  X = torch.cat([param_examples[:, 0, :], momenta_examples[:, 0, :]], dim = 1).detach(),
                                  y = torch.cat([param_examples, momenta_examples], dim = 2).detach(),
                                    t = torch.linspace(start = 0, end = self.L*self.step_size, steps=self.L + 1),
                                    epochs=epochs,
                                    gradient_traj=grad_examples.detach())
        self.burn_state = param_examples[-1, -1, :].detach()

    def step(self, q, p):
        if self.model is None:
            raise ValueError("Surrogate model is not fit")

        initial_positions = torch.cat([q,p])[None,...]
        t = torch.linspace(start = 0, end = self.L*self.step_size, steps=self.L + 1)
        with torch.no_grad():
            _, leapfrog_values = self.model.forward(initial_positions, t)
        return torch.squeeze(leapfrog_values[...,:self.dim]), torch.squeeze(leapfrog_values[...,self.dim:])
        
    
    def sample(self, q_init = None, num_samples=1000):

        """
        returns all trajectories for parameter, momentum, gradient, as well as if sample was accepted
        
        """
        q_init = self.burn_state if q_init is None else q_init
        device = q_init.device
        params = q_init.clone().requires_grad_()
        param_burn_prev = q_init.clone()
        ret_params = [params.clone()]
        num_rejected = 0
        accepted = []
        param_trajectories = []
        momentum_trajectories = []
        util.progress_bar_init('Sampling ({}; {})'.format("HMC", "Leapfrog"), num_samples, 'Samples')
        for n in range(num_samples):
            util.progress_bar_update(n)
            try:
                momentum = self.gibbs(params)
                ham = self.hamiltonian(params, momentum)

                leapfrog_params, leapfrog_momenta = self.step(params, momentum)
                proposed_params = leapfrog_params[-1].to(device).detach().requires_grad_()
                proposed_momentum = leapfrog_momenta[-1].to(device)
                new_ham = self.hamiltonian(proposed_params, proposed_momentum)

                if self.metropolis_accept_step(ham, new_ham):
                    param_burn_prev = proposed_params.detach().clone()
                    params = proposed_params
                    param_trajectories.append(leapfrog_params)
                    momentum_trajectories.append(leapfrog_momenta)
                    accepted.append(1.0)
                else:
                    num_rejected += 1
                    params = param_burn_prev.clone().requires_grad_()
                    # Store the current accepted position so the chain reflects reality
                    param_trajectories.append(
                        param_burn_prev.unsqueeze(0).expand_as(leapfrog_params).clone()
                    )
                    momentum_trajectories.append(
                        momentum.unsqueeze(0).expand_as(leapfrog_momenta).clone()
                    )
                    accepted.append(0.0)
            except util.LogProbError:
                num_rejected += 1
                params = param_burn_prev.clone()

        util.progress_bar_end('Acceptance Rate {:.2f}'.format(1 - num_rejected/num_samples))

        return torch.stack(param_trajectories, axis=0), torch.stack(momentum_trajectories, axis=0), None, torch.Tensor(accepted)
    
    def params_grad(self, *args, **kwargs):
        return super().params_grad(*args, **kwargs)
    
    def gibbs(self, q=None):
        return super().gibbs(q)
    
    def hamiltonian(self, q, p):
        return super().hamiltonian(q, p)
    

    
class SymplecticHMC(SurrogateNeuralODEHMC):
    def __init__(self, step_size: float, L: int, log_prob_func: callable, dim: int, base_sampler: HMC | HMCGaussianAnalytic, model_type: str):
        super().__init__(step_size, L, log_prob_func, dim, base_sampler, model_type)

    def create_surrogate(self, q_init: torch.Tensor, burn: int, epochs: int, use_gradient: bool = False,
                         n_blocks: int = 8):
        """n_blocks sets the depth of the underlying symplectic net. Note the
        time-symmetric wrapper applies that net twice, so a Rev model with
        n_blocks has the same *effective* depth as a plain model with
        2*n_blocks while carrying only half the parameters."""
        param_examples, momenta_examples, grad_examples, _ = self.base_sampler.sample(q_init, num_samples=burn)
        assert n_blocks % 2 == 0, "n_blocks must be even (alternating up/down)"
        modes = ["up", "down"] * (n_blocks // 2)
        model = (
            # 2 gradient blocks = 2 shears, which provably cannot represent a
            # rotation (the exact Gaussian flow); leapfrog itself needs 3.
            SymplecticNeuralNetwork(dim=self.dim * 2, activation_modes=modes, channels=[8] * n_blocks)
            if self.model_type in ("LA", "RevLA")
            else GSymplecticNeuralNetwork(dim=self.dim * 2, activation_modes=modes,
                                          widths=[self.dim * 100] * n_blocks)
        )
        if self.model_type.startswith("Rev"):
            # train and propose with the exactly momentum-reversible composition
            model = TimeSymmetricSymplectic(model)
        input_trajectories = torch.cat([param_examples, momenta_examples], dim=2).detach()
        gradient_traj = None
        # RMHMC base samplers return no gradient trajectories (dp/dt is not
        # grad log p for a non-separable Hamiltonian), so fall back silently
        if use_gradient and grad_examples is not None:
            X, y, t, gradient_traj = create_training_set_symplectic_with_gradients(
                input_trajectories, grad_examples.detach()
            )
        else:
            X, y, t = create_training_set_symplectic(input_trajectories)
        self.model, _ = train_symplectic(model, X=X, y=y, t=t * self.step_size,
                                         epochs=epochs, gradient_traj=gradient_traj)
        self.burn_state = param_examples[-1, -1, :].detach()
    
    def step(self, q, p):
        if self.model is None:
            raise ValueError("Surrogate model is not fit")
        
        initial_positions = torch.cat([q,p])[None,...]
        t = self.L*self.step_size
        with torch.no_grad():
            leapfrog_values = self.model.step(initial_positions, t)
        return leapfrog_values[...,:self.dim], leapfrog_values[...,self.dim:]
        
    
    def sample(self, q_init=None, num_samples=1000):
        return super().sample(q_init, num_samples)
    
    def params_grad(self, *args, **kwargs):
        return super().params_grad(*args, **kwargs)
    
    def gibbs(self, q=None):
        return super().gibbs(q)
    
    def hamiltonian(self, q, p):
        return super().hamiltonian(q, p)
    
    




        



        
    

class RMHMC(HMCBase):
    """Riemannian-manifold HMC with the softabs metric (Betancourt 2013).

    The non-separable Hamiltonian
        H(q, p) = -log p(q) + .5 * log det G(q) + .5 * p^T G(q)^{-1} p
    (G is the softabs-regularized negative Hessian of log p) is integrated
    with Tao's explicit binding integrator (Tao 2016; Cobb et al. 2019) on the
    augmented state (q, p, q_cop, p_cop). Trajectories are returned in the
    same (L+1)-point format as HMC.step so they can train surrogates directly.
    """

    def __init__(self, step_size: float, L: int, log_prob_func: callable, dim: int,
                 softabs_const: float = 1e1, binding_const: float = 100.,
                 jitter: float = None, metric: Metric = Metric.SOFTABS):
        super().__init__(step_size, L, log_prob_func, dim)
        self.softabs_const = softabs_const
        self.binding_const = binding_const
        self.jitter = jitter
        self.metric = metric

    def metric_tensor(self, q):
        G, _ = fisher(q.detach(), self.log_prob_func, jitter=self.jitter,
                      softabs_const=self.softabs_const, metric=self.metric)
        return G.detach()

    def gibbs(self, q=None):
        G = self.metric_tensor(q)
        # symmetrize + jitter so the Cholesky never trips on fp asymmetry
        G = .5 * (G + G.transpose(-1, -2)) + 1e-6 * torch.eye(self.dim, device=G.device, dtype=G.dtype)
        try:
            chol = torch.linalg.cholesky(G)
        except torch.linalg.LinAlgError:
            raise util.LogProbError()
        z = torch.randn(self.dim, device=G.device, dtype=G.dtype)
        return chol @ z

    def hamiltonian(self, q, p):
        return rm_hamiltonian(q, p, self.log_prob_func, self.jitter, 1.,
                              softabs_const=self.softabs_const, sampler=Sampler.RMHMC,
                              integrator=Integrator.IMPLICIT, metric=self.metric)

    def params_grad(self, q, pass_grad=None):
        q = q.detach().requires_grad_()
        log_prob = self.log_prob_func(q)
        q = collect_gradients(log_prob, q, pass_grad)
        return q.grad

    def _dH(self, q, p):
        """Gradients of the non-separable Hamiltonian w.r.t. position and momentum."""
        q = q.detach().requires_grad_()
        p = p.detach().requires_grad_()
        H = self.hamiltonian(q, p)
        dHdq, dHdp = torch.autograd.grad(H, (q, p))
        return dHdq, dHdp

    def step(self, q, p, *args, **kwargs):
        eps = self.step_size
        angle = torch.as_tensor(2. * self.binding_const * eps, dtype=q.dtype, device=q.device)
        c, s = torch.cos(angle), torch.sin(angle)
        q, p = q.detach(), p.detach()
        q_cop, p_cop = q.clone(), p.clone()
        ret_params = [q.clone()]
        ret_momenta = [p.clone()]
        # the exact field (dq/dt, dp/dt) = (dH/dp, -dH/dq) at each stored point
        # is nearly-free training signal (one extra eval on top of six per step)
        dHdq0, dHdp0 = self._dH(q, p)
        ret_fields = [torch.cat([dHdp0, -dHdq0], -1)]
        for n in range(self.L):
            # phi_A^{eps/2}: H(q, p_cop) updates (p, q_cop)
            dHdq, dHdp = self._dH(q, p_cop)
            p = p - .5 * eps * dHdq
            q_cop = q_cop + .5 * eps * dHdp
            # phi_B^{eps/2}: H(q_cop, p) updates (q, p_cop)
            dHdq, dHdp = self._dH(q_cop, p)
            q = q + .5 * eps * dHdp
            p_cop = p_cop - .5 * eps * dHdq
            # phi_C^{eps}: rotate difference coordinates by 2 * omega * eps
            q_sum, q_diff = q + q_cop, q - q_cop
            p_sum, p_diff = p + p_cop, p - p_cop
            q = .5 * (q_sum + c * q_diff + s * p_diff)
            p = .5 * (p_sum - s * q_diff + c * p_diff)
            q_cop = .5 * (q_sum - c * q_diff - s * p_diff)
            p_cop = .5 * (p_sum + s * q_diff - c * p_diff)
            # phi_B^{eps/2}
            dHdq, dHdp = self._dH(q_cop, p)
            q = q + .5 * eps * dHdp
            p_cop = p_cop - .5 * eps * dHdq
            # phi_A^{eps/2}
            dHdq, dHdp = self._dH(q, p_cop)
            p = p - .5 * eps * dHdq
            q_cop = q_cop + .5 * eps * dHdp
            ret_params.append(q.clone())
            ret_momenta.append(p.clone())
            dHdq_n, dHdp_n = self._dH(q, p)
            ret_fields.append(torch.cat([dHdp_n, -dHdq_n], -1))
        return torch.stack(ret_params, axis=0), torch.stack(ret_momenta, axis=0), torch.stack(ret_fields, axis=0)

    def sample(self, q_init, grad_func=None, num_samples=1000):
        """Returns parameter and momentum trajectories (no gradient
        trajectories for the non-separable Hamiltonian) plus acceptances."""
        device = q_init.device
        params = q_init.clone().detach()
        param_burn_prev = q_init.clone().detach()
        num_rejected = 0
        accepted = []
        param_trajectories = []
        momentum_trajectories = []
        field_trajectories = []
        util.progress_bar_init('Sampling ({}; {})'.format("RMHMC", "Explicit (Tao)"), num_samples, 'Samples')
        for n in range(num_samples):
            util.progress_bar_update(n)
            try:
                momentum = self.gibbs(params)
                ham = self.hamiltonian(params, momentum)
                leapfrog_params, leapfrog_momenta, leapfrog_fields = self.step(params, momentum)
                param_trajectories.append(leapfrog_params)
                momentum_trajectories.append(leapfrog_momenta)
                field_trajectories.append(leapfrog_fields)
                params = leapfrog_params[-1].to(device).detach()
                momentum = leapfrog_momenta[-1].to(device).detach()
                new_ham = self.hamiltonian(params, momentum)
                if self.metropolis_accept_step(ham, new_ham):
                    param_burn_prev = params.clone()
                    accepted.append(1.0)
                else:
                    num_rejected += 1
                    params = param_burn_prev.clone()
                    accepted.append(0.0)
            except util.LogProbError:
                num_rejected += 1
                params = param_burn_prev.clone()
                accepted.append(0.0)
        util.progress_bar_end('Acceptance Rate {:.2f}'.format(1 - num_rejected / num_samples))
        return (torch.stack(param_trajectories, axis=0), torch.stack(momentum_trajectories, axis=0),
                torch.stack(field_trajectories, axis=0), torch.Tensor(accepted))


class SurrogateNeuralODERMHMC(SurrogateNeuralODEHMC):
    """Neural-ODE surrogate for RMHMC.

    Learns the non-separable Hamiltonian vector field from RMHMC burn-in
    trajectories and integrates it with Tao's explicit integrator on the
    augmented state, so the surrogate flow stays symplectic. Momentum
    resampling and the Metropolis correction use the exact Riemannian
    Hamiltonian of the base sampler, keeping the chain valid.
    """

    def __init__(self, step_size: float, L: int, log_prob_func: callable, dim: int,
                 base_sampler: RMHMC, model_type: str = ""):
        super().__init__(step_size, L, log_prob_func, dim, base_sampler, model_type)

    def create_surrogate(self, q_init: torch.Tensor, burn: int, epochs: int,
                         solver=None, sensitivity: str = "autograd"):
        param_examples, momenta_examples, field_examples, _ = self.base_sampler.sample(q_init, num_samples=burn)
        if solver is None:
            solver = NonSeparableSynchronousLeapfrog(binding_const=self.base_sampler.binding_const)
        if self.model_type == "explicit_hamiltonian":
            model = RMHNNODE(RMHNN(RMHNNEnergyExplicit(input_dim=self.dim, hidden_dim=100 * self.dim)),
                             solver=solver, sensitivity=sensitivity)
        else:
            model = NNODEgRMHMC(RMHNNEnergyDeriv(input_dim=2 * self.dim, hidden_dim=100 * self.dim),
                                solver=solver, sensitivity=sensitivity)
        X = torch.cat([param_examples[:, 0, :], momenta_examples[:, 0, :]], dim=1).detach()
        # augmented initial state (q, p, q_cop, p_cop) with copies equal to the state
        X_aug = torch.cat([X, X], dim=1)
        y = torch.cat([param_examples, momenta_examples], dim=2).detach()
        t = torch.linspace(start=0, end=self.L * self.step_size, steps=self.L + 1)
        gradient_traj = field_examples.detach() if field_examples is not None else None
        self.model, _ = train_ode(model, X_aug, y, t, epochs=epochs,
                                  gradient_traj=gradient_traj, gradient_mode="full")
        self.burn_state = param_examples[-1, -1, :].detach()

    def step(self, q, p):
        if self.model is None:
            raise ValueError("Surrogate model is not fit")
        initial_positions = torch.cat([q, p, q, p])[None, ...]
        t = torch.linspace(start=0, end=self.L * self.step_size, steps=self.L + 1)
        with torch.no_grad():
            _, leapfrog_values = self.model.forward(initial_positions, t)
        return torch.squeeze(leapfrog_values[..., :self.dim]), torch.squeeze(leapfrog_values[..., self.dim:2 * self.dim])

    def gibbs(self, q=None):
        return self.base_sampler.gibbs(q)

    def hamiltonian(self, q, p):
        return self.base_sampler.hamiltonian(q, p)


class SymplecticRMHMC(SymplecticHMC):
    """Symplectic-network surrogate trained on RMHMC trajectories.

    SNNs approximate arbitrary symplectic maps, so training is identical to
    the separable case; only momentum resampling and the Metropolis
    correction change, delegating to the Riemannian base sampler.
    """

    def gibbs(self, q=None):
        return self.base_sampler.gibbs(q)

    def hamiltonian(self, q, p):
        return self.base_sampler.hamiltonian(q, p)
