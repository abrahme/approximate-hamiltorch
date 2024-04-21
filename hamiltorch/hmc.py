import torch
from abc import abstractmethod, ABC
from . import util
from typing import Union
from .models import train, train_ode,train_symplectic,  NNgHMC, HNNEnergyDeriv, HNNEnergyExplicit, HNNODE, HNN, GSymplecticNeuralNetwork, SymplecticNeuralNetwork

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
    def gibbs(self):
        return torch.distributions.Normal(torch.zeros(self.dim), torch.ones(self.dim)).sample()
    
    @abstractmethod
    def hamiltonian(self, q, p):
        return -self.log_prob_func(q) + .5 * torch.square(p).sum()
    
    @classmethod
    def metropolis_accept_step(cls, hamiltonian_old, hamiltonian_new):
        rho = min(0., float(-hamiltonian_new + hamiltonian_old))
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
        ret_params = []
        ret_momenta = []
        ret_grad = []
        p += 0.5 * self.step_size * self.params_grad(q, grad_func)
        for n in range(self.L):
            q = q + self.step_size * p #/normalizing_const
            p_grad = self.params_grad(q, grad_func)
            p += self.step_size * p_grad
            ret_params.append(q.clone())
            ret_momenta.append(p.clone())
            ret_grad.append(p_grad.clone())
        # only need last for Hamiltoninian check (see p.14) https://arxiv.org/pdf/1206.1901.pdf
        ret_momenta[-1] = ret_momenta[-1] - 0.5 * self.step_size * p_grad.clone()
            # import pdb; pdb.set_trace()
        return torch.stack(ret_params,axis = 0),  torch.stack(ret_momenta,axis=0), torch.stack(ret_grad, axis = 0)
    
    def params_grad(self, q, pass_grad):
        q = q.detach().requires_grad_()
        log_prob = self.log_prob_func(q)
        q = collect_gradients(log_prob, q, pass_grad)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return q.grad
    
    def gibbs(self):
        return super().gibbs()
    
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
                momentum = self.gibbs()

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
        self.a = a  ### this is basically the inverse of the diagonal covariance matrix


    def compute_analytical_hamiltonian_path_gaussian(self, hamiltonian: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        computes analytical hamiltonian solutions of the form p^2/a^2 + q^2/b^2 = 1. 
        """
        b = torch.ones(self.dim)
        t = torch.linspace(0, end=self.L*self.step_size, steps=self.L)
        new_a = torch.sqrt(hamiltonian * a * 2)
        new_b = torch.sqrt(hamiltonian * b * 2)
        return torch.hstack([torch.outer(torch.cos(t),new_a),  torch.outer(torch.sin(t), new_b)])

    def compute_analytical_hamiltonian_gradient_gaussian(self, hamiltonian: torch.Tensor,  a: torch.Tensor) -> torch.Tensor:
        b = torch.ones(self.dim)
        t = torch.linspace(0, end=self.L*self.step_size, steps=self.L)
        new_a = torch.sqrt(hamiltonian * a * 2)
        new_b = torch.sqrt(hamiltonian * b * 2)
        return torch.hstack([-torch.outer(torch.sin(t),new_a),  torch.outer(torch.cos(t), new_b)])

    def step(self, q, p):
        ham = self.hamiltonian(q, p)
        leapfrog_params, leapfrog_momenta = torch.hsplit(self.compute_analytical_hamiltonian_path_gaussian(ham, self.a), 2)
        _, gradient_momenta = torch.hsplit(self.compute_analytical_hamiltonian_gradient_gaussian(ham ,self.a))

        return leapfrog_params, leapfrog_momenta, -gradient_momenta
    
    def sample(self, q_init, grad_func=None, num_samples=1000):
        return super().sample(q_init, grad_func, num_samples)

    def params_grad(self, q, pass_grad):
        raise NotImplementedError
    
    def hamiltonian(self, q, p):
        return super().hamiltonian(q, p)
    
    def gibbs(self):
        return super().gibbs()
    
    

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
    
    def gibbs(self):
        return super().gibbs()
    
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
        self.model, _ = train(model, torch.flatten(param_examples, end_dim=1), torch.flatten(grad_examples, end_dim=1), epochs=epochs)
        self.burn_state = param_examples[-1, -1, :]

    def step(self, q, p, grad_func):
        return super().step(q, p, grad_func)
    
    def sample(self, q_init = None, num_samples=1000):
        return super().sample(self.burn_state if q_init is None else q_init, self.model, num_samples)
    
    def params_grad(self, q, pass_grad):
        return super().params_grad(q, pass_grad)
    
    def hamiltonian(self, q, p):
        return super().hamiltonian(q, p)
    
    def gibbs(self):
        return super().gibbs()
    

class SurrogateNeuralODEHMC(SurrogateHMCBase):
    def __init__(self, step_size: float, L: int, log_prob_func: callable, dim: int, base_sampler: Union[HMC, HMCGaussianAnalytic], model_type:str ):
        super().__init__(step_size, L, log_prob_func, dim, base_sampler)
        self.model_type = model_type
    
    def create_surrogate(self, q_init: torch.Tensor, burn:int, epochs: int, solver:str, sensitivity: str):
        param_examples, momenta_examples, grad_examples, _ = self.base_sampler.sample(q_init, num_samples=burn)
        model = HNNODE(HNNEnergyDeriv(input_dim = self.dim, hidden_dim= 100 * self.dim) , solver = solver, sensitivity=sensitivity)
        if self.model_type == "explicit_hamiltonian":
            model = HNNODE(HNN(HNNEnergyExplicit(self.dim, self.dim * 100)), sensitivity=sensitivity, solver = solver)
        self.model, _ = train_ode(model, 
                                  X = torch.cat([param_examples[:, 0, :], momenta_examples[:, 0, :]], dim = 1),
                                  y = torch.cat([param_examples, momenta_examples], dim = 2),
                                    t = torch.linspace(start = 0, end = self.L*self.step_size, steps=self.L), 
                                    epochs=epochs, 
                                    gradient_traj=grad_examples.detach())
        self.burn_state = param_examples[-1, -1, :]
        
    def step(self, q, p):
        if self.model is None:
            raise ValueError("Surrogate model is not fit")
        
        initial_positions = torch.cat([q,p])[None,...]
        t = torch.linspace(start = 0, end = self.L*self.step_size, steps=self.L)
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
                momentum = self.gibbs()

                ham = self.hamiltonian(params, momentum)

                leapfrog_params, leapfrog_momenta = self.step(params, momentum)
                
                param_trajectories.append(leapfrog_params)
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

        return torch.stack(param_trajectories,axis=0), torch.stack(momentum_trajectories,axis=0), None, torch.Tensor(accepted)
    
    def params_grad(self, *args, **kwargs):
        return super().params_grad(*args, **kwargs)
    
    def gibbs(self):
        return super().gibbs()
    
    def hamiltonian(self, q, p):
        return super().hamiltonian(q, p)
    

    
class SymplecticHMC(SurrogateNeuralODEHMC):
    def __init__(self, step_size: float, L: int, log_prob_func: callable, dim: int, base_sampler: HMC | HMCGaussianAnalytic, model_type: str):
        super().__init__(step_size, L, log_prob_func, dim, base_sampler, model_type)

    def create_surrogate(self, q_init: torch.Tensor, burn:int, epochs: int):
        param_examples, momenta_examples, _, _ = self.base_sampler.sample(q_init, num_samples=burn)
        model = SymplecticNeuralNetwork(dim = self.dim * 2, activation_modes=["up","down"], channels=[8,8]) if self.model_type =="LA" else GSymplecticNeuralNetwork(dim = self.dim * 2, activation_modes=["up","down"], widths=[self.dim*100, self.dim * 100])

        self.model, _ = train_symplectic(model, 
                                  X = torch.cat([param_examples[:, 0, :], momenta_examples[:, 0, :]], dim = 1),
                                  y = torch.cat([param_examples, momenta_examples], dim = 2),
                                    t = torch.linspace(start = 0, end = self.L*self.step_size, steps=self.L), 
                                    epochs=epochs)
        self.burn_state = param_examples[-1, -1, :]
    
    def step(self, q, p):
        return super().step(q, p)
    
    def sample(self, q_init=None, num_samples=1000):
        return super().sample(q_init, num_samples)
    
    def params_grad(self, *args, **kwargs):
        return super().params_grad(*args, **kwargs)
    
    def gibbs(self):
        return super().gibbs()
    
    def hamiltonian(self, q, p):
        return super().hamiltonian(q, p)
    
    




        



        
    
