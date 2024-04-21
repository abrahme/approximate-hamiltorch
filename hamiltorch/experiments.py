import torch
import numpy as np
import hamiltorch
import arviz as az
from hamiltorch.hmc import HMC, HMCGaussianAnalytic, SymplecticHMC, SurrogateGradientHMC, SurrogateNeuralODEHMC
from hamiltorch.ode import SynchronousLeapfrog
from hamiltorch.plot_utils import plot_reversibility, plot_samples
from hamiltorch.experiment_utils import banana_log_prob, gaussian_log_prob, high_dimensional_gaussian_log_prob, compute_reversibility_error, params_grad, normal_normal_conjugate, compute_hamiltonian_error
from arviz import ess
import pandas as pd
import time



hamiltorch.set_random_seed(13)

experiment_hyperparams = {
    "banana": {"step_size": .1, "L":5 , "burn": 3000, "N": 6000 , "params_init": torch.Tensor([0.,100.
                                                                                                ]), "log_prob": banana_log_prob,
                                                                                                "grad_func": lambda p: params_grad(p, banana_log_prob)},
        "gaussian": {"step_size":.3, "L":5, "burn": 1000, "N": 2000, "params_init": torch.zeros(3), "log_prob": gaussian_log_prob,
                     "grad_func": lambda p: params_grad(p, gaussian_log_prob)
                      },
        "high_dimensional_gaussian": {"step_size": .1, "L":5 , "burn": 3000 , "N": 6000 , "D": 30, "params_init": torch.randn(30), 
                                     "log_prob": lambda omega: high_dimensional_gaussian_log_prob(omega, D = 30),
                                     "grad_func": lambda p: params_grad(p, high_dimensional_gaussian_log_prob)},
        "normal_normal": {"step_size": .1, "L":5 , "burn": 3000 , "N": 6000 , "params_init": torch.ones(2), 
                                     "log_prob": lambda omega: normal_normal_conjugate(omega),
                                     "grad_func": lambda p: params_grad(p, normal_normal_conjugate)}
}


def run_experiment(model_type, sensitivity, distribution, solver, percent = 1, is_analytic = False, a = None):
    hamiltorch.set_random_seed(123)
    print(f"Running experiment for: solver: {solver}, sensitivity: {sensitivity}, distribution: {distribution}, model: {model_type}")
    experiment_params = experiment_hyperparams[distribution]
    log_prob = experiment_params["log_prob"]
    params_init = experiment_params["params_init"]
    dim = params_init.shape[0]
    step_size = experiment_params["step_size"]
    L = experiment_params["L"]
    N = experiment_params["N"]
    burn = experiment_params["burn"]
    if solver == "SynchronousLeapfrog":
        solver = SynchronousLeapfrog()
    if model_type == "HMC":
        sampler = HMC(step_size=step_size, L = L, log_prob_func=log_prob, dim=dim) if not is_analytic else HMCGaussianAnalytic(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, a=a)
        params_hmc, _, _, _ = sampler.sample(q_init=params_init, grad_func=None, num_samples=int(burn*percent)) ## burn-in
        params_hmc, _, _, _ = sampler.sample(q_init=params_hmc[-1, -1, :], grad_func=None, num_samples= N - int(burn*percent))
        def model_func(x, t):
            step_results = sampler.step(x[..., :dim], x[..., dim:])
            return (None, torch.cat([step_results[0], step_results[1]], -1))
        gradient_func = experiment_params["grad_func"]
        return params_hmc, model_func, gradient_func
    elif model_type == "NNgHMC":
        base_sampler = HMC(step_size=step_size, L = L, log_prob_func=log_prob, dim=dim) if not is_analytic else HMCGaussianAnalytic(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, a=a)
        sampler = SurrogateGradientHMC(step_size=step_size, L=L, log_prob_func=log_prob, base_sampler=base_sampler, dim=dim)
        sampler.create_surrogate(q_init=params_init, burn=int(burn*percent), epochs=100)
        params_hmc_surrogate, _, _, _ = sampler.sample(q_init=None, num_samples = N - int(burn*percent))
        
        def model_func(x, t):
            step_results = sampler.step(x[..., :dim], x[..., dim:])
            return (None, torch.cat([step_results[0], step_results[1]], -1))

        return params_hmc_surrogate, model_func, sampler.model
    elif model_type == "NNODEgHMC":
        base_sampler = HMC(step_size=step_size, L = L, log_prob_func=log_prob, dim=dim) if not is_analytic else HMCGaussianAnalytic(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, a=a)
        sampler = SurrogateNeuralODEHMC(step_size=step_size, L = L, log_prob_func=log_prob, dim=dim, base_sampler=base_sampler, model_type="")
        sampler.create_surrogate(q_init=params_init, burn = int(burn*percent), epochs = 100, solver=solver, sensitivity=sensitivity)
        params_hmc_surrogate_ode_nnghmc, _, _, _ = sampler.sample(q_init=None, num_samples = N - int(burn*percent))
        gradient_func = sampler.model.odefunc
        return params_hmc_surrogate_ode_nnghmc, sampler.model, gradient_func
    elif model_type == "Explicit NNODEgHMC":
        base_sampler = HMC(step_size=step_size, L = L, log_prob_func=log_prob, dim=dim) if not is_analytic else HMCGaussianAnalytic(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, a=a)
        sampler = SurrogateNeuralODEHMC(step_size=step_size, L = L, log_prob_func=log_prob, dim=dim, base_sampler=base_sampler, model_type="explicit_hamiltonian")
        sampler.create_surrogate(q_init=params_init, burn = int(burn*percent), epochs = 100, solver=solver, sensitivity=sensitivity)
        params_hmc_surrogate_ode_explicit, _, _, _ = sampler.sample(q_init=None, num_samples = N - int(burn*percent))
        gradient_func = sampler.model.odefunc
        return params_hmc_surrogate_ode_explicit, sampler.model, gradient_func

    elif model_type == "SymplecticNNgHMC":
        base_sampler = HMC(step_size=step_size, L = L, log_prob_func=log_prob, dim=dim) if not is_analytic else HMCGaussianAnalytic(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, a=a)
        sampler = SymplecticHMC(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, base_sampler=base_sampler, model_type="LA")
        sampler.create_surrogate(q_init=params_init, burn = int(burn*percent), epochs = 100)
        params_hmc_surrogate_symplectic_nnghmc, _, _, _ = sampler.sample(num_samples=N-int(burn*percent), q_init=None)

        gradient_func = None

        return params_hmc_surrogate_symplectic_nnghmc, sampler.model, gradient_func
    elif model_type == "GSymplecticNNgHMC":
        base_sampler = HMC(step_size=step_size, L = L, log_prob_func=log_prob, dim=dim) if not is_analytic else HMCGaussianAnalytic(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, a=a)
        sampler = SymplecticHMC(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, base_sampler=base_sampler, model_type="GSymp")
        sampler.create_surrogate(q_init=params_init, burn = int(burn*percent), epochs = 100)
        params_hmc_surrogate_gsymplectic_nnghmc, _, _, _ = sampler.sample(num_samples=N-int(burn*percent), q_init=None)
                                                
        gradient_func = None

        return params_hmc_surrogate_gsymplectic_nnghmc, sampler.model, gradient_func




def surrogate_neural_ode_hmc_experiment():
    distributions = ["banana", "gaussian", "high_dimensional_gaussian", "normal_normal"]
    sensitivities = ["autograd"]
    solvers = ["SynchronousLeapfrog"]
    models = ["HMC", "NNgHMC", "Explicit NNODEgHMC", "NNODEgHMC"]
    error_list = []
    for distribution in distributions:
        for sensitivity in sensitivities:
            for solver in solvers:
                model_dict = {}
                for model in models:
                    
                    start = time.time()
                    
                    experiment_samples, experiment_model, experiment_grad_func = run_experiment(model, sensitivity, distribution, solver)

                    end = time.time()
                    model_dict[model] = {"samples":experiment_samples, "model": experiment_model, "time": end - start}
                    
                true_samples = torch.stack(model_dict["HMC"]["samples"], 0)
                
                hamiltorch.set_random_seed(1)
                num_samples = 100
                initial_momentum = torch.distributions.Normal(0,1).sample(sample_shape = (num_samples, true_samples.shape[-1]))
                initial_positions = true_samples[torch.multinomial(torch.ones(true_samples.shape[0]), num_samples = 100, replacement=False), :]
                initial_conditions = torch.cat([initial_positions, initial_momentum], -1)
                
                for model in model_dict:
                    error_dict = {}
                    step_size = experiment_hyperparams[distribution]["step_size"] 
                    L = experiment_hyperparams[distribution]["L"] 
                    error, forward_traj, backward_traj = compute_reversibility_error(model_dict[model]["model"], initial_conditions,
                                                        t = torch.linspace(0, L * step_size, L ))
                    model_dict[model]["forward"] = forward_traj[0:5, :]
                    model_dict[model]["backward"] = backward_traj[0:5, :]

                    error_dict["model"] = model
                    error_dict["sensitivity"] = sensitivity
                    error_dict["distribution"] = distribution
                    error_dict["solver"] = solver
                    error_dict["reversibility_error"] = error
                    error_dict["time"] = model_dict[model]["time"]
                    # error_dict["acf"] = autocorr(torch.stack(model_dict[model]["samples"],0).numpy()[None, :, :])
                    error_dict["ess"] = ess(az.convert_to_inference_data(torch.stack(model_dict[model]["samples"],0).numpy()[None, : ,: ])).x.mean().values
                    error_list.append(error_dict)

                plot_samples(model_dict, mean = experiment_hyperparams[distribution]["params_init"], distribution_name=distribution)
                plot_reversibility(model_dict, initial_positions,
                                        distribution=distribution)
    pd.DataFrame(error_list).to_csv("../experiments/diagnostic_results.csv", index = False)
    

def surrogate_neural_ode_hmc_sample_size_experiment():
    distributions = ["banana", "gaussian", "high_dimensional_gaussian", "normal_normal"]
    sensitivities = ["autograd"]
    solvers = ["SynchronousLeapfrog"]
    models = ["GSymplecticNNgHMC","SymplecticNNgHMC", "HMC", "NNgHMC", "Explicit NNODEgHMC", "NNODEgHMC"]
    percent_of_warmup = np.linspace(0.1, 1, 10)
    error_list = []
    for percent in percent_of_warmup:
        for distribution in distributions:
            for sensitivity in sensitivities:
                for solver in solvers:
                    model_dict = {}
                    for model in models:
                        
                        start = time.time()
                        
                        experiment_samples, experiment_model, experiment_grad_func = run_experiment(model, sensitivity, distribution, solver, percent)

                        end = time.time()
                        model_dict[model] = {"samples":experiment_samples, "model": experiment_model, "time": end - start}
                        
                    true_samples = model_dict["HMC"]["samples"]
                    
                    hamiltorch.set_random_seed(1)
                    num_samples = 100
                    initial_momentum = torch.distributions.Normal(0,1).sample(sample_shape = (num_samples, true_samples.shape[-1]))
                    initial_positions = true_samples[torch.multinomial(torch.ones(true_samples.shape[0]), num_samples = 100, replacement=False), :]
                    initial_conditions = torch.cat([initial_positions, initial_momentum], -1)
                    
                    for model in model_dict:
                        error_dict = {}
                        step_size = experiment_hyperparams[distribution]["step_size"] 
                        L = experiment_hyperparams[distribution]["L"] 
                        error, forward_traj, backward_traj = compute_reversibility_error(model_dict[model]["model"], initial_conditions,
                                                            t = torch.linspace(0, L * step_size, L ))
                        hamiltonian_error = compute_hamiltonian_error(model_dict[model]["model"], initial_conditions,
                                                            t = torch.linspace(0, L * step_size, L ), 
                                                            log_prob_func=experiment_hyperparams[distribution]["log_prob"])
                        model_dict[model]["forward"] = forward_traj[0:5, :]
                        model_dict[model]["backward"] = backward_traj[0:5, :]

                        error_dict["model"] = model
                        error_dict["training_size"] = percent
                        error_dict["sensitivity"] = sensitivity
                        error_dict["distribution"] = distribution
                        error_dict["solver"] = solver
                        error_dict["step_size"] = experiment_hyperparams[distribution]["step_size"]
                        error_dict["hamiltonian_error"] = hamiltonian_error.detach().numpy()
                        error_dict["reversibility_error"] = error
                        error_dict["time"] = model_dict[model]["time"]
                        # error_dict["acf"] = autocorr(torch.stack(model_dict[model]["samples"],0).numpy()[None, :, :])
                        error_dict["ess"] = ess(az.convert_to_inference_data(torch.stack(model_dict[model]["samples"],0).numpy()[None, : ,: ])).x.mean().values
                        error_list.append(error_dict)

                    plot_samples(model_dict, mean = experiment_hyperparams[distribution]["params_init"], distribution_name=distribution)
                    plot_reversibility(model_dict, initial_positions,
                                        distribution=distribution)
    pd.DataFrame(error_list).to_csv("../experiments/diagnostic_results.csv", index = False)






