from hamiltorch import surrogate_neural_ode_hmc_experiment, surrogate_neural_ode_hmc_sample_size_experiment, surrogate_neural_ode_hmc_sample_size_experiment_analytic
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='experiment type', required=True)
    parser.add_argument('--is_analytic', help='whether to use analytic samples or leapfrog samples', action="store_true")
    parser.add_argument("--device", help="either cpu or mps:0")
    args = vars(parser.parse_args())
    is_analytic = args["is_analytic"]
    experiment = args["experiment"]
    device = args["device"]


    torch.set_default_device(device)

    if experiment == "sample_size":
        if is_analytic:
            surrogate_neural_ode_hmc_sample_size_experiment_analytic()
        else:
            surrogate_neural_ode_hmc_sample_size_experiment()