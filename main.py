from hamiltorch import surrogate_neural_ode_hmc_experiment, surrogate_neural_ode_hmc_sample_size_experiment, surrogate_neural_ode_hmc_sample_size_experiment_analytic
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='experiment type', required=True)
    parser.add_argument('--is_analytic', help='whether to use analytic samples or leapfrog samples', required=True, default=False)
    args = vars(parser.parse_args())
    is_analytic = args["is_analytic"]
    experiment = args["experiment"]

    if experiment == "sample_size":
        if is_analytic:
            surrogate_neural_ode_hmc_sample_size_experiment_analytic()
        else:
            surrogate_neural_ode_hmc_sample_size_experiment()