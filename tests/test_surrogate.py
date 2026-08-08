import unittest
import torch
import hamiltorch
from hamiltorch.hmc import HMC, RMHMC
from hamiltorch.symplectic import (
    SymplecticNeuralNetwork, GSymplecticNeuralNetwork, TimeSymmetricSymplectic,
)
from hamiltorch.experiment_utils import banana_log_prob, funnel_log_prob, make_gp_regression_log_prob


class LeapfrogTrajectoryTestCase(unittest.TestCase):
    """HMC.step must return synchronized (q, p) pairs including the initial
    state, without changing the leapfrog proposal itself."""

    def setUp(self):
        hamiltorch.set_random_seed(0)
        self.sampler = HMC(step_size=0.1, L=5, log_prob_func=banana_log_prob, dim=2)
        self.q0 = torch.tensor([0., 100.])
        self.p0 = torch.tensor([0.3, -0.2])

    def _grad(self, q):
        q = q.detach().requires_grad_()
        return torch.autograd.grad(banana_log_prob(q), q)[0]

    def test_trajectory_shape_and_initial_state(self):
        qs, ps, gs = self.sampler.step(self.q0.clone(), self.p0.clone())
        self.assertEqual(qs.shape, (6, 2))
        self.assertEqual(ps.shape, (6, 2))
        self.assertEqual(gs.shape, (6, 2))
        self.assertTrue(torch.allclose(qs[0], self.q0))
        self.assertTrue(torch.allclose(ps[0], self.p0))

    def test_endpoint_matches_reference_leapfrog(self):
        qs, ps, _ = self.sampler.step(self.q0.clone(), self.p0.clone())
        q, p = self.q0.clone(), self.p0.clone()
        p = p + 0.05 * self._grad(q)
        for _ in range(5):
            q = q + 0.1 * p
            g = self._grad(q)
            p = p + 0.1 * g
        p = p - 0.05 * g
        self.assertTrue(torch.allclose(qs[-1], q, atol=1e-5))
        self.assertTrue(torch.allclose(ps[-1], p, atol=1e-5))

    def test_momentum_not_mutated_in_place(self):
        p0 = self.p0.clone()
        self.sampler.step(self.q0.clone(), p0)
        self.assertTrue(torch.allclose(p0, self.p0))

    def test_synchronized_pairs_conserve_hamiltonian(self):
        qs, ps, _ = self.sampler.step(self.q0.clone(), self.p0.clone())
        H = lambda q, p: -banana_log_prob(q) + 0.5 * (p ** 2).sum()
        hs = torch.stack([H(qs[i], ps[i]) for i in range(6)]).detach()
        self.assertLess(float((hs - hs[0]).abs().max()), 1e-3)


class TaoIntegratorTestCase(unittest.TestCase):
    """The explicit RMHMC integrator must be exactly reversible on the
    augmented state and approximately conserve the Riemannian Hamiltonian."""

    def setUp(self):
        hamiltorch.set_random_seed(0)
        torch.set_default_dtype(torch.float64)
        self.sampler = RMHMC(step_size=0.05, L=5, log_prob_func=banana_log_prob,
                             dim=2, softabs_const=10.)
        self.q0 = torch.tensor([0., 100.])
        self.p0 = self.sampler.gibbs(self.q0)

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def _tao(self, q, p, q_cop, p_cop):
        r = self.sampler
        eps = r.step_size
        angle = torch.as_tensor(2. * r.binding_const * eps, dtype=q.dtype)
        c, s = torch.cos(angle), torch.sin(angle)
        for _ in range(r.L):
            dHdq, dHdp = r._dH(q, p_cop); p = p - .5 * eps * dHdq; q_cop = q_cop + .5 * eps * dHdp
            dHdq, dHdp = r._dH(q_cop, p); q = q + .5 * eps * dHdp; p_cop = p_cop - .5 * eps * dHdq
            qs_, qd_ = q + q_cop, q - q_cop
            ps_, pd_ = p + p_cop, p - p_cop
            q = .5 * (qs_ + c * qd_ + s * pd_); p = .5 * (ps_ - s * qd_ + c * pd_)
            q_cop = .5 * (qs_ - c * qd_ - s * pd_); p_cop = .5 * (ps_ + s * qd_ - c * pd_)
            dHdq, dHdp = r._dH(q_cop, p); q = q + .5 * eps * dHdp; p_cop = p_cop - .5 * eps * dHdq
            dHdq, dHdp = r._dH(q, p_cop); p = p - .5 * eps * dHdq; q_cop = q_cop + .5 * eps * dHdp
        return q, p, q_cop, p_cop

    def test_exact_augmented_reversibility(self):
        qf, pf, qcf, pcf = self._tao(self.q0.clone(), self.p0.clone(),
                                     self.q0.clone(), self.p0.clone())
        qb, _, _, _ = self._tao(qf, -pf, qcf, -pcf)
        self.assertLess(float((qb - self.q0).abs().max()), 1e-8)

    def test_step_matches_full_augmented_integration(self):
        qs, ps, _ = self.sampler.step(self.q0.clone(), self.p0.clone())
        qf, pf, _, _ = self._tao(self.q0.clone(), self.p0.clone(),
                                 self.q0.clone(), self.p0.clone())
        self.assertTrue(torch.allclose(qs[-1], qf, atol=1e-10))
        self.assertTrue(torch.allclose(ps[-1], pf, atol=1e-10))

    def test_hamiltonian_error_second_order_in_step_size(self):
        # Fixed total time T = 0.25; halving eps should shrink |dH| by ~4x
        # for a second-order integrator. Require at least 3x per halving.
        drifts = []
        for eps, L in [(0.05, 5), (0.025, 10), (0.0125, 20)]:
            r = RMHMC(step_size=eps, L=L, log_prob_func=banana_log_prob,
                      dim=2, softabs_const=10.)
            qs, ps, _ = r.step(self.q0.clone(), self.p0.clone())
            drifts.append(abs(float(r.hamiltonian(qs[-1], ps[-1]).detach())
                              - float(r.hamiltonian(qs[0], ps[0]).detach())))
        self.assertLess(drifts[1] * 3, drifts[0])
        self.assertLess(drifts[2] * 3, drifts[1])


class SympNetInverseTestCase(unittest.TestCase):
    """SympNet blocks are shears with closed-form inverses; the time-symmetric
    composition Psi = (R Phi^{-1} R) Phi must be exactly momentum-reversible."""

    def setUp(self):
        hamiltorch.set_random_seed(3)
        torch.set_default_dtype(torch.float64)
        self.D = 2
        self.z = torch.randn(7, 2 * self.D)
        self.dt = torch.tensor(0.37)
        self.nets = [
            SymplecticNeuralNetwork(dim=2 * self.D, activation_modes=["up", "down"] * 4,
                                    channels=[8, 8] * 4),
            GSymplecticNeuralNetwork(dim=2 * self.D, activation_modes=["up", "down"],
                                     widths=[100, 100]),
        ]

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def _flip(self, z):
        return torch.cat([z[..., :self.D], -z[..., self.D:]], -1)

    def test_inverse_round_trip(self):
        for net in self.nets:
            err = (net.inverse(net.step(self.z, self.dt), self.dt) - self.z).abs().max()
            self.assertLess(float(err), 1e-10)

    def test_time_symmetric_map_exactly_reversible(self):
        for net in self.nets:
            psi = TimeSymmetricSymplectic(net)
            fwd = psi.step(self.z, self.dt)
            back = psi.step(self._flip(fwd), self.dt)
            err = (self._flip(back) - self.z).abs().max()
            self.assertLess(float(err), 1e-10)


class RMHMCFieldStorageTestCase(unittest.TestCase):
    """RMHMC trajectories must carry the exact (dq/dt, dp/dt) field for
    surrogate gradient supervision."""

    def setUp(self):
        hamiltorch.set_random_seed(0)
        torch.set_default_dtype(torch.float64)
        self.sampler = RMHMC(step_size=0.05, L=5, log_prob_func=banana_log_prob,
                             dim=2, softabs_const=10.)

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def test_fields_match_autograd_at_start(self):
        q0 = torch.tensor([0., 100.])
        p0 = self.sampler.gibbs(q0)
        _, _, fields = self.sampler.step(q0, p0)
        self.assertEqual(fields.shape, (6, 4))
        qg, pg = q0.detach().requires_grad_(), p0.detach().requires_grad_()
        H = self.sampler.hamiltonian(qg, pg)
        dHdq, dHdp = torch.autograd.grad(H, (qg, pg))
        self.assertTrue(torch.allclose(fields[0], torch.cat([dHdp, -dHdq], -1), atol=1e-10))


class NewTargetsTestCase(unittest.TestCase):
    """Funnel and GP-regression log probs: finite values, gradients, and
    batch evaluation summing over rows."""

    def test_targets(self):
        for lp, w in [(funnel_log_prob, torch.tensor([0.5, 1.0])),
                      (make_gp_regression_log_prob(30), torch.zeros(3))]:
            wg = w.clone().requires_grad_()
            v = lp(wg)
            grad = torch.autograd.grad(v, wg)[0]
            self.assertTrue(torch.isfinite(v))
            self.assertTrue(torch.isfinite(grad).all())
            vb = lp(w[None, :].repeat(4, 1))
            self.assertLess(abs(float(vb) - 4 * float(v)), 1e-4)


if __name__ == "__main__":
    unittest.main()
