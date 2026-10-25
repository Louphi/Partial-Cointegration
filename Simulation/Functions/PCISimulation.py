"""Historical generator interface backed by explicit independent RNG streams."""

import numpy as np
from partial_cointegration import Parameters
from partial_cointegration.simulation import simulate_pair


class pci_generator:
    def __init__(
        self,
        N,
        alpha,
        beta,
        rho,
        sigma_M,
        sigma_R,
        sigma_X,
        *,
        seed=0,
        initialization="known",
    ):
        self.N = N
        self.parameters = Parameters(beta, rho, sigma_M, sigma_R, alpha)
        self.sigma_X = sigma_X
        self.seed = seed
        self.initialization = initialization

    def f_generate_pci(self):
        pair = simulate_pair(
            self.N,
            self.parameters,
            seed=self.seed,
            sigma_x=self.sigma_X,
            initialization=self.initialization,
        )
        return pair.x, pair.y, pair.m + pair.r, pair.m, pair.r

    def f_generate_par(self):
        _, _, w, m, r = self.f_generate_pci()
        return w, m, r

    def f_generate_trajectories(self, trajectory=100):
        if trajectory < 1:
            raise ValueError("Need at least one trajectory")
        out = np.empty((self.N, trajectory, 2))
        for i in range(trajectory):
            seed = int(np.random.SeedSequence([self.seed, i]).generate_state(1)[0])
            pair = simulate_pair(
                self.N,
                self.parameters,
                seed=seed,
                sigma_x=self.sigma_X,
                initialization=self.initialization,
            )
            out[:, i, :] = np.column_stack([pair.x, pair.y])
        return out
