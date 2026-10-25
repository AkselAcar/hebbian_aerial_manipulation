"""Batched (GPU) version of hebbianNNController.

Runs ALL brains of a generation (n_envs environments x n_agents drones) as
one set of tensor operations, instead of one small numpy call per brain per
step. The maths is identical to Controllers.hebbianNNController:

    state normalisation (same bounds as map_state)
    x1  = tanh(W1 s),  x2 = tanh(W2 x1),  out = tanh(Wo x2)
    dW  = lr * (A*post*pre + B*pre + C*post + D)      (per synapse)
    action = out * [5, 5, 5, 2, 2]

Genome layout is the same as Controllers.geno2pheno:
    [ W0 (n) | A (n) | B (n) | C (n) | D (n) ],  n = number of synapses (1139)
and inside each block the synapses are ordered W1 (row-major), W2, Wout.
A genome of length n (static baseline) means A = B = C = D = 0.

Brain b = agent * n_envs + env.  Env j runs genome j % pop_size, so all
drones and all reps of a genome start from the same weights and rules.
"""

import torch


class BatchedHebbian:
    def __init__(self, n_states=45, n_hidden=17, n_actions=5, lr=0.001,
                 max_lin_acc=5.0, max_body_rate=2.0, device="cuda", dtype=torch.float32):
        self.ns, self.nh, self.na = n_states, n_hidden, n_actions
        self.n1 = n_hidden * n_states          # 765 synapses input -> hidden1
        self.n2 = n_hidden * n_hidden          # 289 synapses hidden1 -> hidden2
        self.n3 = n_actions * n_hidden         # 85 synapses hidden2 -> output
        self.n_syn = self.n1 + self.n2 + self.n3  # 1139
        self.lr = lr
        self.device, self.dtype = device, dtype

        self.action_scale = torch.tensor([max_lin_acc] * 3 + [max_body_rate] * 2,
                                         device=device, dtype=dtype)

        # Per-feature divisor = the map_state bounds in Controllers.py:
        # positions / 5 m, linear velocity / 5 m/s, angular velocity / 10 rad/s,
        # everything else (rotation matrices, one-hot id) unchanged.
        div = torch.ones(n_states, device=device, dtype=dtype)
        for sl, b in [((0, 3), 5.0), ((15, 18), 5.0), ((33, 36), 5.0),
                      ((27, 30), 5.0), ((30, 33), 10.0)]:
            div[sl[0]:sl[1]] = b
        self.state_div = div

    # ------------------------------------------------------------------ setup
    def load(self, genomes, n_envs, n_agents):
        """Build one brain per (agent, env) from the population.

        genomes: array (pop_size, D) with D = n_syn (static) or 5*n_syn.
        """
        g = torch.as_tensor(genomes, device=self.device, dtype=self.dtype)
        pop, D = g.shape
        if D == self.n_syn:                                   # static baseline
            g = torch.cat([g, torch.zeros(pop, 4 * self.n_syn, device=self.device, dtype=self.dtype)], 1)
        elif D != 5 * self.n_syn:
            raise ValueError(f"genome length {D} is neither {self.n_syn} nor {5 * self.n_syn}")

        idx = torch.arange(n_envs, device=self.device) % pop   # genome of each env
        idx = idx.repeat(n_agents)                              # brain b = agent*n_envs + env
        g = g[idx]                                              # (n_brains, 5n), a copy
        self.n_envs, self.n_agents, self.n_brains = n_envs, n_agents, g.shape[0]

        blocks = g.view(self.n_brains, 5, self.n_syn)           # W0, A, B, C, D
        def split(x):                                           # (n_brains, n) -> 3 matrices
            return (x[:, :self.n1].reshape(-1, self.nh, self.ns),
                    x[:, self.n1:self.n1 + self.n2].reshape(-1, self.nh, self.nh),
                    x[:, self.n1 + self.n2:].reshape(-1, self.na, self.nh))

        self.W1, self.W2, self.Wo = [w.clone() for w in split(blocks[:, 0])]
        self.A = split(blocks[:, 1])
        self.B = split(blocks[:, 2])
        self.C = split(blocks[:, 3])
        self.D = split(blocks[:, 4])

    # ------------------------------------------------------------------- step
    @torch.no_grad()
    def act(self, obs, update_mask=None):
        """obs: (n_brains, n_states). Returns actions (n_brains, n_actions).

        update_mask: bool (n_brains,), True = this brain's weights get the
        Hebbian update (False for swarms that already crashed, so their
        weights stay frozen exactly like the numpy loop, which skipped them).
        """
        s = obs.to(self.dtype) / self.state_div
        x1 = torch.tanh(torch.bmm(self.W1, s.unsqueeze(-1))).squeeze(-1)
        x2 = torch.tanh(torch.bmm(self.W2, x1.unsqueeze(-1))).squeeze(-1)
        out = torch.tanh(torch.bmm(self.Wo, x2.unsqueeze(-1))).squeeze(-1)

        m = None if update_mask is None else update_mask.to(self.dtype).view(-1, 1, 1)
        for W, k, pre, post in ((self.W1, 0, s, x1), (self.W2, 1, x1, x2), (self.Wo, 2, x2, out)):
            pre_ = pre.unsqueeze(1)     # (b, 1, n_pre)
            post_ = post.unsqueeze(2)   # (b, n_post, 1)
            dW = self.lr * (self.A[k] * post_ * pre_ + self.B[k] * pre_ + self.C[k] * post_ + self.D[k])
            W += dW if m is None else dW * m

        return out * self.action_scale

    def mean_abs_w1(self, agent=0):
        """Mean |weight| of layer 1 over the brains of one drone (for logging)."""
        a = agent * self.n_envs
        return float(self.W1[a:a + self.n_envs].abs().mean())