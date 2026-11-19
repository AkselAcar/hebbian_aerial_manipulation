"""Hebbian network for all drones at once, on the GPU.

This is the same controller as hebbianNNController in Controllers.py, but instead of calling a small
numpy network once per drone and per step, every drone of every swarm in a generation is updated
in one batch of tensor operations. Both versions were compared and give the same outputs.

Each drone has its own copy of the network (17-17 hidden units, tanh everywhere):

    s   = observation scaled to roughly [-1, 1]
    x1  = tanh(W1 s),  x2 = tanh(W2 x1),  out = tanh(Wo x2)
    action = out * [5, 5, 5, 2, 2]         (3 accelerations in m/s^2, 2 body rates in rad/s)

After every forward pass each weight is updated with its own evolved Hebbian rule:

    dW = lr * (A * post * pre + B * pre + C * post + D)

where pre / post are the activities of the two neurons the weight connects.

Genome layout (same as Controllers.geno2pheno), with n = 1139 synapses:

    [ start weights (n) | A (n) | B (n) | C (n) | D (n) ]

and inside each block the synapses go W1 (row by row), then W2, then Wo. A genome with only the
n start weights is the static network (A = B = C = D = 0, no plasticity).

Indexing: brain b = agent * n_envs + env, and env j runs genome j % pop_size. So the three drones
of a swarm, and the repeated evaluations of a genome, all start from the same weights and rules,
but each brain then adapts on its own.

Two optional ways to limit the plasticity (both off by default, so older genomes behave as before):
    decay: each step W also moves back towards its evolved start value,
           W += dW - decay * (W - W_start), so a constant push gives a fixed offset
           instead of a weight that keeps drifting.
    gate:  act() can be given a factor in [0, 1] per brain that scales dW, e.g. small when the
           payload is already close to its goal.
"""

import torch


class BatchedHebbian:
    def __init__(self, n_states=45, n_hidden=17, n_actions=5, lr=0.001,
                 max_lin_acc=5.0, max_body_rate=2.0, device="cuda", dtype=torch.float32, decay=0.0):
        self.ns, self.nh, self.na = n_states, n_hidden, n_actions
        self.n1 = n_hidden * n_states          # input -> hidden 1 (765)
        self.n2 = n_hidden * n_hidden          # hidden 1 -> hidden 2 (289)
        self.n3 = n_actions * n_hidden         # hidden 2 -> output (85)
        self.n_syn = self.n1 + self.n2 + self.n3  # 1139
        self.lr = lr
        self.decay = decay            # per step, 0 = off
        self.device, self.dtype = device, dtype

        self.action_scale = torch.tensor([max_lin_acc] * 3 + [max_body_rate] * 2,
                                         device=device, dtype=dtype)

        # Input scaling, same bounds as map_state in Controllers.py: positions / 5 m,
        # linear velocity / 5 m/s, angular velocity / 10 rad/s. Rotation matrices and the
        # one-hot drone ID are already in [-1, 1] and are left as they are.
        div = torch.ones(n_states, device=device, dtype=dtype)
        for sl, b in [((0, 3), 5.0), ((15, 18), 5.0), ((33, 36), 5.0),
                      ((27, 30), 5.0), ((30, 33), 10.0)]:
            div[sl[0]:sl[1]] = b
        self.state_div = div

    # ------------------------------------------------------------------ setup
    def load(self, genomes, n_envs, n_agents):
        """Build one brain per (drone, env) from the population.

        genomes: array (pop_size, D), with D = n_syn (static network) or 5 * n_syn (Hebbian).
        """
        g = torch.as_tensor(genomes, device=self.device, dtype=self.dtype)
        pop, D = g.shape
        if D == self.n_syn:                                   # static network: rules all zero
            g = torch.cat([g, torch.zeros(pop, 4 * self.n_syn, device=self.device, dtype=self.dtype)], 1)
        elif D != 5 * self.n_syn:
            raise ValueError(f"genome length {D} is neither {self.n_syn} nor {5 * self.n_syn}")

        idx = torch.arange(n_envs, device=self.device) % pop   # which genome each env runs
        idx = idx.repeat(n_agents)                              # brain b = agent * n_envs + env
        g = g[idx]                                              # (n_brains, 5n); indexing makes a copy
        self.n_envs, self.n_agents, self.n_brains = n_envs, n_agents, g.shape[0]

        blocks = g.view(self.n_brains, 5, self.n_syn)           # start weights, A, B, C, D
        def split(x):                                           # (n_brains, n) -> the 3 layer matrices
            return (x[:, :self.n1].reshape(-1, self.nh, self.ns),
                    x[:, self.n1:self.n1 + self.n2].reshape(-1, self.nh, self.nh),
                    x[:, self.n1 + self.n2:].reshape(-1, self.na, self.nh))

        self.W1, self.W2, self.Wo = [w.clone() for w in split(blocks[:, 0])]
        self.W_start = [w.clone() for w in (self.W1, self.W2, self.Wo)]   # kept for the decay
        self.A = split(blocks[:, 1])
        self.B = split(blocks[:, 2])
        self.C = split(blocks[:, 3])
        self.D = split(blocks[:, 4])

    # ------------------------------------------------------------------- step
    @torch.no_grad()
    def act(self, obs, update_mask=None, gate=None):
        """One control step for all brains: forward pass, then the Hebbian update.

        obs: (n_brains, n_states). Returns the actions, (n_brains, n_actions).
        update_mask: bool (n_brains,). Brains set to False keep their weights; the training loop
            uses it for swarms that already crashed, so their weights are not updated any more.
        gate: optional float (n_brains,) in [0, 1] that scales this step's update.
        """
        s = obs.to(self.dtype) / self.state_div
        x1 = torch.tanh(torch.bmm(self.W1, s.unsqueeze(-1))).squeeze(-1)
        x2 = torch.tanh(torch.bmm(self.W2, x1.unsqueeze(-1))).squeeze(-1)
        out = torch.tanh(torch.bmm(self.Wo, x2.unsqueeze(-1))).squeeze(-1)

        m = None if update_mask is None else update_mask.to(self.dtype).view(-1, 1, 1)
        g = None if gate is None else gate.to(self.dtype).view(-1, 1, 1)
        # Same rule for the 3 layers; for each weight, pre is the input neuron and post the output neuron.
        for W, k, pre, post in ((self.W1, 0, s, x1), (self.W2, 1, x1, x2), (self.Wo, 2, x2, out)):
            pre_ = pre.unsqueeze(1)     # (b, 1, n_pre)
            post_ = post.unsqueeze(2)   # (b, n_post, 1)
            dW = self.lr * (self.A[k] * post_ * pre_ + self.B[k] * pre_ + self.C[k] * post_ + self.D[k])
            if g is not None:
                dW = dW * g
            if self.decay > 0:
                dW = dW - self.decay * (W - self.W_start[k])
            W += dW if m is None else dW * m

        return out * self.action_scale

    def mean_abs_w1(self, agent=0):
        """Mean |weight| of the first layer over the brains of one drone (for the logs)."""
        a = agent * self.n_envs
        return float(self.W1[a:a + self.n_envs].abs().mean())