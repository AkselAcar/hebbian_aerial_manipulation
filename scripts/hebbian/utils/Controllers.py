"""Reference (numpy, one drone at a time) implementation of the Hebbian controller.

The training and replay scripts do not use this file any more: they use utils/batched_hebbian.py,
which runs the same network and Hebbian rule for all drones at once on the GPU and was checked
against hebbianNNController below. It is kept as the readable reference of the rule.

The original file also contained the other controllers of the swarm code it was adapted from
(static, recurrent, graph networks, active elastic, random walk). None of them is used for the
drones in this task so they were removed and remain in the git history.
"""

import copy

import numpy as np


class Controller(object):
    def __init__(self, n_states, n_actions, gcn_output_dim=0):
        self.n_input = n_states
        self.n_output = n_actions
        self.gcn_output_dim = gcn_output_dim
        self.controller_type = "default"
        self.umax_const = 0.1
        self.wmax = 1.5708 / 2.5

    @staticmethod
    def velocity_commands(state: np.ndarray) -> np.ndarray:
        return np.array([0])

    @staticmethod
    def geno2pheno(genotype: np.array):
        return

    @staticmethod
    def save_geno(path: str):
        return

    @staticmethod
    def load_geno(path: str):
        return


class hebbianNNController(Controller):
    def __init__(self, n_states, n_actions, n_hidden=17):
        super().__init__(n_states, n_actions)
        self.controller_type = "hNN"
        self.dir = './results'
        self.n_hidden = n_hidden

        # weights for the Hebbian neural network, initialized with small random values
        # 2 hidden layers, each with n_hidden neurons, and an output layer with n_actions neurons
        self.weights_l1 = np.random.normal(0, 0.1, (n_hidden, n_states))   # input -> hidden1
        self.weights_l2 = np.random.normal(0, 0.1, (n_hidden, n_hidden))   # hidden1 -> hidden2
        self.weights_out = np.random.normal(0, 0.1, (n_actions, n_hidden)) # hidden2 -> output

        self.lr = 0.1 # mu in the paper, learning rate for the Hebbian update
        n_synapses = n_hidden * n_states + n_hidden * n_hidden + n_actions * n_hidden
        # A, B, C, D for all the weigths initialized above.
        self.A = np.random.normal(0, 0.1, n_synapses)
        self.B = np.random.normal(0, 0.1, n_synapses)
        self.C = np.random.normal(0, 0.1, n_synapses)
        self.D = np.random.normal(0, 0.1, n_synapses)

        self.max_lin_acc = 5.0    # m/s^2
        self.max_body_rate = 2.0  # rad/s
        self.action_scale = np.array([self.max_lin_acc] * 3 + [self.max_body_rate] * 2)

        # Optional weight logging: if True, the weights are stored every step and saved after 6000 steps.
        self.log_weights = False
        self.l1_log = [copy.deepcopy(self.weights_l1)]
        self.l2_log = [copy.deepcopy(self.weights_l2)]
        self.out_log = [copy.deepcopy(self.weights_out)]

    def map_state(self, min_from, max_from, min_to, max_to, state_portion):
        # Linear map of part of the state from [min_from, max_from] to [min_to, max_to].
        return min_to + np.multiply((max_to - min_to), np.divide((state_portion - min_from), (max_from - min_from)))

    def velocity_commands(self, state: np.ndarray) -> np.ndarray:
        """
        Given a state, give an appropriate action

        :param <np.array> state: A single observation of the current state, dimension is (state_dim)
        :return: <np.array> action: A vector of motor inputs
        """
        assert (len(state) == self.n_input), "State does not correspond with expected input size"

        # Position-type features (meters) -> [-1, 1]. Bound is the env's own
        # bounding_box_threshold (5.0m) -- going past it already ends the episode.
        pos_bound = 5.0
        state[0:3]   = self.map_state(-pos_bound, pos_bound, -1, 1, state[0:3])   # load_position
        state[15:18] = self.map_state(-pos_bound, pos_bound, -1, 1, state[15:18]) # own drone position
        state[33:36] = self.map_state(-pos_bound, pos_bound, -1, 1, state[33:36]) # goal_pos_error

        # Velocities -> [-1, 1]. The env config gives no bound; 5 m/s and 10 rad/s are guesses
        # (suggested by Claude) that should be checked against flight data.
        lin_vel_bound = 5.0   # m/s, guess
        ang_vel_bound = 10.0  # rad/s, guess
        state[27:30] = self.map_state(-lin_vel_bound, lin_vel_bound, -1, 1, state[27:30])
        state[30:33] = self.map_state(-ang_vel_bound, ang_vel_bound, -1, 1, state[30:33])

        # state[3:12], state[12:15], state[18:27], state[36:45] are rotation-matrix
        # entries / one-hot ID -- already in [-1,1] or {0,1}, nothing to rescale.

        # Forward pass, tanh on every layer, so the outputs are in [-1, 1].
        x1 = np.tanh(np.dot(self.weights_l1, state))
        x2 = np.tanh(np.dot(self.weights_l2, x1))
        out = np.tanh(np.dot(self.weights_out, x2))

        pre_synaptic = np.hstack((np.tile(state, x1.size),
                                  np.tile(x1, x2.size),
                                  np.tile(x2, out.size)))
        pos_synaptic = np.hstack(( x1.repeat(state.size),x2.repeat(x1.size),out.repeat(x2.size),))

        # Hebbian update of every weight from the activity of the two neurons it connects:
        # dW = lr * (A * pre * post + B * pre + C * post + D), one A, B, C, D per weight.
        weights_delta = self.lr*(self.A*pre_synaptic*pos_synaptic +
                                 self.B*pre_synaptic +
                                 self.C*pos_synaptic +
                                 self.D)

        self.weights_l1 += weights_delta[:self.weights_l1.size].reshape(self.weights_l1.shape)
        self.weights_l2 += weights_delta[self.weights_l1.size:-self.weights_out.size].reshape(self.weights_l2.shape)
        self.weights_out += weights_delta[-self.weights_out.size:].reshape(self.weights_out.shape)

        if self.log_weights:
            self.l1_log.append(copy.deepcopy(self.weights_l1))
            self.l2_log.append(copy.deepcopy(self.weights_l2))
            self.out_log.append(copy.deepcopy(self.weights_out))
            if len(self.l1_log) == 6000:
                np.save(f"{self.dir}/logs_{np.random.randint(1e5)+100}", [self.l1_log,self.l2_log,self.out_log])

        # Scale the outputs to the command limits (accelerations and body rates).
        control_input = out * self.action_scale
        return control_input

    def geno2pheno(self, genotype: np.array):
        g = np.array(genotype, dtype=float)  # a copy, so Hebbian updates never touch the CMA-ES population
        n = self.A.size            # number of synapses (1139)
        n1 = self.weights_l1.size  # 765
        n2 = self.weights_l2.size  # 289

        # first n values: the starting weights, layer by layer
        self.weights_l1 = g[:n1].reshape(self.weights_l1.shape)
        self.weights_l2 = g[n1:n1 + n2].reshape(self.weights_l2.shape)
        self.weights_out = g[n1 + n2:n].reshape(self.weights_out.shape)

        # remaining 4n values: one A, B, C and D per synapse
        self.A, self.B, self.C, self.D = g[n:].reshape(4, n)
    
    

    def load_geno(self, path: str):
        genotype = np.load(path + "/ABCD.npy", allow_pickle=True)
        self.geno2pheno(genotype)

    def save_geno(self, path: str):
        np.save(path + "/ABCD", [[self.A], [self.B], [self.C], [self.D]], allow_pickle=True)