"""Optimizers used to evolve the Hebbian genomes: CMA-ES and OpenAI-ES (implemented later).

The original file also had differential evolution and MAP-Elites; they were never used for the drones 
and were removed. They are still in the git history.
"""

import numpy as np
import copy
import cma


class EA(object):
    """Common interface of the optimizers: x_new holds the population to evaluate next, the training
    script writes the matching f_new (lower is better, so it passes -fitness), then calls
    get_new_genome() to get the next population. Best-so-far values are kept for saving."""

    def __init__(self, params, output_dir="./results"):
        self.x_new = None
        self.x_best_so_far = None
        self.f_best_so_far = None
        self.x = None
        self.f = None
        self.directory_name = output_dir

    def save_results(self):
        np.save(self.directory_name + '/' + 'f_best', np.array(self.f_best_so_far))
        np.save(self.directory_name + '/' + 'x_best', np.array(self.x_best_so_far))

    def save_checkpoint(self):
        self.save_results()
        np.save(self.directory_name + '/' + 'last_x_new', np.array(self.x_new))
        np.save(self.directory_name + '/' + 'last_x', np.array(self.x))
        np.save(self.directory_name + '/' + 'last_f', np.array(self.f))

    def load_checkpoint(self):
        self.load_results()
        self.x_new = np.load(self.directory_name + '/' + 'last_x_new.npy')
        self.x = np.load(self.directory_name + '/' + 'last_x.npy')
        self.f = np.load(self.directory_name + '/' + 'last_f.npy')

    def load_results(self):
        self.f_best_so_far = np.load(self.directory_name + '/' + 'f_best.npy').tolist()
        self.x_best_so_far = np.load(self.directory_name + '/' + 'x_best.npy').tolist()

    @staticmethod
    def get_new_genome(self):
        return


class CMAes(EA):
    es: cma.CMAEvolutionStrategy
    def __init__(self, params, output_dir='./results'):
        super().__init__(params, output_dir)
        self.Np = params['pop_size']  # Number of individuals
        self.D = params['D']  # Dimension
        self.bounds = params['bounds']  # lower/upper bound on design variables

        if 'x0' in params:  # start point given by the training script (zeros or a saved genome)
            solution_seed = np.asarray(params['x0'], dtype=float)
        else:               # otherwise a random start in [-1, 1]
            solution_seed = np.random.uniform(self.bounds[0], self.bounds[1], self.D) / 5
        self.f_new = np.empty(self.Np)

        opts = {
            'popsize': params['pop_size'],
            'bounds': (
                [self.bounds[0]] * self.D, # lower bounds per dimension
                [self.bounds[1]] * self.D, # upper bounds per dimension
	        ),
        }
        self.es = cma.CMAEvolutionStrategy(solution_seed, params['sigma0'], inopts=opts)
        self.x_new = np.array(self.es.ask())

        self.x = copy.deepcopy(self.x_new)
        self.f = np.ones(self.Np) * np.inf

        self.f_best_so_far = []
        self.x_best_so_far = []

    @property
    def sigma(self):
        return float(self.es.sigma)

    def get_new_genome(self):
        # keep track of the best genome seen so far
        pop_best = np.min(self.f_new)
        if self.f_best_so_far == [] or pop_best < self.f_best_so_far[-1]:
            self.f_best_so_far.append(pop_best)
            self.x_best_so_far.append(copy.deepcopy(self.x_new[np.argmin(self.f_new), :].squeeze()))
        else:
            self.f_best_so_far.append(self.f_best_so_far[-1])
            self.x_best_so_far.append(self.x_best_so_far[-1])

        self.es.tell(self.x_new, self.f_new)
        self.x = copy.deepcopy(self.x_new)
        self.f = copy.deepcopy(self.f_new)
        self.x_new = np.array(self.es.ask())   # cma keeps the samples inside the bounds itself
        return self.x_new



class OpenAIES(EA):
    """OpenAI-ES (Salimans et al. 2017).

    There is one search centre, theta. Each generation:
      1. we sample pop_size / 2 noise vectors eps and evaluate the mirrored pairs
         theta + sigma * eps and theta - sigma * eps;
      2. the fitnesses are replaced by their centred ranks in [-0.5, 0.5], so only their order
         matters, and the gradient is estimated as  g = sum_i rank_i * eps_i / (pop_size * sigma);
      3. theta moves up that gradient, with Adam (default) or a plain step theta += lr * g
         ("sgd", as in Najarro & Risi).
    sigma stays fixed and there is no covariance matrix, which keeps it cheap with ~5700 parameters
    (CMA-ES scales with the square of that).
    """

    def __init__(self, params, output_dir="./results"):
        super().__init__(params, output_dir)
        self.Np = params["pop_size"]
        if self.Np % 2:
            raise ValueError("OpenAI-ES uses mirrored pairs: pop_size must be even")
        self.D = params["D"]
        self.sigma = params.get("sigma0", 0.05)          # fixed noise std
        self.lr = params.get("es_lr", 0.01)               # Adam step size (start value)
        self.lr0 = self.lr
        self.lr_schedule = params.get("lr_schedule", "const")      # "const" or "cosine"
        self.lr_total_gens = max(1, int(params.get("lr_total_gens", 1000)))
        self.lr_min = self.lr0 * params.get("lr_min_frac", 0.1)
        self.gen = 0                                                # generations done
        self.weight_decay = params.get("weight_decay", 0.005)
        self.es_optimizer = params.get("es_optimizer", "adam")   # "adam" or "sgd"
        if self.es_optimizer not in ("adam", "sgd"):
            raise ValueError("es_optimizer must be 'adam' or 'sgd'")
        self.beta1, self.beta2, self.adam_eps = 0.9, 0.999, 1e-8

        if "x0" in params:
            self.theta = np.asarray(params["x0"], dtype=float).copy()
        else:
            self.theta = np.zeros(self.D)
        self.m = np.zeros(self.D)   # Adam first moment
        self.v = np.zeros(self.D)   # Adam second moment
        self.t = 0                  # Adam step counter
        self.last_step_norm = 0.0   # size of the last theta update (for logging)

        self.f_new = np.empty(self.Np)
        self.f_best_so_far = []
        self.x_best_so_far = []
        self.x_new = self._ask()
        self.x = copy.deepcopy(self.x_new)
        self.f = np.ones(self.Np) * np.inf

    def _ask(self):
        half = np.random.randn(self.Np // 2, self.D)
        self.eps = np.concatenate([half, -half])          # mirrored noise
        return self.theta + self.sigma * self.eps

    @staticmethod
    def _centered_ranks(f):
        ranks = np.empty(len(f))
        ranks[np.argsort(f)] = np.arange(len(f))
        return ranks / (len(f) - 1) - 0.5                 # in [-0.5, 0.5]

    def get_new_genome(self):
        # keep track of the best genome seen so far (f_new: lower is better)
        pop_best = np.min(self.f_new)
        if self.f_best_so_far == [] or pop_best < self.f_best_so_far[-1]:
            self.f_best_so_far.append(pop_best)
            self.x_best_so_far.append(copy.deepcopy(self.x_new[np.argmin(self.f_new), :].squeeze()))
        else:
            self.f_best_so_far.append(self.f_best_so_far[-1])
            self.x_best_so_far.append(self.x_best_so_far[-1])

        # gradient of the expected fitness (fitness = -f_new) w.r.t. theta
        ranks = self._centered_ranks(-np.asarray(self.f_new))
        grad = ranks @ self.eps / (self.Np * self.sigma)
        grad -= self.weight_decay * self.theta           # L2 penalty, as in the paper

        if self.lr_schedule == "cosine":              # lr0 -> lr_min over lr_total_gens, then stays there
            frac = min(self.gen / self.lr_total_gens, 1.0)
            self.lr = self.lr_min + 0.5 * (self.lr0 - self.lr_min) * (1 + np.cos(np.pi * frac))
        self.gen += 1
        if self.es_optimizer == "sgd":
            step = self.lr * grad                     # plain gradient ascent
        else:                                         # Adam ascent step
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            step = self.lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
        self.theta = self.theta + step
        self.last_step_norm = float(np.linalg.norm(step))

        self.x = copy.deepcopy(self.x_new)
        self.f = copy.deepcopy(self.f_new)
        self.x_new = self._ask()
        return self.x_new

    def save_checkpoint(self):
        super().save_checkpoint()
        np.save(self.directory_name + "/theta", self.theta)  # the ES search centre