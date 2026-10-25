import numpy as np
import random as rand
import copy
import math
import cma
# from ribs.archives import GridArchive
# from ribs.emitters import EvolutionStrategyEmitter
# from ribs.schedulers import Scheduler


class EA(object):
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


class DE(EA):
    def __init__(self, params, output_dir="./results"):
        super().__init__(params, output_dir)
        self.Np = params['pop_size']  # Number of individuals
        self.D = params['D']  # Dimension
        self.Cr = params['CR']  # Crossover probability
        self.F = params['F']  # Mutation factor
        self.bounds = params['bounds']  # lower/upper bound on design variables

        # Initialize the new population array
        self.x_new = np.random.uniform(self.bounds[0], self.bounds[1], (self.Np, self.D))/5
        self.f_new = np.empty(self.Np)

        # Initialize the current population array
        self.x = copy.deepcopy(self.x_new)
        self.f = np.ones(self.Np) * np.inf

        self.f_best_so_far = []
        self.x_best_so_far = []

        self.directory_name = output_dir
        self.fitnesses = []
        self.genomes = []

    def get_new_genome(self):
        pop_best = np.min(self.f_new)  # some book keeping
        if self.f_best_so_far == [] or pop_best < self.f_best_so_far[-1]:
            self.f_best_so_far.append(pop_best)
            self.x_best_so_far.append(copy.deepcopy(self.x_new[np.argmin(self.f_new), :].squeeze()))
        else:
            self.f_best_so_far.append(self.f_best_so_far[-1])
            self.x_best_so_far.append(self.x_best_so_far[-1])

        for i in range(self.Np):
            if self.f_new[i] < self.f[i]:  # update population
                self.f[i] = self.f_new[i]
                self.x[i] = copy.deepcopy(self.x_new[i])

            r0 = i
            while (r0 == i):
                r0 = math.floor(rand.random() * self.Np)
            r1 = r0
            while (r1 == r0 or r1 == i):
                r1 = math.floor(rand.random() * self.Np)
            r2 = r1
            while (r2 == r1 or r2 == r0 or r2 == i):
                r2 = math.floor(rand.random() * self.Np)

            jrand = math.floor(rand.random() * self.D)

            for j in range(self.D):
                if (rand.random() <= self.Cr or j == jrand):
                    # Mutation
                    self.x_new[i][j] = copy.deepcopy(self.x[r0][j] + self.F * (self.x[r1][j] - self.x[r2][j]))
                else:
                    self.x_new[i][j] = copy.deepcopy(self.x[i][j])

        self.x_new = np.clip(self.x_new, self.bounds[0], self.bounds[1])
        return self.x_new


class CMAes(EA):
    es: cma.CMAEvolutionStrategy
    def __init__(self, params, output_dir='./results'):
        super().__init__(params, output_dir)
        self.Np = params['pop_size']  # Number of individuals
        self.D = params['D']  # Dimension
        # self.Cr = params['CR']  # Crossover probability
        # self.F = params['F']  # Mutation factor
        self.bounds = params['bounds']  # lower/upper bound on design variables

        if 'x0' in params:  # start point chosen by the training script
            solution_seed = np.asarray(params['x0'], dtype=float)
        else:               # old behaviour: random start in [-1, 1]
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
        ## CMA Documentation Example:
        # while not self.es.stop():
        #     solutions = self.es.ask()
        #     fitness = [cma.ff.rosen(x) for x in solutions]
        #     self.es.tell(solutions, fitness)
        #     self.es.logger.add() # write data to disc to be plotted
        #     self.es.disp()
        # self.es.result.pretty()

        pop_best = np.min(self.f_new)  # some book keeping
        if self.f_best_so_far == [] or pop_best < self.f_best_so_far[-1]:
            self.f_best_so_far.append(pop_best)
            self.x_best_so_far.append(copy.deepcopy(self.x_new[np.argmin(self.f_new), :].squeeze()))
        else:
            self.f_best_so_far.append(self.f_best_so_far[-1])
            self.x_best_so_far.append(self.x_best_so_far[-1])

        self.es.tell(self.x_new, self.f_new)
        # self.es.disp()
        self.x = copy.deepcopy(self.x_new)
        self.f = copy.deepcopy(self.f_new)
        self.x_new = np.array(self.es.ask())
        # clipped = np.clip(self.x_new, self.bounds[0], self.bounds[1])
        # assert (self.x_new == clipped).all()
        return self.x_new



class OpenAIES(EA):
    """OpenAI-ES (Salimans et al. 2017), same interface as CMAes.

    One search centre `theta`. Every generation:
      1. ask:  sample pop_size/2 noise vectors eps and evaluate the mirrored
               pairs theta + sigma*eps and theta - sigma*eps (antithetic sampling).
      2. tell: turn the fitnesses into centred ranks in [-0.5, 0.5] (so only the
               ORDER of the fitnesses matters, not their scale), estimate the
               gradient  g = sum_i rank_i * eps_i / (pop_size * sigma),
               and move theta uphill, either with Adam (es_optimizer="adam",
               every parameter moves ~es_lr per step) or with a plain gradient
               step theta += es_lr * g (es_optimizer="sgd", as in Najarro & Risi;
               the step shrinks automatically when the gradient is noisy/small).
    Unlike CMA-ES, sigma is fixed and there is no covariance matrix, so the cost
    per generation is O(D) and does not grow with D squared.

    Same convention as CMAes: the training script sets f_new = -fitness
    (lower is better), and x_new holds the population to evaluate next.
    """

    def __init__(self, params, output_dir="./results"):
        super().__init__(params, output_dir)
        self.Np = params["pop_size"]
        if self.Np % 2:
            raise ValueError("OpenAI-ES uses mirrored pairs: pop_size must be even")
        self.D = params["D"]
        self.sigma = params.get("sigma0", 0.05)          # fixed noise std
        self.lr = params.get("es_lr", 0.01)               # Adam step size
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
        # same best-so-far bookkeeping as CMAes (f_new: lower is better)
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


class MAP_elites(EA):
    def __init__(self, params, output_dir='./results'):
        super().__init__(params, output_dir)
        self.x_measures = None
        self.Np = params['pop_size']  # Number of individuals
        self.D = params['D']  # Dimension
        self.bounds = params['bounds']  # lower/upper bound on design variables
        self.N_emitters = params['N_emitters']
        if self.Np % self.N_emitters != 0:
            print(f'WARNING: number of emitter ({self.N_emitters}) is not a divisor of population size ({self.Np})')
            self.N_emitters = 1

        self.N_measures = len(params['measures']['bounds'])
        self.bounds_measure = params['measures']['bounds']
        self.D_measure = params['measures']['D']
        self.measure_funcs = params['measures']['functions']

        self.archive = GridArchive(
            solution_dim=self.D,
            dims=self.D_measure,
            ranges=self.bounds_measure,  # (-1, 1) for x-pos and (-3, 0) for y-vel.
            qd_score_offset=-600,  # See the note below.
        )

        solution_seed = np.random.uniform(self.bounds[0], self.bounds[1], self.D) / 5
        self.emitters = [
            EvolutionStrategyEmitter(
                archive=self.archive,
                x0=np.zeros_like(solution_seed),
                sigma0=1.0,  # Initial step size.
                ranker="2imp",
                bounds= [self.bounds]*params['D'],
                batch_size=int(self.Np/self.N_emitters), # If we do not specify a batch size, the emitter will
                # automatically use a batch size equal to the default
                # population size of CMA-ES.
            ) for _ in range(self.N_emitters)  # Create 5 separate emitters.
        ]

        self.scheduler = Scheduler(self.archive, self.emitters)

        self.f_new = np.empty(self.Np)

        self.x_new = np.array(self.scheduler.ask())

        self.x = copy.deepcopy(self.x_new)
        self.f = np.ones(self.Np) * np.inf

        self.f_best_so_far = []
        self.x_best_so_far = []

    def update_measure(self, measures=None):
        if measures is None:
            measures = []
            for measure_func in self.measure_funcs:
                measures.append(measure_func(self.x_new, self.f_new))
            measures = np.array(measures).T
        self.x_measures = measures

    def get_new_genome(self, measures=None):
        pop_best = np.min(self.f_new)  # some book keeping
        if self.f_best_so_far == [] or pop_best < self.f_best_so_far[-1]:
            self.f_best_so_far.append(pop_best)
            self.x_best_so_far.append(copy.deepcopy(self.x_new[np.argmin(self.f_new), :].squeeze()))
        else:
            self.f_best_so_far.append(self.f_best_so_far[-1])
            self.x_best_so_far.append(self.x_best_so_far[-1])

        self.update_measure(measures)

        self.scheduler.tell(-self.f_new.squeeze(), self.x_measures)
        self.x = copy.deepcopy(self.x_new)
        self.f = copy.deepcopy(self.f_new)
        self.x_new = np.array(self.scheduler.ask())
        self.x_new = np.clip(self.x_new, self.bounds[0], self.bounds[1])
        if True:
            print(f"  - Size: {self.archive.stats.num_elites}")
            print(f"  - Coverage: {self.archive.stats.coverage}")
            print(f"  - QD Score: {self.archive.stats.qd_score}")
            print(f"  - Max Obj: {self.archive.stats.obj_max}")
            print(f"  - Mean Obj: {self.archive.stats.obj_mean}")
        return self.x_new

    def save_results(self):
        np.save(self.directory_name + '/' + 'f_best', np.array(self.f_best_so_far))
        np.save(self.directory_name + '/' + 'x_best', np.array(self.x_best_so_far))
        self.archive.as_pandas().to_csv(self.directory_name + '/' + f"MAP_archive.csv")