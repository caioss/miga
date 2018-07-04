from miga.types cimport *
from miga.Population cimport Population, make_population
import multiprocessing
import numpy as np

np_seq_t = np.uint32
np_data_t = np.double

cdef class MIGA:
    cdef:
        Population *_population
        seq_t _q
        data_t _lambda
        bool _minimize
        size_t _threads
        double _mutation
        double _elite
        double _death
        object _genome
        object _fitness
        object _seq_a
        object _seq_b

    def __cinit__(self):
        self._population = make_population("CPU")

    def __init__(self):
        self.__clear_population()
        self._seq_a = np.empty((0, 0), np_seq_t, "C")
        self._seq_b = np.empty((0, 0), np_seq_t, "C")

        self.pop_size = 20
        self.mutation = 0.01
        self.elite = 0.1
        self.death = 0.25
        self.minimize = False
        self.q = 21
        self.lambda_ = 0.5
        self.threads = multiprocessing.cpu_count()

    @property
    def threads(self):
        return self._threads

    @threads.setter
    def threads(self, value):
        if value <= 0:
            raise ValueError("Threads must be > 0")

        self._threads = value

    @property
    def mutation(self):
        return self._mutation

    @mutation.setter
    def mutation(self, value):
        if value <= 0:
            raise ValueError("Mutation must be > 0")

        self._mutation = value

    @property
    def death(self):
        return self._death

    @death.setter
    def death(self, value):
        if value > 1 or value < 0:
            raise ValueError("Death must be 0 <= e < 1")

        if 1 - self.elite < value:
            raise ValueError("Elite and death values are incompatible")

        self._death = value

    @property
    def elite(self):
        return self._elite

    @elite.setter
    def elite(self, value):
        if value > 1 or value < 0:
            raise ValueError("Elite must be 0 <= e < 1")

        if 1 - value < self.death:
            raise ValueError("Elite and death values are incompatible")

        self._elite = value

    @property
    def genome(self):
        return self._genome.copy()

    @genome.setter
    def genome(self, value):
        self._genome[:] = value

    @property
    def pop_size(self):
        return self._genome.shape[0]

    @pop_size.setter
    def pop_size(self, value):
        if value < 1:
            raise ValueError("There must be at least one individual")

        if value != self.pop_size:
            self.__resize(value)

    def set_msa(self, seq_a, seq_b):
        if seq_a.shape[0] != seq_b.shape[0]:
            raise ValueError("Number of sequences must be equal on both alignments")

        if seq_a.min() < 0 or seq_a.max() >= self.q:
            raise ValueError("Invalid values in seq_a")

        if seq_b.min() < 0 or seq_b.max() >= self.q:
            raise ValueError("Invalid values in seq_b")

        self._seq_a = np.require(
            seq_a.T.copy(),
            np_seq_t,
            ("C", "W", "O")
        )
        self._seq_b = np.require(
            seq_b.T.copy(),
            np_seq_t,
            ("C", "W", "O")
        )

        pop_size = self.pop_size
        self.__clear_population()
        self.pop_size = pop_size

    @property
    def seq_a(self):
        return self._seq_a.T.copy()

    @property
    def seq_b(self):
        return self._seq_b.T.copy()

    @property
    def fitness(self):
        return self._fitness.copy()

    @property
    def minimize(self):
        return self._minimize

    @minimize.setter
    def minimize(self, value):
        self._minimize = value

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        if value < 1:
            raise ValueError("q must be >= 1")

        self._q = value

    @property
    def lambda_(self):
        return self._lambda

    @lambda_.setter
    def lambda_(self, value):
        if value <= 0:
            raise ValueError("lambda must be > 0")

        self._lambda = value

    @property
    def platform(self):
        return self._population.platform().decode("UTF-8")

    @platform.setter
    def platform(self, platform):
        cdef Population *new_population = make_population(platform.encode("UTF-8"))

        if new_population == NULL:
            raise ValueError("Platform {} not available.".format(platform))

        self._population = new_population

    def __resize(self, new_size):
        old_size = self._genome.shape[0]
        old_genome = self._genome
        old_fitness = self._fitness
        num_seqs = self._seq_a.shape[1] # Transposed!
        index = min(new_size, old_size)

        new_genome = np.require(
            np.empty((new_size, num_seqs), np_seq_t, "C"),
            np_seq_t,
            ("C", "W", "O")
        )

        new_fitness = np.require(
            np.zeros(new_size, np_seq_t, "C"),
            np_data_t,
            ("C", "W", "O")
        )

        if old_size != 0:
            # Copy back old genomes and fitness
            new_genome[:index, :] = old_genome[:index, :]
            new_fitness[:index] = old_fitness[:index]

        # Apply shuffled genomes to new individuals
        sample_genome = np.arange(num_seqs, dtype=np_seq_t)
        for i in range(index, new_size):
            np.random.shuffle(sample_genome)
            new_genome[i, :] = sample_genome

        self._genome = new_genome
        self._fitness = new_fitness

    def __clear_population(self):
        self._genome = np.empty((0, 0), np_seq_t, "C")
        self._fitness = np.empty((0, 0), np_seq_t, "C")

    def __update_parameters(self):
        self._population.setQ(self._q)
        self._population.setLambda(self._lambda)
        # TODO
        # self._population.setThreads(self._threads)

    def run(self, size_t generations):
        self.__update_parameters()

        # Reordering population
        # TODO
        # self._population.sort(self._minimize)

        # Groups sizes
        cdef size_t pop_size = self.pop_size
        cdef size_t elite = int(self._elite * pop_size)
        cdef size_t surv_num = int(pop_size - self._death * self.pop_size)
        cdef size_t rep_num = self.pop_size - surv_num
        
        # Initiating simulation loop
        cdef size_t n
        # for n in range(generations):
# TODO

            # Selection and reproduction
            # self._population.kill_and_reproduce(surv_num, pop_size, 0, surv_num, self._mutation)

            # Mutate non-elite members
            # self._population.mutate(self._mutation, self._elite, surv_num)

            # Reordering population
            # It's done here to keep the ordering after the simulation
            # self._population.sort(self._minimize)
