from miga.types cimport *
from miga.Population cimport Population, make_population
import multiprocessing
import numpy as np

np_seq_t = np.int32
np_index_t = np.int32
np_data_t = np.float32

cdef class MIGA:
    """Mutual Information Genetic Algorithm main class.

    This class stores the current state of the genetic algorithm and dispatch its
    calculations to the selected platform."""

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

    def __dealloc__(self):
        del self._population

    def __init__(self):
        """Constructs the object and initialize it to the default state."""
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
        """Number of threads to be used by multithreading platforms.

        There must be at least 1 thread. This attribute is innefective when using any
        GPU platform."""

        return self._threads

    @threads.setter
    def threads(self, value):
        if value <= 0:
            raise ValueError("Threads must be > 0")

        self._threads = value

    @property
    def mutation(self):
        """Genetic algorithm mutation rate.

        This rate indicates the proportion of swaps (based on genome size) that will be
        applied to each entity between the generations. Mutations are not applied to
        elite entities. Mutation rate must be a non-zero value.

        See also
        --------
        :attr:`genome`
        :attr:`elite`"""

        return self._mutation

    @mutation.setter
    def mutation(self, value):
        if value <= 0:
            raise ValueError("Mutation must be > 0")

        self._mutation = value

    @property
    def death(self):
        """Genetic algorithm death rate.

        This rate indicates the proportion of entities (based on population size) that
        will be killed between the generations. Worst fitness entities are killed first.
        Elite entities never get killed. Death rate must be a value in the range [0, 1).

        See also
        --------
        :attr:`pop_size`
        :attr:`minimize`
        :attr:`elite`"""

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
        """Genetic algorithm elite ratio.

        This ratio indicates the proportion of entities (based on population size) that
        will never be killed or mutated between generations. Elite ratio must be a value
        in the range [0, 1).

        See also
        --------
        :attr:`pop_size`
        :attr:`mutation`
        :attr:`death`
        :attr:`minimize`"""

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
        """Population genome.

        Encoded genome for each entity. Each number indicates to which sequence in group
        B the current index is paired. Genomes are randomly initialized when
        :attr:`set_msa` is called or when :attr:`pop_size` is increased. When setting
        genome to another matrix, shape must be preserved.

        See also
        --------
        :attr:`pop_size`
        :func:`set_msa`"""

        return self._genome.copy()

    @genome.setter
    def genome(self, value):
        self._genome[:] = value

    @property
    def pop_size(self):
        """Population size.

        Number of entities used by the genetic algorithm. There must be at least one
        entity.

        See also
        --------
        :attr:`elite`
        :attr:`death`
        :attr:`genome`"""

        return self._genome.shape[0]

    @pop_size.setter
    def pop_size(self, value):
        if value < 1:
            raise ValueError("There must be at least one individual")

        if value != self.pop_size:
            self.__resize(value)

    def set_msa(self, seq_a, seq_b):
        """Set MSA used to do MI calculations.

        Set the MSA that represents the two groups used in the calculation and
        initialize the genomes of all population to a random state.

        Parameters
        ----------
        seq_a : numpy.ndarray
            First group encoded MSA. All entries must be positive and lower then
            :attr:`q`.
        seq_b : numpy.ndarray
            Second group encoded MSA. All entries must be positive and lower then
            :attr:`q`.

        See also
        --------
        :attr:`genome`
        :attr:`seq_a`
        :attr:`seq_b`
        :attr:`q`"""

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
        """Read-only copy of encoded MSA representing the first group.

        See also
        --------
        :func:`set_msa`"""

        return self._seq_a.T.copy()

    @property
    def seq_b(self):
        """Read-only copy of encoded MSA representing the second group.

        See also
        --------
        :func:`set_msa`"""

        return self._seq_b.T.copy()

    @property
    def fitness(self):
        """Population fitness

        Read-only copy of of the array containing each entity fitness value. This
        array is first initialized with zeros.

        See also
        --------
        :func:`run`"""

        return self._fitness.copy()

    @property
    def minimize(self):
        """Optimization target.

        `True` if the genetic algorithm must minimize the fitness, `False` otherwise.

        See also
        --------
        :attr:`fitness`
        :func:`run`"""

        return self._minimize

    @minimize.setter
    def minimize(self, value):
        self._minimize = value

    @property
    def q(self):
        """Number of symbols in the MSA.

        Total number of possible symbols used to encode both MSAs. Must be a non-zero
        value.

        See also
        --------
        :attr:`seq_a`
        :attr:`seq_b`
        :func:`set_msa`"""

        return self._q

    @q.setter
    def q(self, value):
        if value < 1:
            raise ValueError("q must be >= 1")

        self._q = value

    @property
    def lambda_(self):
        """Mutual information pseudocounter parameter.

        Parameter used by the pseudocounter in the mutual information calculations.

        See also
        --------
        :attr:`q`
        :func:`run`"""

        return self._lambda

    @lambda_.setter
    def lambda_(self, value):
        if value <= 0:
            raise ValueError("lambda must be > 0")

        self._lambda = value

    @property
    def platform(self):
        """Platform which will be used by the genetic algorithm.

        Platform where the genetic algorithm calculations will run. Possible values are
        'CPU', 'GPU' and 'SimpleGPU'. Platforms 'GPU' and 'SimpleGPU' will be available
        only if the package was compiled with CUDA support. Platform 'SimpleGPU' is a
        non-optimized reference platform to help on writing new platforms for GPU use.

        See also
        --------
        :func:`run`"""

        return self._population.platform().decode("UTF-8")

    @platform.setter
    def platform(self, platform):
        cdef Population *new_population = make_population(platform.encode("UTF-8"))

        if new_population == NULL:
            raise ValueError("Platform {} not available.".format(platform))

        self._population = new_population

    def __resize(self, new_size):
        """Resize the population and initialize or copy genomes if necessary."""

        old_size = self._genome.shape[0]
        old_genome = self._genome
        old_fitness = self._fitness
        num_seqs = self._seq_a.shape[1] # Transposed!
        index = min(new_size, old_size)

        new_genome = np.require(
            np.empty((new_size, num_seqs), np_index_t, "C"),
            np_index_t,
            ("C", "W", "O")
        )

        new_fitness = np.require(
            np.zeros(new_size, np_data_t, "C"),
            np_data_t,
            ("C", "W", "O")
        )

        if old_size != 0:
            # Copy back old genomes and fitness
            new_genome[:index, :] = old_genome[:index, :]
            new_fitness[:index] = old_fitness[:index]

        # Apply shuffled genomes to new individuals
        sample_genome = np.arange(num_seqs, dtype=np_index_t)
        for i in range(index, new_size):
            np.random.shuffle(sample_genome)
            new_genome[i, :] = sample_genome

        self._genome = new_genome
        self._fitness = new_fitness

    def __clear_population(self):
        """Empty population-based attributes."""

        self._genome = np.empty((0, 0), np_index_t, "C")
        self._fitness = np.empty(0, np_data_t, "C")

    def __update_population(self):
        """Set current state on the platform object."""

        self._population.set_q(self._q)
        self._population.set_lambda(self._lambda)
        self._population.set_threads(self._threads)

        cdef index_t num_seqs = self._seq_a.shape[1]
        cdef seq_t[:, :] seq_a = self._seq_a
        cdef seq_t[:, :] seq_b = self._seq_b
        self._population.set_msa(
            num_seqs,
            &seq_a[0, 0],
            seq_a.shape[0],
            &seq_b[0, 0],
            seq_b.shape[0]
        )

        cdef index_t pop_size = self.pop_size
        cdef index_t[:, :] genome = self._genome
        self._population.set_genome(&genome[0, 0], pop_size)

        cdef data_t[:] fitness = self._fitness
        self._population.set_fitness(&fitness[0])


    def run(self, index_t generations):
        """Run the genetic algorithm.

        Run the genetic algorithm for `generations` generations with the current state
        (e.g. population, MSA, platform). In each generation the following steps are
        repeated:

        1. Entities will be sorted based on their fitness (see :attr:`minimize`).
        2. Worst fitness entities will be replaced by copies of the remaining entities (see :attr:`death`).
        3. Non-elite entities genomes will be mutated (see :attr:`elite` and :attr:`mutation`).

        Parameters
        ----------
        generations: :type:`int`
            Number of generations to run the genetic algorithm. If it's 0, population
            will be sorted based on the calculated fitness values."""

        self.__update_population()

        # GA parameters
        cdef index_t pop_size = self.pop_size
        cdef index_t elite = int(self._elite * pop_size)
        cdef index_t surv_num = int(pop_size - self._death * pop_size)
        cdef double mutation = self._mutation
        cdef bool minimize = self._minimize

        # Prepare population to run GA
        self._population.initialize()

        # Reordering population
        self._population.sort(minimize)
        
        # Initiating simulation loop
        cdef index_t n
        for n in range(generations):
            # Selection and reproduction
            self._population.kill_and_reproduce(surv_num, pop_size, 0, surv_num)

            # Mutate non-elite members
            self._population.mutate(mutation, elite, pop_size)

            # Reordering population
            # It's done here to keep the ordering after the simulation
            self._population.sort(minimize)

        # Cleanup population
        self._population.finalize()
