from types cimport *
from Population cimport Population, make_population
import multiprocessing

cdef class MIGA:
    cdef:
        Population *_population
        seq_t _q
        data_t _lambda
        bool _minimize
        size_t _threads
        size_t _pop_size
        double _mutation
        double _elite
        double _death

    def __cinit__(self):
        self._population = make_population("CPU")

    def __init__(self):
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
    def pop_size(self):
        return self._pop_size

    @pop_size.setter
    def pop_size(self, value):
        if value < 1:
            raise ValueError("There must be at least one individual")

        self._pop_size = value

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

    @property
    def fitness(self):
        # TODO
        pass

    def __update_parameters(self):
        self._population.setQ(self._q)
        self._population.setLambda(self._lambda)
        # TODO
        # self._population.setPopulationSize(self._pop_size)
        # self._population.setThreads(self._threads)

    def run(self, size_t generations):
        self.__update_parameters()

        # Reordering population
        # TODO
        # self._population.sort(self._minimize)

        # Groups sizes
        cdef size_t elite = int(self._elite * self._pop_size)
        cdef size_t surv_num = int(self.pop_size - self._death * self.pop_size)
        cdef size_t rep_num = self.pop_size - surv_num
        
        # Initiating simulation loop
        cdef size_t n
        # for n in range(generations):
# TODO

            # Selection and reproduction
            # self._population.kill_and_reproduce(surv_num, self._pop_size, 0, surv_num, self._mutation)

            # Mutate non-elite members
            # self._population.mutate(self._mutation, self._elite, surv_num)

            # Reordering population
            # It's done here to keep the ordering after the simulation
            # self._population.sort(self._minimize)
