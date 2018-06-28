from Population cimport Population, make_population

cdef class MIGA:
    cdef Population *population

    def __cinit__(self):
        self.population = make_population("CPU")

    @property
    def platform(self):
        return self.population.platform().decode("UTF-8")

    @platform.setter
    def platform(self, platform):
        cdef Population *new_population = make_population(platform.encode("UTF-8"))

        if new_population == NULL:
            raise ValueError("Platform {} not available.".format(platform))

        self.population = new_population

    def single_fitness(self, index):
        self.population.singleFitness(index)

