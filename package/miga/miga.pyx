from Population cimport Population, make_population

cdef class MIGA:
    cdef Population *population

    def set_platform(self, platform):
        self.population = make_population(platform.encode("UTF-8"))

        if self.population == NULL:
            raise ValueError("Platform {} not available.".format(platform))
