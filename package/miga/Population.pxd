from libcpp.string cimport string

cdef extern from "src/Population.hpp":
    cdef cppclass Population:
        pass

    cdef Population *make_population(const string platform)
