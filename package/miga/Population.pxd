from libcpp.string cimport string
from types cimport *

cdef extern from "src/Population.hpp":
    cdef cppclass Population:
        string platform();
        data_t singleFitness(const seq_t index) const;

    cdef Population *make_population(const string platform)
