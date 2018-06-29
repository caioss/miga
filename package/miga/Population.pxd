from libcpp.string cimport string
from types cimport *

cdef extern from "src/Population.hpp":
    cdef cppclass Population:
        void setQ(const seq_t value)
        void setLambda(const data_t value);
        string platform();

    cdef Population *make_population(const string platform)
