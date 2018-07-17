from libcpp.string cimport string
from types cimport *

cdef extern from "src/Population.hpp":
    cdef cppclass Population:
        string platform()
        void set_q(const seq_t value)
        void set_lambda(const data_t value)
        void set_threads(const size_t threads)
        void set_msa(const size_t num_seqs, seq_t *seq_a, const size_t ic_a, seq_t *seq_b, const size_t ic_b)
        void set_genome(size_t *genome, const size_t pop_size)
        void set_fitness(data_t *fitness)

        void initialize() except +
        void finalize() except +
        void sort(const bool minimize)

    cdef Population *make_population(const string platform)
