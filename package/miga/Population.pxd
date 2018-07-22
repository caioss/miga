from libcpp.string cimport string
from types cimport *

cdef extern from "src/Population.hpp":
    cdef cppclass Population:
        string platform()
        void set_q(const seq_t value)
        void set_lambda(const data_t value)
        void set_threads(const size_t threads)
        void set_msa(const index_t num_seqs, seq_t *seq_a, const index_t ic_a, seq_t *seq_b, const index_t ic_b)
        void set_genome(index_t *genome, const index_t pop_size)
        void set_fitness(data_t *fitness)

        void initialize() except +
        void finalize() except +
        void sort(const bool minimize)
        void kill_and_reproduce(const index_t kill_start, const index_t kill_end, const index_t repr_start, const index_t repr_end)
        void mutate(const double ratio, const index_t start, const index_t end)

    cdef Population *make_population(const string platform)
