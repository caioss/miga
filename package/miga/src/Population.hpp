#ifndef POPULATION_HPP
#define POPULATION_HPP

#include "types.h"
#include <string>

class Population
{
public:
    virtual ~Population() {}
    virtual std::string platform() const = 0;
    virtual void set_q(const seq_t value) = 0;
    virtual void set_lambda(const data_t value) = 0;
    virtual void set_threads(const size_t threads) = 0;
    virtual void set_msa(const index_t num_seqs, seq_t *seq_a, const index_t ic_a, seq_t *seq_b, const index_t ic_b) = 0;
    virtual void set_genome(index_t *genome, const index_t pop_size) = 0;
    virtual void set_fitness(data_t *fitness) = 0;

    virtual void initialize() = 0;
    virtual void finalize() = 0;
    virtual void sort(const bool minimize) = 0;
    virtual void kill_and_reproduce(const index_t kill_start, const index_t kill_end, const index_t repr_start, const index_t repr_end) = 0;
    virtual void mutate(const double ratio, const index_t start, const index_t end) = 0;

};

Population *make_population(const std::string platform);

#endif
