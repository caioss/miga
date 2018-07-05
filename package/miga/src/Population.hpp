#ifndef POPULATION_HPP
#define POPULATION_HPP

#include "types.h"
#include <string>

class Population {
public:
    virtual ~Population() {}
    virtual std::string platform() const = 0;
    virtual void set_q(const seq_t value) = 0;
    virtual void set_lambda(const data_t value) = 0;
    virtual void set_threads(const size_t threads) = 0;
    virtual void set_msa(const size_t num_seqs, seq_t *seq_a, const size_t ic_a, seq_t *seq_b, const size_t ic_b) = 0;
    virtual void set_genome(size_t *genome, const size_t pop_size) = 0;
    virtual void set_fitness(data_t *fitness) = 0;

    virtual void sort(const bool minimize) = 0;

};

Population *make_population(const std::string platform);

#endif
