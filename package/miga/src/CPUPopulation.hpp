#ifndef CPUPOPULATION_HPP
#define CPUPOPULATION_HPP

#include "Population.hpp"

class CPUPopulation : public Population {
public:
	CPUPopulation();
    ~CPUPopulation();
    std::string platform() const { return "CPU"; };
    void set_q(const seq_t value);
    void set_lambda(const data_t value);
    void set_threads(const size_t threads);
    void set_msa(const size_t num_seqs, seq_t *seq_a, const size_t ic_a, seq_t *seq_b, const size_t ic_b);
    void set_genome(size_t *genome, const size_t pop_size);
    void set_fitness(data_t *fitness);
    void sort(const bool minimize);

private:
    void reset_changed();
    void population_fitness();
    void update_site_probs();
    void site_prob(const size_t num_ic, const seq_t *msa, data_t *site_prob);
    data_t single_fitness(const size_t index) const;

private:
	size_t _num_threads;
	size_t _pop_size;
    seq_t _q;
    size_t _num_seqs;
    size_t _num_ic_a;
    size_t _num_ic_b;
    data_t _lambda;
    bool *_changed;
    size_t *_genome;
    seq_t *_seq_a;
    seq_t *_seq_b;
    data_t *_fitness;
    data_t *_site_prob_a;
    data_t *_site_prob_b;
};

#endif
