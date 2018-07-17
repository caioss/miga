#ifndef CPUPOPULATION_HPP
#define CPUPOPULATION_HPP

#include "Population.hpp"
#include <random>

class CPUPopulation : public Population {
public:
	CPUPopulation();
    ~CPUPopulation();
    std::string platform() const override { return "CPU"; };
    void set_q(const seq_t value) override;
    void set_lambda(const data_t value) override;
    void set_threads(const size_t threads) override;
    void set_msa(const size_t num_seqs, seq_t *seq_a, const size_t ic_a, seq_t *seq_b, const size_t ic_b) override;
    void set_genome(size_t *genome, const size_t pop_size) override;
    void set_fitness(data_t *fitness) override;
    void initialize() override;
    void finalize() override;
    void sort(const bool minimize) override;
    void kill_and_reproduce(const size_t kill_start, const size_t kill_end, const size_t repr_start, const size_t repr_end) override;
    void mutate(const double ratio, const size_t start, const size_t end) override;

private:
    void reset_changed();
    void population_fitness();
    void update_site_probs();
    void site_prob(const size_t num_ic, const seq_t *msa, data_t *site_prob);
    data_t single_fitness(const size_t index) const;

private:
	size_t _num_threads;
	size_t _pop_size;
    size_t _num_seqs;
    size_t _num_ic_a;
    size_t _num_ic_b;
    seq_t _q;
    data_t _lambda;
    std::default_random_engine _rng_engine;
    bool *_changed;
    size_t *_genome;
    seq_t *_seq_a;
    seq_t *_seq_b;
    data_t *_fitness;
    data_t *_site_prob_a;
    data_t *_site_prob_b;
};

#endif
