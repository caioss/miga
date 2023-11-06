#ifndef CPUPOPULATION_HPP
#define CPUPOPULATION_HPP

#include "Population.hpp"
#include <random>

class CPUPopulation : public Population
{
public:
	CPUPopulation();
    ~CPUPopulation();
    std::string platform() const override { return "CPU"; };
    void set_q(const seq_t value) override;
    void set_lambda(const data_t value) override;
    void set_threads(const size_t threads) override;
    void set_msa(const index_t num_seqs, seq_t *seq_a, const index_t ic_a, seq_t *seq_b, const index_t ic_b) override;
    void set_genome(index_t *genome, const index_t pop_size) override;
    void set_contacts(index_t *contacts, const index_t size) override;
    void set_fitness(data_t *fitness) override;
    void initialize() override;
    void finalize() override;
    void sort(const bool minimize) override;
    void kill_and_reproduce(const index_t kill_start, const index_t kill_end, const index_t repr_start, const index_t repr_end) override;
    void mutate(const double ratio, const index_t start, const index_t end) override;

private:
    void reset_changed();
    void population_fitness();
    void update_site_probs();
    void site_prob(const index_t num_ic, const seq_t *msa, data_t *site_prob);
    double single_fitness(const index_t index) const;
    double compute_coupling(const index_t ic_a, const index_t ic_b, uint32_t *pair_count, const index_t *genome, const double residual) const;

private:
	size_t _num_threads;
	index_t _pop_size;
    index_t _num_seqs;
    index_t _num_ic_a;
    index_t _num_ic_b;
    index_t _contact_size;
    seq_t _q;
    data_t _lambda;
    std::default_random_engine _rng_engine;
    bool *_changed;
    index_t *_genome;
    index_t *_contacts;
    seq_t *_seq_a;
    seq_t *_seq_b;
    data_t *_fitness;
    data_t *_site_prob_a;
    data_t *_site_prob_b;
};

#endif
