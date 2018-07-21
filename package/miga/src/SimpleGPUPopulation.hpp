#ifndef SIMPLEGPUPOPULATION_HPP
#define SIMPLEGPUPOPULATION_HPP

#include "Population.hpp"

class SimpleGPUPopulation : public Population {
public:
	SimpleGPUPopulation();
    ~SimpleGPUPopulation();
    std::string platform() const override { return "SimpleGPU"; };
    void set_q(const seq_t value) override;
    void set_lambda(const data_t value) override;
    void set_threads(const size_t threads) override;
    void set_msa(const index_t num_seqs, seq_t *seq_a, const index_t ic_a, seq_t *seq_b, const index_t ic_b) override;
    void set_genome(index_t *genome, const index_t pop_size) override;
    void set_fitness(data_t *fitness) override;
    void initialize() override;
    void finalize() override;
    void sort(const bool minimize) override;
    void kill_and_reproduce(const index_t kill_start, const index_t kill_end, const index_t repr_start, const index_t repr_end) override;
    void mutate(const double ratio, const index_t start, const index_t end) override;

private:
    void check_device();
    void init_gpu_data();
    void retrieve_data();
    void free_device_memory();
    void population_fitness();
    void update_site_probs();
    void site_prob(const index_t num_ic, const seq_t *msa, data_t *site_prob);

private:
	index_t _pop_size;
    index_t _num_seqs;
    index_t _num_ic_a;
    index_t _num_ic_b;
    seq_t _q;
    data_t _lambda;
    index_t *_genome;
    seq_t *_seq_a;
    seq_t *_seq_b;
    data_t *_fitness;
    data_t *_site_prob_a;
    data_t *_site_prob_b;

    // CUDA variables
    class cudaParams;

    int _warp_size;
    int _threads;
    size_t _sort_bytes;
    cudaParams *_params;
    void *d_sort_buffer;
    index_t *d_genome;
    index_t *d_indices;
    index_t *d_indices_sorted;
    seq_t *d_seq_a;
    seq_t *d_seq_b;
    float *d_fitness;
    float *d_fitness_sorted;
    data_t *d_site_prob_a;
    data_t *d_site_prob_b;
};

#endif
