#include "CPUPopulation.hpp"
#include <algorithm>
#include <cmath>

typedef std::uniform_int_distribution<index_t> rng_distrib_t;

CPUPopulation::CPUPopulation()
: _contact_size { 0 },
  _changed { nullptr },
  _site_prob_a { nullptr },
  _site_prob_b { nullptr }
{
    std::random_device random_source;
    _rng_engine = std::default_random_engine(random_source());
}

CPUPopulation::~CPUPopulation()
{
    delete[] _changed;
    delete[] _site_prob_a;
    delete[] _site_prob_b;
}

void CPUPopulation::set_q(const seq_t value)
{
    _q = value;
}

void CPUPopulation::set_lambda(const data_t value)
{
    _lambda = value;
}

void CPUPopulation::set_threads(const size_t threads)
{
	_num_threads = threads;
}

void CPUPopulation::set_msa(const index_t numSeqs, seq_t *seq_a, const index_t ic_a, seq_t *seq_b, const index_t ic_b)
{
    _num_seqs = numSeqs;
    _num_ic_a = ic_a;
    _num_ic_b = ic_b;
    _seq_a = seq_a;
    _seq_b = seq_b;

    update_site_probs();
}

void CPUPopulation::set_genome(index_t *genome, const index_t pop_size)
{
    _pop_size = pop_size;
    _genome = genome;
}

void CPUPopulation::set_contacts(index_t *contacts, const index_t size)
{
    _contact_size = size;
    _contacts = contacts;
}

void CPUPopulation::set_fitness(data_t *fitness)
{
    _fitness = fitness;
}

void CPUPopulation::reset_changed()
{
    delete[] _changed;
    _changed = new bool[_pop_size];
    std::fill(_changed, _changed + _pop_size, true);
}

void CPUPopulation::sort(const bool minimize)
{
    if (_pop_size == 0 || _num_seqs == 0)
    {
        return;
    }

    index_t *indices = new index_t[_pop_size];
    for (index_t i = 0; i < _pop_size; i++)
    {
        indices[i] = i;
    }

    population_fitness();

    if (minimize)
    {
        std::sort(indices, indices + _pop_size, [&](index_t a, index_t b) -> bool {
            return _fitness[a] < _fitness[b];
        });
    }
    else
    {
        std::sort(indices, indices + _pop_size, [&](index_t a, index_t b) -> bool {
            return _fitness[a] > _fitness[b];
        });
    }

    index_t *sorted_genome = new index_t[_pop_size * _num_seqs];
    data_t *sorted_fitness = new data_t[_pop_size];

    for (index_t i = 0; i < _pop_size; i++)
    {
        const index_t old_index { indices[i] };
        index_t *genome { _genome + old_index * _num_seqs };

        std::copy(genome, genome + _num_seqs, sorted_genome + i * _num_seqs);
        
        sorted_fitness[i] = _fitness[old_index];

        // _changed was already set to false
        // by populationFitness()
    }

    std::copy(sorted_genome, sorted_genome + _pop_size * _num_seqs, _genome);
    std::copy(sorted_fitness, sorted_fitness + _pop_size, _fitness);

    delete[] indices;
    delete[] sorted_fitness;
    delete[] sorted_genome;
}

void CPUPopulation::site_prob(const index_t num_ic, const seq_t *msa, data_t *site_prob)
{
    const data_t residual { _lambda / _q };
    const data_t scale { data_t(1.0) - _lambda };

    std::fill(site_prob, site_prob + num_ic * _q, 0.0);

    for (index_t ic = 0; ic < num_ic; ++ic)
    {
        for (index_t seq = 0; seq < _num_seqs; ++seq)
        {
            const seq_t aa { msa[ic * _num_seqs + seq] };
            ++site_prob[ic * _q + aa];
        }
    }
    
    for (index_t ic = 0; ic < num_ic; ++ic)
    {
        for (index_t aa = 0; aa < _q; ++aa)
        {
            const index_t index { ic * _q + aa };
            site_prob[index] = scale * (site_prob[index] / _num_seqs) + residual;
        }
    }
}

void CPUPopulation::update_site_probs()
{
    delete[] _site_prob_a;
    delete[] _site_prob_b;

    _site_prob_a = new data_t[_num_ic_a * _q];
    _site_prob_b = new data_t[_num_ic_b * _q];

    site_prob(_num_ic_a, _seq_a, _site_prob_a);
    site_prob(_num_ic_b, _seq_b, _site_prob_b);
}

void CPUPopulation::kill_and_reproduce(const index_t kill_start, const index_t kill_end, const index_t repr_start, const index_t repr_end)
{
    rng_distrib_t rng_distrib(repr_start, repr_end - 1);

    for (index_t index = kill_start; index < kill_end; index++) {
        const index_t parent { rng_distrib(_rng_engine) };
        index_t *genome { _genome + parent * _num_seqs };

        std::copy(
            genome,
            genome + _num_seqs,
            _genome + index * _num_seqs
        );

        _fitness[index] = _fitness[parent];
        _changed[index] = _changed[parent];
    }
}

void CPUPopulation::mutate(const double ratio, const index_t start, const index_t end)
{
    rng_distrib_t rng_distrib(0, _num_seqs - 1);
    const index_t swaps = ratio * _num_seqs;

    for (index_t index = start; index < end; index++) {
        for (index_t n = 0; n < swaps; n++) {
            const index_t i { index * _num_seqs + rng_distrib(_rng_engine) };
            const index_t j { index * _num_seqs + rng_distrib(_rng_engine) };

            if (i == j)
                continue;

            const index_t temp { _genome[i] };
            _genome[i] = _genome[j];
            _genome[j] = temp;

            _changed[index] = true;
        }
    }
}

void CPUPopulation::initialize()
{
    reset_changed();
}

void CPUPopulation::finalize()
{
    // Everything was as expected. We made it!!
}

void CPUPopulation::population_fitness()
{
#pragma omp parallel for schedule(dynamic) num_threads(_num_threads)
    for (index_t i = 0; i < _pop_size; i++)
    {
        if (_changed[i])
        {
            _fitness[i] = single_fitness(i);
            _changed[i] = false;
        }
    }
}

double CPUPopulation::single_fitness(const index_t index) const
{
    double coupling { 0 };
    uint32_t *pair_count { new uint32_t[_q * _q] };
    const index_t *genome { _genome + index * _num_seqs };
    const data_t residual { _lambda / (_q * _q) };

    if (_contact_size > 0)
    {
        for (index_t c_index = 0; c_index < _contact_size; ++c_index)
        {
            index_t ic_a { _contacts[c_index * 2] };
            index_t ic_b { _contacts[c_index * 2 + 1] };
            coupling += compute_coupling(ic_a, ic_b, pair_count, genome, residual);
        }
    }
    else
    {
        for (index_t ic_a = 0; ic_a < _num_ic_a; ++ic_a)
        {
            for (index_t ic_b = 0; ic_b < _num_ic_b; ++ic_b)
            {
                coupling += compute_coupling(ic_a, ic_b, pair_count, genome, residual);
            }
        }
    }

    return coupling;
}

double CPUPopulation::compute_coupling(const index_t ic_a, const index_t ic_b, uint32_t *pair_count, const index_t *genome, const double residual) const
{
    double coupling { 0 };
    std::fill(pair_count, pair_count + _q * _q, 0);

    for (index_t seq = 0; seq < _num_seqs; ++seq)
    {
        const seq_t aa1 { _seq_a[ic_a * _num_seqs + seq] };
        const seq_t aa2 { _seq_b[ic_b * _num_seqs + genome[seq]] };
        ++pair_count[aa1 * _q + aa2];
    }

    for (index_t aa1 = 0; aa1 < _q; ++aa1)
    {
        for (index_t aa2 = 0; aa2 < _q; ++aa2)
        {
            const double pair_prob = (data_t(1.0) - _lambda) * pair_count[aa1 * _q + aa2] / _num_seqs + residual;
            const double aa1_prob { _site_prob_a[ic_a * _q + aa1] };
            const double aa2_prob { _site_prob_b[ic_b * _q + aa2] };

            coupling += pair_prob * log(pair_prob / (aa1_prob * aa2_prob));
        }
    }

    return coupling;
}
