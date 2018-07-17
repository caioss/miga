#include "CPUPopulation.hpp"
#include <algorithm>
#include <cmath>

CPUPopulation::CPUPopulation()
: _site_prob_a { nullptr },
  _site_prob_b { nullptr }
{
}

CPUPopulation::~CPUPopulation()
{
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

void CPUPopulation::set_msa(const size_t numSeqs, seq_t *seq_a, const size_t ic_a, seq_t *seq_b, const size_t ic_b)
{
    _num_seqs = numSeqs;
    _num_ic_a = ic_a;
    _num_ic_b = ic_b;
    _seq_a = seq_a;
    _seq_b = seq_b;

    update_site_probs();
}

void CPUPopulation::set_genome(size_t *genome, const size_t pop_size)
{
    _pop_size = pop_size;
    _genome = genome;
    reset_changed();
}

void CPUPopulation::set_fitness(data_t *fitness)
{
    _fitness = fitness;
}

void CPUPopulation::reset_changed()
{
}

void CPUPopulation::sort(const bool minimize)
{
    if (_pop_size == 0 || _num_seqs == 0)
    {
        return;
    }

    size_t *indices = new size_t[_pop_size];
    for (size_t i = 0; i < _pop_size; i++)
    {
        indices[i] = i;
    }

    population_fitness();

    if (minimize)
    {
        std::sort(indices, indices + _pop_size, [&](size_t a, size_t b) -> bool {
            return _fitness[a] < _fitness[b];
        });
    }
    else
    {
        std::sort(indices, indices + _pop_size, [&](size_t a, size_t b) -> bool {
            return _fitness[a] > _fitness[b];
        });
    }

    size_t *sorted_genome = new size_t[_pop_size * _num_seqs];
    data_t *sorted_fitness = new data_t[_pop_size];

    for (size_t i = 0; i < _pop_size; i++)
    {
        const size_t old_index { indices[i] };
        size_t *genome { _genome + old_index * _num_seqs };

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

void CPUPopulation::site_prob(const size_t num_ic, const seq_t *msa, data_t *site_prob)
{
    const data_t residual { _lambda / (_lambda * _q + _num_seqs * _q) };
    const data_t scale { 1.0 / (_lambda + _num_seqs) };

    std::fill(site_prob, site_prob + num_ic * _q, 0.0);

    for (size_t ic = 0; ic < num_ic; ++ic)
    {
        for (size_t seq = 0; seq < _num_seqs; ++seq)
        {
            const seq_t aa { msa[ic * _num_seqs + seq] };
            ++site_prob[ic * _q + aa];
        }
    }
    for (size_t ic = 0; ic < num_ic; ++ic)
    {
        for (size_t aa = 0; aa < _q; ++aa)
        {
            const size_t index { ic * _q + aa };
            site_prob[index] *= scale;
            site_prob[index] += residual;
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

void CPUPopulation::initialize()
{
    // We're ready to do the job. Let's go!
}

void CPUPopulation::finalize()
{
    // Everything was as expected. We made it!
}

void CPUPopulation::population_fitness()
{
#pragma omp parallel for schedule(dynamic) num_threads(_num_threads)
    for (size_t i = 0; i < _pop_size; i++)
    {
        _fitness[i] = single_fitness(i);
    }
}

data_t CPUPopulation::single_fitness(const size_t index) const
{
    data_t coupling { 0 };
    uint32_t *pair_count { new uint32_t[_q * _q] };
    const size_t *genome { _genome + index * _num_seqs };

    const data_t residual { _lambda / (_lambda * _q * _q + _num_seqs * _q * _q) };

    for (size_t ic_a = 0; ic_a < _num_ic_a; ++ic_a)
    {
        for (size_t ic_b = 0; ic_b < _num_ic_b; ++ic_b)
        {
            std::fill(pair_count, pair_count + _q * _q, 0);

            for (size_t seq = 0; seq < _num_seqs; ++seq)
            {
                const seq_t aa1 { _seq_a[ic_a * _num_seqs + seq] };
                const seq_t aa2 { _seq_b[ic_b * _num_seqs + genome[seq]] };
                ++pair_count[aa1 * _q + aa2];
            }

            for (size_t aa1 = 0; aa1 < _q; ++aa1)
            {
                for (size_t aa2 = 0; aa2 < _q; ++aa2)
                {
                    const data_t pair_prob { residual + pair_count[aa1 * _q + aa2] / (_num_seqs + _lambda) };
                    const data_t aa1_prob { _site_prob_a[ic_a * _q + aa1] };
                    const data_t aa2_prob { _site_prob_b[ic_b * _q + aa2] };

                    coupling += pair_prob * log(pair_prob / (aa1_prob * aa2_prob));
                }
            }
        }
    }

    return coupling;
}
