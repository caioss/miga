#include "SimpleGPUPopulation.hpp"
#include "cuda_tools.cuh"
#include "cub/cub.cuh"
#include <curand_kernel.h>
#include <random>

class SimpleGPUPopulation::cudaParams
{
public:
    curandState *prng_states;
};

// Kernels associated with SimpleGPUPopulation class
// Rename this namespace when creating a new implementation
namespace simple_kernel {

__global__ void init_prng(curandState *states, const unsigned long int seed)
{
    const int t_idx = threadIdx.x;
    curand_init(seed, t_idx, 0, &states[t_idx]);
}

__global__ void kill_and_reproduce(index_t *pop_genome, const index_t num_seqs, const index_t kill_start, const index_t repr_start, const index_t repr_end, curandState *prng_states)
{
    extern __shared__ int parent_idx[];
    const int t_idx = threadIdx.x;
    const int stride = blockDim.x;
    const int son = kill_start + blockIdx.x;

    if (t_idx == 0)
    {
        const index_t repr_n = repr_end - repr_start;
        // Using integer casts is not a big deal for the current problem
        // and it's faster than ceilf
        parent_idx[0] = curand_uniform(prng_states + son) * repr_n + 1;
    }
    __syncthreads();

    index_t *genome = pop_genome + son * num_seqs;
    index_t *parent_genome = pop_genome + parent_idx[0] * num_seqs;

    for (int i = t_idx; i < num_seqs; i += stride)
    {
        genome[i] = parent_genome[i];
    }
}

__global__ void mutate(index_t *pop_genome, const index_t swaps, const index_t num_seqs, const index_t start, curandState *prng_states)
{
    const int entity = start + threadIdx.x;
    index_t *genome = pop_genome + entity * num_seqs;
    curandState state = prng_states[entity];

    for (int n = 0; n < swaps; ++n) {
        const int i = curand_uniform(&state) * num_seqs;
        const int j = curand_uniform(&state) * num_seqs;

        if (i == j)
            continue;

        const index_t temp { genome[i] };
        genome[i] = genome[j];
        genome[j] = temp;
    }

    prng_states[entity] = state;
}

__global__ void reorder_genome(const index_t *indices, index_t *genome, const index_t num_seqs, const index_t pop_size)
{
    extern __shared__ index_t old_genome[];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_seqs)
    {
        return;
    }

    for (int entity = 0; entity < pop_size; ++entity)
    {
        old_genome[entity * blockDim.x + threadIdx.x] = genome[entity * num_seqs + idx];
    }
    __syncthreads();

    for (int entity = 0; entity < pop_size; ++entity)
    {
        genome[entity * num_seqs + idx] = old_genome[indices[entity] * blockDim.x + threadIdx.x];
    }
}

__global__ void fitness(const data_t residual, const data_t scale, const index_t num_seqs, const seq_t q, const index_t *pop_genome, const seq_t *seq_a, const seq_t *seq_b, float *fitness, const data_t *site_prob_a, const data_t *site_prob_b, const index_t num_ic_a)
{
    extern __shared__ int32_t pair_count[];
    const index_t entity = blockIdx.x;
    const index_t ic_a = blockIdx.y;
	const index_t ic_b = blockIdx.z;
    const int t_idx = threadIdx.x;
    const int stride = blockDim.x;
    const index_t *genome = pop_genome + entity * num_seqs;
    float coupling { 0 };

    // Zero pair_count
    for (int i = t_idx; i < q * q; i += stride)
    {
        pair_count[i] = 0;
    }
    __syncthreads();

    // Pairs counting
    for (int i = t_idx; i < num_seqs; i += stride)
    {
        const seq_t aa1 = seq_a[ic_a * num_seqs + i];
        const seq_t aa2 = seq_b[ic_b * num_seqs + genome[i]];
        atomicAdd(pair_count + aa1 * q + aa2, 1);

    }
    __syncthreads();

    // Coupling
    for (int i = t_idx; i < q * q; i += stride)
    {
        const seq_t aa1 = i / q;
        const seq_t aa2 = i % q;

        const float aa1_prob = site_prob_a[ic_a * q + aa1];
        const float aa2_prob = site_prob_b[ic_b * q + aa2];

        const float pair_prob { residual + pair_count[i] * scale };

        coupling += pair_prob * logf(pair_prob / (aa1_prob * aa2_prob));
    }

    atomicAdd(fitness + entity, coupling);
}

};

SimpleGPUPopulation::SimpleGPUPopulation()
: _site_prob_a { nullptr },
  _site_prob_b { nullptr },
  _params { new cudaParams },

  // CUDA variables
  d_sort_buffer { nullptr },
  d_genome { nullptr },
  d_indices { nullptr },
  d_indices_sorted { nullptr },
  d_seq_a { nullptr },
  d_seq_b { nullptr },
  d_fitness { nullptr },
  d_fitness_sorted { nullptr },
  d_site_prob_a { nullptr },
  d_site_prob_b { nullptr }
{
}

SimpleGPUPopulation::~SimpleGPUPopulation()
{
    delete[] _site_prob_a;
    delete[] _site_prob_b;
    delete[] _params;

    free_device_memory();
}

void SimpleGPUPopulation::free_device_memory()
{
    cudaErr( cudaFree(d_sort_buffer) );
    cudaErr( cudaFree(d_genome) );
    cudaErr( cudaFree(d_indices) );
    cudaErr( cudaFree(d_indices_sorted) );
    cudaErr( cudaFree(d_seq_a) );
    cudaErr( cudaFree(d_seq_b) );
    cudaErr( cudaFree(d_fitness) );
    cudaErr( cudaFree(d_fitness_sorted) );
    cudaErr( cudaFree(d_site_prob_a) );
    cudaErr( cudaFree(d_site_prob_b) );

    d_sort_buffer = nullptr;
    d_genome = nullptr;
    d_indices = nullptr;
    d_indices_sorted = nullptr;
    d_seq_a = nullptr;
    d_seq_b = nullptr;
    d_fitness = nullptr;
    d_fitness_sorted = nullptr;
    d_site_prob_a = nullptr;
    d_site_prob_b = nullptr;
}

void SimpleGPUPopulation::set_q(const seq_t value)
{
    _q = value;
}

void SimpleGPUPopulation::set_lambda(const data_t value)
{
    _lambda = value;
}

void SimpleGPUPopulation::set_threads(const size_t threads)
{
}

void SimpleGPUPopulation::set_msa(const index_t numSeqs, seq_t *seq_a, const index_t ic_a, seq_t *seq_b, const index_t ic_b)
{
    _num_seqs = numSeqs;
    _num_ic_a = ic_a;
    _num_ic_b = ic_b;
    _seq_a = seq_a;
    _seq_b = seq_b;

    update_site_probs();
}

void SimpleGPUPopulation::set_genome(index_t *genome, const index_t pop_size)
{
    _pop_size = pop_size;
    _genome = genome;
}

void SimpleGPUPopulation::set_fitness(data_t *fitness)
{
    _fitness = fitness;
}

void SimpleGPUPopulation::sort(const bool minimize)
{
    if (_pop_size == 0 || _num_seqs == 0)
    {
        return;
    }

    population_fitness();

    cuda::range<<<1, _threads>>>(d_indices, _pop_size, 0);

    if (minimize)
    {
		cub::DeviceRadixSort::SortPairs(d_sort_buffer, _sort_bytes, d_fitness, d_fitness_sorted, d_indices, d_indices_sorted, _pop_size);
    }
    else
    {
        cub::DeviceRadixSort::SortPairsDescending(d_sort_buffer, _sort_bytes, d_fitness, d_fitness_sorted, d_indices, d_indices_sorted, _pop_size);
    }

    const int block { _threads };
    const dim3 grid((_num_seqs + block - 1) / block, 1, 1);
	const size_t shared = _pop_size * block * sizeof(index_t);

	cuda::copy<<<1, block>>>(d_fitness_sorted, d_fitness, _pop_size);
	simple_kernel::reorder_genome<<<grid, block, shared>>>(d_indices_sorted, d_genome, _num_seqs, _pop_size);
}

void SimpleGPUPopulation::site_prob(const index_t num_ic, const seq_t *msa, data_t *site_prob)
{
    const data_t residual { _lambda / (_lambda * _q + _num_seqs * _q) };
    const data_t scale { data_t(1.0) / (_lambda + _num_seqs) };

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
            site_prob[index] *= scale;
            site_prob[index] += residual;
        }
    }
}

void SimpleGPUPopulation::update_site_probs()
{
    delete[] _site_prob_a;
    delete[] _site_prob_b;

    _site_prob_a = new data_t[_num_ic_a * _q];
    _site_prob_b = new data_t[_num_ic_b * _q];

    site_prob(_num_ic_a, _seq_a, _site_prob_a);
    site_prob(_num_ic_b, _seq_b, _site_prob_b);
}

void SimpleGPUPopulation::kill_and_reproduce(const index_t kill_start, const index_t kill_end, const index_t repr_start, const index_t repr_end)
{
    const int grid { kill_end - kill_start };
    const int block { _threads };

    simple_kernel::kill_and_reproduce<<<grid, block, sizeof(int)>>>(d_genome, _num_seqs, kill_start, repr_start, repr_end, _params->prng_states);
}

void SimpleGPUPopulation::mutate(const double ratio, const index_t start, const index_t end)
{
    const index_t swaps = ratio * _num_seqs;

    simple_kernel::mutate<<<1, end - start>>>(d_genome, swaps, _num_seqs, start, _params->prng_states);
}

void SimpleGPUPopulation::initialize()
{
    check_device();
    init_gpu_data();
}

void SimpleGPUPopulation::finalize()
{
    retrieve_data();
}

void SimpleGPUPopulation::check_device()
{
    int device_count { 0 };
    cudaErr( cudaGetDeviceCount(&device_count) );

    if (device_count == 0)
    {
        throw std::runtime_error("No CUDA capable device found");
    }

    // We support only one GPU
    cudaSetDevice(0);
    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);

    _warp_size = device_props.warpSize;
    _threads = 2 * _warp_size;
}

void SimpleGPUPopulation::init_gpu_data()
{
    cudaErr(
        cudaMalloc(&d_fitness, _pop_size * sizeof(float))
    );
    cudaErr(
        cudaMalloc(&d_fitness_sorted, _pop_size * sizeof(float))
    );
    cudaErr(
        cudaMalloc(&d_indices_sorted, _pop_size * sizeof(bool))
    );
    cudaErr(
        cudaMalloc(&d_indices, _pop_size * sizeof(bool))
    );

    cudaErr(
        cudaMalloc(&d_genome, _pop_size * _num_seqs * sizeof(index_t))
    );
    cudaErr(
        cuda::memcpyH2D(d_genome, _genome, _pop_size * _num_seqs * sizeof(index_t))
    );

    cudaErr(
        cudaMalloc(&d_seq_a, _num_ic_a * _num_seqs * sizeof(seq_t))
    );
    cudaErr(
        cuda::memcpyH2D(d_seq_a, _seq_a, _num_ic_a * _num_seqs * sizeof(seq_t))
    );

    cudaErr(
        cudaMalloc(&d_seq_b, _num_ic_b * _num_seqs * sizeof(seq_t))
    );
    cudaErr(
        cuda::memcpyH2D(d_seq_b, _seq_b, _num_ic_b * _num_seqs * sizeof(seq_t))
    );

    cudaErr(
        cudaMalloc(&d_site_prob_a, _num_ic_a * _q * sizeof(data_t))
    );
    cudaErr(
        cuda::memcpyH2D(d_site_prob_a, _site_prob_a, _num_ic_a * _q * sizeof(data_t))
    );

    cudaErr(
        cudaMalloc(&d_site_prob_b, _num_ic_b * _q * sizeof(data_t))
    );
    cudaErr(
        cuda::memcpyH2D(d_site_prob_b, _site_prob_b, _num_ic_b * _q * sizeof(data_t))
    );

    cudaErr(
        cudaMalloc(&_params->prng_states, _pop_size * sizeof(curandState))
    );

    // Allocate storage for sorting
    cudaErr(
        cub::DeviceRadixSort::SortPairs(nullptr, _sort_bytes, d_fitness, d_fitness_sorted, d_indices, d_indices_sorted, _pop_size)
    );
    cudaErr(
        cudaMalloc(&d_sort_buffer, _sort_bytes)
    );

    // Initializing values
    cuda::fill<<<1, _warp_size>>>(d_fitness, _pop_size, 0.0f);

    // Initialize PRNGs
    std::random_device random_source;
    simple_kernel::init_prng<<<1, _pop_size>>>(_params->prng_states, random_source());

}

void SimpleGPUPopulation::retrieve_data()
{
    cudaErr(
        cuda::memcpyD2H(_genome, d_genome, _pop_size * _num_seqs * sizeof(index_t))
    );

    // Copy fitness back correcting data type
    float *temp_fitness = new float[_pop_size];
    cudaErr(
        cuda::memcpyD2H(temp_fitness, d_fitness, _pop_size * sizeof(float))
    );
    std::copy(temp_fitness, temp_fitness + _pop_size, _fitness);
    delete[] temp_fitness;

    free_device_memory();
}


void SimpleGPUPopulation::population_fitness()
{
    const data_t residual { _lambda / (_lambda * _q * _q + _num_seqs * _q * _q) };
    const data_t scale { data_t(1.0) / (_num_seqs + _lambda) };

	const dim3 grid(_pop_size, _num_ic_a, _num_ic_b);
    const dim3 block(4 * _warp_size, 1, 1);
    const size_t shared { _q * _q * sizeof(int32_t) };

    cuda::fill<<<1, _warp_size>>>(d_fitness, _pop_size, 0.0f);

    simple_kernel::fitness<<<grid, block, shared>>>(residual, scale, _num_seqs, _q, d_genome, d_seq_a, d_seq_b, d_fitness, d_site_prob_a, d_site_prob_b, _num_ic_a);
}
