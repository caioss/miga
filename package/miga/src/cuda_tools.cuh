#ifndef CUDA_TOOLS_CUH
#define CUDA_TOOLS_CUH

#include <sstream>

#define cudaErr(ans) { cuda::error((ans), __FILE__, __LINE__); }

namespace cuda {
    inline void error(cudaError_t code, const char *file, int line, bool abort=true)
    {
        if (code != cudaSuccess) 
        {
            std::stringstream error("");
            error << "CUDA error: " << cudaGetErrorString(code)
            << " at file " << file
            << ":" << line;

            if (abort)
            {
                throw std::runtime_error(error.str());
            }
        }
    }

    template <class T>
    inline cudaError_t memcpyH2D(T *d_target, const T *h_source, const size_t size)
    {
        return cudaMemcpy(d_target, h_source, size, cudaMemcpyHostToDevice);
    }

    template <class T>
    inline cudaError_t memcpyD2H(T *h_target, const T *d_source, const size_t size)
    {
        return cudaMemcpy(h_target, d_source, size, cudaMemcpyDeviceToHost);
    }

    template <class T>
    __global__ void echof(const T *in, const int size)
    {
        const int t_idx = threadIdx.x;
        const int stride = blockDim.x;

        printf("ARRAY: ");
        for (int i = t_idx; i < size; i += stride)
        {
            printf("%.3f ", in[i]);
        }
        printf("\n");
    }

    template <class T>
    __global__ void echoi(const T *in, const int size)
    {
        const int t_idx = threadIdx.x;
        const int stride = blockDim.x;

        printf("ARRAY: ");
        for (int i = t_idx; i < size; i += stride)
        {
            printf("%d ", in[i]);
        }
        printf("\n");
    }

    template <class T>
    __global__ void copy(const T *in, T *out, const int size)
    {
        const int t_idx = threadIdx.x;
        const int stride = blockDim.x;

        for (int i = t_idx; i < size; i += stride)
        {
            out[i] = in[i];
        }
    }

    template <class T>
    __global__ void fill(T *data, const int size, const T value)
    {
        const int t_idx = threadIdx.x;
        const int stride = blockDim.x;

        for (int i = t_idx; i < size; i += stride)
        {
            data[i] = value;
        }
    }

    template <class T, class U>
    __global__ void fill_if(T *data, const U *cond, const int size, const T value)
    {
        const int t_idx = threadIdx.x;
        const int stride = blockDim.x;

        for (int i = t_idx; i < size; i += stride)
        {
            if (cond[i])
            {
                data[i] = value;
            }
        }
    }

    template <class T>
    __global__ void range(T *data, const int size, const T start = 0)
    {
        const int t_idx = threadIdx.x;
        const int stride = blockDim.x;

        for (int i = t_idx; i < size; i += stride)
        {
            data[i] = start + i;
        }
    }

}

#endif
