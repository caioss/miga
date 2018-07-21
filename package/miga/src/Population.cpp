#include "Population.hpp"
#include "CPUPopulation.hpp"

#ifdef HAS_CUDA
#include "GPUPopulation.hpp"
#include "SimpleGPUPopulation.hpp"
#endif

Population *make_population(const std::string platform)
{
    if (platform == "CPU")
    {
        return new CPUPopulation();
    }
#ifdef HAS_CUDA
    else if (platform == "GPU")
    {
        return new GPUPopulation();
    }
    else if (platform == "SimpleGPU")
    {
        return new SimpleGPUPopulation();
    }
#endif
    else
    {
        return nullptr;
    }
}
