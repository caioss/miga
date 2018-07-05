#include "Population.hpp"
#include "CPUPopulation.hpp"

Population *make_population(const std::string platform)
{
    if (platform == "CPU")
    {
        return new CPUPopulation();
    }
    else if (platform == "GPU")
    {
        return nullptr;
        //return new GPUPopulation();
    }
    else
    {
        return nullptr;
    }
}
