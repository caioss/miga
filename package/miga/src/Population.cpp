#include "Population.hpp"

Population *make_population(const std::string platform) {
    if (platform == "CPU") {
        return nullptr;
        //return new CPUPopulation();

    } else if (platform == "GPU") {
        return nullptr;
        //return new GPUPopulation();

    } else {
        return nullptr;
    }
}
