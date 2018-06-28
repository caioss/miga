#ifndef CPUPOPULATION_HPP
#define CPUPOPULATION_HPP

#include "Population.hpp"

class CPUPopulation : public Population {
public:
    data_t singleFitness(const seq_t index) const;
};

#endif
