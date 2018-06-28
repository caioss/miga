#ifndef POPULATION_HPP
#define POPULATION_HPP

#include "types.h"
#include <string>

class Population {
public:
    virtual data_t singleFitness(const seq_t index) const = 0;

};

Population *make_population(const std::string platform);

#endif
