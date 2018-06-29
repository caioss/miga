#ifndef POPULATION_HPP
#define POPULATION_HPP

#include "types.h"
#include <string>

class Population {
public:
    virtual void setQ(const seq_t value) = 0;
    virtual void setLambda(const data_t value) = 0;
    virtual std::string platform() const = 0;

};

Population *make_population(const std::string platform);

#endif
