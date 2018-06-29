#ifndef CPUPOPULATION_HPP
#define CPUPOPULATION_HPP

#include "Population.hpp"

class CPUPopulation : public Population {
public:
    void setQ(const seq_t value);
    void setLambda(const data_t value);
    std::string platform() const { return "CPU"; };

private:
    data_t computeSingleFitness(const seq_t index) const;

private:
    seq_t _q;
    seq_t _numSeqs;
    seq_t _numICA;
    seq_t _numICB;
    data_t _lambda;
    seq_t *_genome;
    seq_t *_seqA;
    seq_t *_seqB;
    data_t *_siteProbsA;
    data_t *_siteProbsB;
};

#endif
