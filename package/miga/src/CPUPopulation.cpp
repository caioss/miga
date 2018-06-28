#include "CPUPopulation.hpp"
#include <algorithm>
#include <cmath>

data_t CPUPopulation:: singleFitness(const seq_t index) const {
    data_t coupling { 0 };
    uint32_t *pairCount { new uint32_t[_q * _q] };

    const data_t residual { _lambda / (_lambda * _q * _q + _numSeqs * _q * _q) };

    for (size_t lineA = 0; lineA < _numICA; ++lineA) {
        for (size_t lineB = 0; lineB < _numICB; ++lineB) {
            std::fill(pairCount, pairCount + _q * _q, 0);

            for (size_t column = 0; column < _numSeqs; ++column) {
                const seq_t aa1 { _seqA[lineA * _numSeqs + column] };
                const seq_t aa2 { _seqB[lineB * _numSeqs + column] };
                ++pairCount[aa1 * _q + aa2];
            }

            for (size_t aa1 = 0; aa1 < _q; ++aa1) {
                for (size_t aa2 = 0; aa2 < _q; ++aa2) {
                    const data_t pairProb { residual + pairCount[aa1 * _q + aa2] / (_numSeqs + _lambda) };

                    const data_t aa1Prob { _siteProbsA[lineA * _q + aa1] };
                    const data_t aa2Prob { _siteProbsB[lineB * _q + aa2] };

                    coupling += pairProb * log(pairProb / (aa1Prob * aa2Prob));
                }
            }

    return coupling;
}
