//
// Created by jiang on 2022/5/20.
//

#ifndef CPIPO_SIMULATOR_H
#define CPIPO_SIMULATOR_H

#include <vector>
#include <map>
#include <unordered_map>

#include "Types.h"

class Simulator {
private:
//    std::vector<FeatureType> _features;
    std::vector<ColocationType> _prevalentPatterns;
    std::vector<std::vector<FeatureType>> _preferredPatterns;

    // The set is the difference set of prevalent patterns and preferred patterns.
    std::vector<ColocationType> _dislikePatterns;

    // Store the reply time.
    unsigned int _answerTime;
    // Store the time of reply which answers true to user.
    unsigned int _positiveAnswerTime;
private:
    void _generateDislikePattern();
public:
    Simulator(const std::vector<ColocationType> &prevalentPatterns,
              const std::vector<std::vector<FeatureType>> &preferredPatterns);

    // Answer whether the pattern is preferred.
    bool answer(const ColocationType &pattern);

    unsigned int getAnswerTime();
    unsigned int getPositiveAnswerTime();

    ConfusionMatrixType evaluate(const std::vector<ColocationType> &predictedPatterns);
};


#endif //CPIPO_SIMULATOR_H
