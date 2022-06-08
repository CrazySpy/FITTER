//
// Created by jiang on 2022/5/20.
//

#ifndef CPIPO_SIMULATOR_H
#define CPIPO_SIMULATOR_H

#include <vector>
#include <map>
#include <unordered_map>

#include "Types.h"

#include <Eigen/Dense>

class Simulator {
private:
    std::vector<FeatureType> _features;
    std::vector<ColocationType> _prevalentPatterns;

    std::unordered_map<FeatureType, size_t> _featureIndex;

    unsigned int _maxFavorSize;
    unsigned int _minFavorNum;
    unsigned int _maxFavorNum;

    unsigned int _minDislikeFavorNum;
    unsigned int _maxDislikeFavorNum;

    std::vector<ColocationType> _preferredPatterns;

    // The set is the difference set of prevalent patterns and preferred patterns.
    std::vector<ColocationType> _dislikePatterns;

    Eigen::MatrixXd _embeddingRepresentation;

    // Store the reply time.
    unsigned int _answerTime;
    // Store the time of reply which answers true to user.
    unsigned int _positiveAnswerTime;

    double _similarityThreshold = 4;
private:
    // Calculate semantic distance.
    double Simulator::_calculateSemanticDistance(const ColocationType &pattern1, const ColocationType &pattern2);

    // Calculate distance.
    std::vector<std::vector<double>> _calculateDistance(const std::vector<ColocationType> &candidatePatterns,
                                                        const std::vector<ColocationType> &preferredPatterns);

    ColocationType _generateFavorRandom(unsigned int favorSize);

    std::vector<ColocationType> _generateFavors();

    void _generatePreferredPatterns(const std::vector<ColocationType> &favors);
    // Generate some exceptional patterns, and remove them from preferred pattern set.
    void _removeDislikePatterns(const std::vector<ColocationType> &favors);

    void _generateDislikePattern();
public:
    Simulator(const std::vector<ColocationType> &prevalentPatterns,
              const std::vector<FeatureType> &features,
              const Eigen::MatrixXd &embeddingRepresentation);

    // Answer whether the pattern is preferred.
    bool answer(const ColocationType &pattern);

    unsigned int getAnswerTime();
    unsigned int getPositiveAnswerTime();

    ConfusionMatrixType evaluate(const std::vector<ColocationType> &predictedPatterns);
};


#endif //CPIPO_SIMULATOR_H
