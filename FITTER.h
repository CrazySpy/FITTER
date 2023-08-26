//
// Created by jiang on 2022/5/31.
//

#ifndef EMBEDDINGBASED_FITTER_H
#define EMBEDDINGBASED_FITTER_H

#include <vector>
#include <map>

#include "Types.h"
#include "Simulator.h"

#include <Eigen/Dense>

class FITTER {
private:
    std::vector<FeatureType> _features;
    std::unordered_map<FeatureType, size_t> _featureIndex;

    std::vector<ColocationType> _prevalentPatterns;
    std::vector<ColocationType> _coarsePatterns;

    unsigned int _sampleSize;
    Simulator *_simulator;

    Eigen::MatrixXd _coOccurrenceMatrix;
    Eigen::MatrixXd _indicatorMatrix;
    Eigen::MatrixXd _embeddingRepresentation;

    double _alpha = 0.5;
    double _influenceIndex;
    double _mu;
    unsigned int _nearestSize;

    double _markovBoundary;

    std::map<ColocationType, std::map<ColocationType, double>> _patternDistanceCache;

//    std::map<ColocationType, double> _singularity;
//    std::map<ColocationType, double> _universality;
//    std::map<ColocationType, double> _sampleRank;

private:
    double _calculateCoOccurrenceValue(const FeatureType &feature1, const FeatureType &feature2);
    void _constructCoOccurrenceMatrix();
    void _constructIndicatorMatrix();
    void _constructEmbeddingRepresentation();

    double _calculateSemanticDistance(const ColocationType &pattern1, const ColocationType &pattern2);
    double _calculateStructuralDistance(const ColocationType &pattern1, const ColocationType &pattern2);
    double _calculatePatternDistance(const ColocationType &pattern1, const ColocationType &pattern2);

//    double _calculateSingularity(const ColocationType &pattern);
//    double _calculateUniversality(const ColocationType &pattern);
//    void _generateSampleRank();

    std::vector<ColocationType> _samplePatterns(std::vector<ColocationType> &candidatePatterns);
    std::vector<bool> _interactiveSurvey(const std::vector<ColocationType> &samplePatterns);

    void _selectCoarsePatterns(std::vector<ColocationType> &candidatePatterns,
                               const std::vector<ColocationType> &preferredPatterns);
    void _filterDislikePatterns(std::vector<ColocationType> &candidatePatterns,
                                                const std::vector<ColocationType> &dislikePatterns);
    std::vector<ColocationType> _filterCoarsePatterns();

    Eigen::MatrixXd _calculateRepresentationColumnWiseDistances(const Eigen::MatrixXd &N);
    double _calculateBetaByMarkovInequality(const Eigen::MatrixXd &distances);
    Eigen::MatrixXd _selectRepresentation(const Eigen::MatrixXd &representation, const Eigen::MatrixXd &columnWiseDistances, double beta);

public:
    FITTER(const std::vector<ColocationType> &prevalentPatterns,
           unsigned int sampleSize, double alpha, double markovBoundary,
           double influenceIndex,
           double mu,
           unsigned int ns,
           Simulator *simulator = nullptr);

    void printNearestFeature();

    std::vector<ColocationType> execute();
};


#endif //EMBEDDINGBASED_FITTER_H
