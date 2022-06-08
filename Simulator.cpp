//
// Created by jiang on 2022/5/20.
//

#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>

#include "Simulator.h"
#include "Utils.h"

Simulator::Simulator(const std::vector<ColocationType> &prevalentPatterns,
                     const std::vector<FeatureType> &features,
                     const Eigen::MatrixXd &embeddingRepresentation)
    : _prevalentPatterns(prevalentPatterns),
      _answerTime(0),
      _positiveAnswerTime(0),
      _features(features),
      _embeddingRepresentation(embeddingRepresentation) {
    unsigned int maxPatternLength = 0;
    for(auto &prevalentPattern : _prevalentPatterns) {
        maxPatternLength = std::max<unsigned int>(maxPatternLength, prevalentPattern.size());
    }

    for(int i = 0; i < _features.size(); ++i) {
        _featureIndex[_features[i]] = i;
    }

    _maxFavorSize = std::min<unsigned int>(maxPatternLength, _features.size() / 2);
    _minFavorNum = 1; // In the paper, the variable should be 1.
    _maxFavorNum = std::sqrt(_features.size()); // In the paper, the variable should be std::sqrt(_features.size()).

    std::sort(_prevalentPatterns.begin(), _prevalentPatterns.end());

    auto favors = _generateFavors();

    _minDislikeFavorNum = 1;
    _maxDislikeFavorNum = std::ceil(favors.size() / 2.0); // In the paper, the variable should be favors.size() / 2.

    _generatePreferredPatterns(favors);
    _generateDislikePattern();

    std::cout << "Simulator favor number : " << favors.size() << std::endl;
    std::cout << "Simulator preferred pattern number : " << _preferredPatterns.size() << std::endl;
    std::cout << "Simulator dislike pattern number : " << _dislikePatterns.size() << std::endl;
}

double Simulator::_calculateSemanticDistance(const ColocationType &pattern1, const ColocationType &pattern2) {
    double totalDistance = 0;

    for(int i = 0; i < pattern1.size(); ++i) {
        auto featureIndex1 = _featureIndex[pattern1[i]];
        Eigen::VectorXd embedding1 = _embeddingRepresentation(featureIndex1, Eigen::all);
        double bestDistance = -1;
        for(int j = 0; j < pattern2.size(); ++j) {
            auto featureIndex2 = _featureIndex[pattern2[j]];
            Eigen::VectorXd embedding2 = _embeddingRepresentation(featureIndex2, Eigen::all);

            double distance = (embedding1 - embedding2).lpNorm<2>();
            if(bestDistance < 0 || distance < bestDistance) {
                bestDistance = distance;
            }
        }

        totalDistance += bestDistance;
    }

    for(int i = 0; i < pattern2.size(); ++i) {
        auto featureIndex2 = _featureIndex[pattern2[i]];
        Eigen::VectorXd embedding2 = _embeddingRepresentation(featureIndex2, Eigen::all);
        double bestDistance = -1;
        for(int j = 0; j < pattern1.size(); ++j) {
            auto featureIndex1 = _featureIndex[pattern1[j]];
            Eigen::VectorXd embedding1 = _embeddingRepresentation(featureIndex1, Eigen::all);

            double distance = (embedding2 - embedding1).lpNorm<2>();
            if(bestDistance < 0 || distance < bestDistance) {
                bestDistance = distance;
            }
        }

        totalDistance += bestDistance;
    }

    return totalDistance / 2;
}

std::vector<std::vector<double>> Simulator::_calculateDistance(const std::vector<ColocationType> &candidatePatterns,
                                                               const std::vector<ColocationType> &preferredPatterns) {
    std::vector<std::vector<double>> ans(candidatePatterns.size(), std::vector<double>(preferredPatterns.size()));

    for(int i = 0; i < candidatePatterns.size(); ++i) {
        for(int j = 0; j < preferredPatterns.size(); ++j) {
            ans[i][j] = _calculateSemanticDistance(candidatePatterns[i], preferredPatterns[j]);
        }
    }

    return ans;
}

ColocationType Simulator::_generateFavorRandom(unsigned int favorSize) {
    ColocationType combination;

    std::sample(std::begin(_features), std::end(_features),
                std::back_inserter(combination), favorSize,
                std::mt19937{std::random_device{}()});

    return combination;
}

std::vector<ColocationType> Simulator::_generateFavors() {
    unsigned int favorNum = randomInt(_minFavorNum, _maxFavorNum);

    std::vector<ColocationType> favors;
    for(int i = 0; i < favorNum; ++i) {
        double favorSize = randomInt(2, _maxFavorSize);

        ColocationType favor;
        while(std::binary_search(favors.begin(), favors.end(), favor = _generateFavorRandom(favorSize)));
        favors.push_back(favor);
    }

    std::sort(std::begin(favors), std::end(favors));

    return favors;
}


void Simulator::_generatePreferredPatterns(const std::vector<ColocationType> &favors) {
    auto favorDistance = _calculateDistance(_prevalentPatterns, favors);

    for(int i = 0; i < _prevalentPatterns.size(); ++i) {
        for(int j = 0; j < favors.size(); ++j) {
            if(favorDistance[i][j] < _similarityThreshold) {
                _preferredPatterns.push_back(_prevalentPatterns[i]);
                break;
            }
        }
    }
}

void Simulator::_removeDislikePatterns(const std::vector<ColocationType> &favors) {
    auto dislikeFavorNum = randomInt(_minDislikeFavorNum, _maxDislikeFavorNum);

    std::vector<ColocationType> dislikeFavors;
    std::sample(std::begin(favors), std::end(favors),
                std::back_inserter(dislikeFavors), dislikeFavorNum,
                std::mt19937{std::random_device{}()});

    auto distance = _calculateDistance(_preferredPatterns, dislikeFavors);
    std::vector<ColocationType> dislikePatterns;
    for(int i = 0; i < _preferredPatterns.size(); ++i) {
        for(int j = 0; j < dislikeFavors.size(); ++j) {
            if(distance[i][j] < _similarityThreshold) {
                dislikePatterns.push_back(_prevalentPatterns[i]);
                break;
            }
        }
    }

    auto preferredEnd = _preferredPatterns.end();
    for(auto &dislikePattern : dislikePatterns) {
        preferredEnd = std::remove(std::begin(_preferredPatterns), preferredEnd, dislikePattern);
    }
    _preferredPatterns.erase(preferredEnd, _preferredPatterns.end());
}

inline void Simulator::_generateDislikePattern() {
    std::set_difference(std::begin(_prevalentPatterns), std::end(_prevalentPatterns),
                        std::begin(_preferredPatterns), std::end(_preferredPatterns),
                        std::back_inserter(_dislikePatterns));
}

bool Simulator::answer(const ColocationType &pattern) {
    ++_answerTime;
    if(std::binary_search(std::begin(_preferredPatterns), std::end(_preferredPatterns), pattern)) {
        ++_positiveAnswerTime;
        return true;
    }
    return false;
}

unsigned int Simulator::getAnswerTime() {
    return _answerTime;
}

unsigned int Simulator::getPositiveAnswerTime() {
    return _positiveAnswerTime;
}

ConfusionMatrixType Simulator::evaluate(const std::vector<ColocationType> &predictedPatterns) {
    ConfusionMatrixType confusionMatrix;
    std::memset((void *)&confusionMatrix, 0, sizeof(ConfusionMatrixType));

    for(auto &preferredPattern : _preferredPatterns) {
        if(std::binary_search(std::begin(predictedPatterns), std::end(predictedPatterns), preferredPattern)) {
            ++confusionMatrix.TP;
        } else {
            ++confusionMatrix.FN;
        }
    }

    for(auto &dislikePattern : _dislikePatterns) {
        if(std::binary_search(std::begin(predictedPatterns), std::end(predictedPatterns), dislikePattern)) {
            ++confusionMatrix.FP;
        } else {
            ++confusionMatrix.TN;
        }
    }

    return confusionMatrix;
}
