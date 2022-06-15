//
// Created by jiang on 2022/5/20.
//

#include <algorithm>
#include <random>
#include <iostream>

#include "Simulator.h"

Simulator::Simulator(const std::vector<ColocationType> &prevalentPatterns,
                     const std::vector<std::vector<FeatureType>> &preferredPatterns)
    : _prevalentPatterns(prevalentPatterns),
      _preferredPatterns(preferredPatterns),
      _answerTime(0),
      _positiveAnswerTime(0) {
    std::sort(_prevalentPatterns.begin(), _prevalentPatterns.end());
    std::sort(_preferredPatterns.begin(), _preferredPatterns.end());
    _generateDislikePattern();
//    unsigned int maxPatternLength = 0;
//    for(auto &prevalentPattern : _prevalentPatterns) {
//        maxPatternLength = std::max<unsigned int>(maxPatternLength, prevalentPattern.size());
//    }
//
//    for(int i = 0; i < _features.size(); ++i) {
//        _featureIndex[_features[i]] = i;
//    }
//
//    _maxFavorSize = std::min<unsigned int>(maxPatternLength, _features.size() / 2);
//    _minFavorNum = 1; // In the paper, the variable should be 1.
//    _maxFavorNum = std::sqrt(_features.size()); // In the paper, the variable should be std::sqrt(_features.size()).
//
//    std::sort(_prevalentPatterns.begin(), _prevalentPatterns.end());
//
//    auto favors = _generateFavors();
//
//    _minDislikeFavorNum = 1;
//    _maxDislikeFavorNum = std::ceil(favors.size() / 2.0); // In the paper, the variable should be favors.size() / 2.
//
//    _generatePreferredPatterns(favors);
//    _generateDislikePattern();
//
//    std::cout << "Simulator favor number : " << favors.size() << std::endl;
    std::cout << "Simulator preferred pattern number : " << _preferredPatterns.size() << std::endl;
    std::cout << "Simulator dislike pattern number : " << _dislikePatterns.size() << std::endl;
}

void Simulator::_generateDislikePattern() {
    for(auto &prevalentPattern : _prevalentPatterns) {
        if(!std::binary_search(_preferredPatterns.begin(), _preferredPatterns.end(), prevalentPattern)) {
            _dislikePatterns.push_back(prevalentPattern);
        }
    }
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
