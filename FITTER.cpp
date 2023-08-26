//
// Created by jiang on 2022/5/31.
//

#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <set>
#include <string>

#include "FITTER.h"
#include "Utils.h"
#include "FCM.h"


std::ostream& operator << (std::ostream &os, const ColocationType &pattern) {
    os << std::accumulate(std::next(pattern.begin()), pattern.end(), pattern.front(), [](std::string str, FeatureType feature) {
        return str + ", " + feature;
    });

    return os;
}

FITTER::FITTER(const std::vector<ColocationType> &prevalentPatterns,
               unsigned int sampleSize, double alpha, double markovBoundary, double influenceIndex, double mu, unsigned int ns,
               Simulator *simulator)
    : _prevalentPatterns(prevalentPatterns),
      _sampleSize(sampleSize),
      _alpha(alpha),
      _markovBoundary(markovBoundary),
      _influenceIndex(influenceIndex),
      _mu(mu),
      _nearestSize(ns),
      _simulator(simulator) {
    for(auto &prevalentPattern : _prevalentPatterns) {
        std::sort(prevalentPattern.begin(), prevalentPattern.end());
        for(int i = 0; i < prevalentPattern.size(); ++i) {
            _features.push_back(prevalentPattern[i]);
        }
    }
    std::sort(_prevalentPatterns.begin(), _prevalentPatterns.end());
    std::sort(_features.begin(), _features.end());
    _features.erase(std::unique(_features.begin(), _features.end()), _features.end());

    for(int i = 0; i < _features.size(); ++i) {
        _featureIndex[_features[i]] = i;
    }

    _indicatorMatrix = Eigen::MatrixXd(_features.size(), 0);

    _constructCoOccurrenceMatrix();
    _constructIndicatorMatrix();
    _constructEmbeddingRepresentation();

//    _generateSampleRank();
}

double FITTER::_calculateCoOccurrenceValue(const FeatureType &feature1, const FeatureType &feature2) {
    unsigned int coOccurrenceTime = 0;
    unsigned int feature1OccurrenceTime = 0;
    for(auto &prevalentPattern : _prevalentPatterns) {
        bool isOccur1 = std::binary_search(prevalentPattern.begin(), prevalentPattern.end(), feature1);
        bool isOccur2 = std::binary_search(prevalentPattern.begin(), prevalentPattern.end(), feature2);
        if(isOccur1) {
            feature1OccurrenceTime++;
        }

        if(isOccur1 && isOccur2) {
            coOccurrenceTime++;
        }
    }

    return 1.0 * coOccurrenceTime / feature1OccurrenceTime;
}

void FITTER::_constructCoOccurrenceMatrix() {
    _coOccurrenceMatrix = Eigen::MatrixXd(_features.size(), _features.size());
    for(int i = 0; i < _features.size(); ++i) {
        for(int j = 0; j < _features.size(); ++j) {
            _coOccurrenceMatrix(i, j) = _calculateCoOccurrenceValue(_features[i], _features[j]);
        }
    }
}

void FITTER::_constructIndicatorMatrix() {
//    cv::Mat cvCoOccurrenceMatrix;
//    Eigen::MatrixXf eigenCoOccurrenceMatrix = _coOccurrenceMatrix.cast<float>();

//    cv::eigen2cv(eigenCoOccurrenceMatrix, cvCoOccurrenceMatrix);

    FCM fcm(2, 0.5);
    fcm.setData(&_coOccurrenceMatrix);

    int smallClusterCount = 0;
    int iterationRound = 0;
    for(int k = 2; _features.size() >= k; k++) {
        iterationRound++;
//        cv::Mat labels, centers;
//        cv::kmeans(cvCoOccurrenceMatrix, k, labels,
//                   cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
//                   3, cv::KMEANS_PP_CENTERS, centers);
//
//        Eigen::MatrixXi eigenLabels;
//        cv::cv2eigen(labels, eigenLabels);
//        Eigen::MatrixXi eigenOneHot = oneHot(eigenLabels, centers.rows);
        fcm.setClusterNum(k);
        fcm.initMembership();

        fcm.execute();
        Eigen::MatrixXd *fuzzyMembership = fcm.getMembership();


        // Erase small clusters.
        Eigen::VectorXd pointNum = (*fuzzyMembership).colwise().sum();
        smallClusterCount += (pointNum.array() <= 1.0 / k * _features.size()).count();

        std::vector<int> bigClusterIndices;
        for(int i = 0; i < pointNum.size(); ++i) {
            if(pointNum(i) > 1) {
                bigClusterIndices.push_back(i);
            }
        }

        // Concat the indicator matrix.
        Eigen::MatrixXd indicatorMatrix(_features.size(), _indicatorMatrix.cols() + bigClusterIndices.size());
        indicatorMatrix << _indicatorMatrix, (*fuzzyMembership)(Eigen::indexing::all, bigClusterIndices);

        _indicatorMatrix = indicatorMatrix;

        // Quit loop condition.
        if(smallClusterCount >= std::ceil(k / _alpha)) {
            break;
        }
    }
    std::cout << "Iteration round: " << iterationRound << std::endl;
}

void FITTER::_constructEmbeddingRepresentation() {
    Eigen::MatrixXd centered = _indicatorMatrix.cast<double>().rowwise() - _indicatorMatrix.cast<double>().colwise().mean();
    Eigen::MatrixXd cov = (centered.transpose() * centered) / double(centered.cols() - 1);

    Eigen::BDCSVD<Eigen::MatrixXd> svd(cov, Eigen::ComputeFullV);
    Eigen::MatrixXd V = svd.matrixV();

    Eigen::MatrixXd N = centered * V.transpose();

    Eigen::MatrixXd distances = _calculateRepresentationColumnWiseDistances(N);
    double beta = _calculateBetaByMarkovInequality(distances);
    std::cout << "Beta: " << beta << std::endl;

//    std::vector<int> goodNColumnIndex;
//    for(int i = 0; i < N.cols(); ++i) {
//        double maxDistance = 0;
//        for(int j = 0; j < N.cols(); ++j) {
//            maxDistance = std::max(maxDistance, distances(i, j));
//        }
//
//        if(maxDistance >= beta) {
//            goodNColumnIndex.push_back(i);
//        }
//    }
//
//    _embeddingRepresentation = N(Eigen::indexing::all, goodNColumnIndex);
    _embeddingRepresentation = _selectRepresentation(N, distances, beta);
    std::cout << "Raw embedding column number: " << N.cols() << std::endl;
    std::cout << "Eliminate column number: " << N.cols() - _embeddingRepresentation.cols() << std::endl;
}

double FITTER::_calculateSemanticDistance(const ColocationType &pattern1, const ColocationType &pattern2) {
    std::vector<Eigen::Index> pattern1Indices, pattern2Indices;
    for(int i = 0; i < pattern1.size(); ++i) {
        pattern1Indices.push_back(_featureIndex[pattern1[i]]);
    }
    for(int i = 0; i < pattern2.size(); ++i) {
        pattern2Indices.push_back(_featureIndex[pattern2[i]]);
    }
    Eigen::MatrixXd pattern1Embeddings = _embeddingRepresentation(pattern1Indices, Eigen::indexing::all);
    Eigen::MatrixXd pattern2Embeddings = _embeddingRepresentation(pattern2Indices, Eigen::indexing::all);

    double totalDistance = 0;

    for(int i = 0; i < pattern1.size(); ++i) {
        Eigen::RowVectorXd embedding1 = pattern1Embeddings.row(i);
        double bestDistance = (pattern2Embeddings.rowwise() - embedding1).rowwise().lpNorm<2>().minCoeff() / sqrt(_features.size());

//        Eigen::VectorXd embedding1 = _embeddingRepresentation.row(featureIndex1);
//        double bestDistance = -1;
//        for(int j = 0; j < pattern2.size(); ++j) {
//            auto featureIndex2 = _featureIndex[pattern2[j]];
//            Eigen::VectorXd embedding2 = _embeddingRepresentation.row(featureIndex2);
//
//            // Normalize distance.
//            double distance = (embedding1 - embedding2).lpNorm<2>() / sqrt(_features.size());
//            if(bestDistance < 0 || distance < bestDistance) {
//                bestDistance = distance;
//            }
//        }

        totalDistance += bestDistance;
    }

    for(int i = 0; i < pattern2.size(); ++i) {
        Eigen::RowVectorXd embedding2 = pattern2Embeddings.row(i);
        double bestDistance = (pattern1Embeddings.rowwise() - embedding2).rowwise().lpNorm<2>().minCoeff() / sqrt(_features.size());
//        double bestDistance = -1;
//        for(int j = 0; j < pattern1.size(); ++j) {
//            auto featureIndex1 = _featureIndex[pattern1[j]];
//            Eigen::VectorXd embedding1 = _embeddingRepresentation.row(featureIndex1);
//
//            double distance = (embedding2 - embedding1).lpNorm<2>() / sqrt(_features.size());
//            if(bestDistance < 0 || distance < bestDistance) {
//                bestDistance = distance;
//            }
//        }

        totalDistance += bestDistance;
    }

    return totalDistance / 2;
}

double FITTER::_calculateStructuralDistance(const ColocationType &pattern1, const ColocationType &pattern2) {
    ColocationType union_features, intersected_features;
    std::set_union(pattern1.begin(), pattern1.end(), pattern2.begin(), pattern2.end(), std::back_inserter(union_features));
    std::set_intersection(pattern1.begin(), pattern1.end(),pattern2.begin(), pattern2.end(), std::back_inserter(intersected_features) );

    return 1 - intersected_features.size() * 1.0 / union_features.size();
}

double FITTER::_calculatePatternDistance(const ColocationType &pattern1, const ColocationType &pattern2) {
    if(_patternDistanceCache.count(pattern1) && _patternDistanceCache[pattern1].count(pattern2))
        return _patternDistanceCache[pattern1][pattern2];

    double semanticDistance = _calculateSemanticDistance(pattern1, pattern2);
    double structuralDistance = _calculateStructuralDistance(pattern1, pattern2);

    //return 2 * semanticDistance * (1 - structuralDistance) / (semanticDistance + (1 - structuralDistance));
    return _patternDistanceCache[pattern1][pattern2] = semanticDistance / std::pow(structuralDistance, _influenceIndex);
    //return (semanticDistance + (1 - structuralDistance)) / 2;
}

//double FITTER::_calculateSingularity(const ColocationType &pattern) {
//    // $D(P) = max_{f \in P}(1 / \Vert c(f) \Vert)$
//    // c(f) is the co-occurrence vector of feature f.
//
//    std::vector<unsigned int> featureIndices;
//    for(auto &feature : pattern) {
//        featureIndices.push_back(_featureIndex[feature]);
//    }
//
//    auto coOccurrenceMatrix = _coOccurrenceMatrix(featureIndices, Eigen::indexing::all);
////    double singularity = coOccurrenceMatrix.rowwise().lpNorm<2>().cwiseInverse().maxCoeff(); // S1
//    double singularity = 1 - coOccurrenceMatrix.rowwise().lpNorm<2>().minCoeff() / std::sqrt(coOccurrenceMatrix.cols()); // S2
////    double singularity = 1 - (coOccurrenceMatrix * coOccurrenceMatrix).diagonal().minCoeff() / coOccurrenceMatrix.cols(); // S3
//
//    return singularity;
//}
//
//double FITTER::_calculateUniversality(const ColocationType &pattern) {
//    std::vector<unsigned int> featureIndices;
//    for(auto &feature : pattern) {
//        featureIndices.push_back(_featureIndex[feature]);
//    }
//
//    auto coOccurrenceMatrix = _coOccurrenceMatrix(featureIndices, featureIndices);
//
////    return (coOccurrenceMatrix * coOccurrenceMatrix).trace() / pattern.size(); //U1
////    return coOccurrenceMatrix.norm() / pattern.size(); // U2
//
////    return coOccurrenceMatrix.rowwise().lpNorm<2>().maxCoeff() / std::sqrt(pattern.size()); //U3
////    return coOccurrenceMatrix.rowwise().mean().maxCoeff(); // U4
//    return (coOccurrenceMatrix * coOccurrenceMatrix).diagonal().maxCoeff() / pattern.size(); // U5
//}
//
//void FITTER::_generateSampleRank() {
//    for(auto &pattern : _prevalentPatterns) {
//        double singularity = _calculateSingularity(pattern);
//        double universality = _calculateUniversality(pattern);
////        double sampleRank = (1 + _mu * _mu) * singularity * universality / (_mu * _mu * singularity + universality); // R1
////        double sampleRank = std::pow(singularity, _mu) * universality; // R2
//        double sampleRank = _mu * singularity + (1 - _mu) * universality; // R3
////        std::cout << sampleRank << std::endl;
//
//        _singularity[pattern] = singularity;
//        _universality[pattern] = universality;
//        _sampleRank[pattern] = sampleRank;
//    }
//}

//std::vector<ColocationType> FITTER::_samplePatterns(std::vector<ColocationType> &candidatePatterns) {
//    if(_sampleSize >= candidatePatterns.size()) {
//        auto sample = candidatePatterns;
//        candidatePatterns.clear();
//        return sample;
//    }
//
//    // Store the final sample result.
//    std::vector<ColocationType> sample;
//    std::vector<std::pair<double, ColocationType>> candidateSampleRank;
//    for(auto &pattern : candidatePatterns) {
//        candidateSampleRank.push_back({_sampleRank[pattern], pattern});
//    }
//    std::sort(candidateSampleRank.begin(), candidateSampleRank.end(), std::less());
//
//    for(int i = 0; i < _sampleSize; ++i) {
//        auto &pattern = candidateSampleRank[i].second;
//        sample.push_back(pattern);
//        candidatePatterns.erase(std::remove(std::begin(candidatePatterns), std::end(candidatePatterns), pattern), std::end(candidatePatterns));
//    }
//
//    return sample;
//}

std::vector<ColocationType> FITTER::_samplePatterns(std::vector<ColocationType> &candidatePatterns) {
    if(_sampleSize >= candidatePatterns.size()) {
        auto sample = candidatePatterns;
        candidatePatterns.clear();
        return sample;
    }

    std::vector<ColocationType> samples;

    static std::unordered_set<FeatureType> globalFeatures;
//    static std::vector<Eigen::Index> globalIndices;

    std::unordered_set<FeatureType> localFeatures;
//    std::vector<Eigen::Index> localIndices;

    for(int i = 0; i < _sampleSize; ++i) {
        double bestSampleRank = 0;
        ColocationType bestPattern;
//        std::vector<Eigen::Index> bestPatternIndices;

        for(auto &candidatePattern : candidatePatterns) {
//            std::vector<Eigen::Index> globalPatternIndices = globalIndices;
//            std::vector<Eigen::Index> localPatternIndices = localIndices;
//            for(auto &feature : candidatePattern) {
//                globalPatternIndices.push_back(_featureIndex[feature]);
//                localPatternIndices.push_back(_featureIndex[feature]);
//            }
//            std::sort(globalPatternIndices.begin(), globalPatternIndices.end());
//            globalPatternIndices.erase(std::unique(globalPatternIndices.begin(), globalPatternIndices.end()), globalPatternIndices.end());
//            std::sort(localPatternIndices.begin(), localPatternIndices.end());
//            globalPatternIndices.erase(std::unique(localPatternIndices.begin(), localPatternIndices.end()), localPatternIndices.end());

            //Eigen::MatrixXd patternCoOccurrenceMatrix = _coOccurrenceMatrix(globalPatternIndices, globalPatternIndices);

            //double sampleRank = (1 - _mu) * (1 - ((patternCoOccurrenceMatrix * patternCoOccurrenceMatrix).diagonal() / globalPatternIndices.size()).array()).maxCoeff() + _mu * globalPatternIndices.size() / _features.size();

            std::unordered_set<FeatureType> testGlobalFeatures = globalFeatures;
            std::unordered_set<FeatureType> testLocalFeatures = localFeatures;

            testGlobalFeatures.insert(candidatePattern.begin(), candidatePattern.end());
            testLocalFeatures.insert(candidatePattern.begin(), candidatePattern.end());
            double testGlobalInformation = 1.0 * testGlobalFeatures.size() / _features.size();
            double testLocalInformation = 1.0 * testLocalFeatures.size() / _features.size();

            double sampleRank = _mu * testLocalInformation + (1 - _mu) * testGlobalInformation;

            if(sampleRank >= bestSampleRank) {
                bestSampleRank = sampleRank;
                bestPattern = candidatePattern;
//                bestPatternIndices = globalPatternIndices;
            }
        }

        samples.push_back(bestPattern);
        globalFeatures.insert(bestPattern.begin(), bestPattern.end());
        localFeatures.insert(bestPattern.begin(), bestPattern.end());
//        globalIndices = bestPatternIndices;
        std::erase(candidatePatterns, bestPattern);
    }

    return samples;
}

std::vector<bool> FITTER::_interactiveSurvey(const std::vector<ColocationType> &samplePatterns) {
    std::vector<bool> feedback;
    for(auto &pattern : samplePatterns) {
        if(_simulator == nullptr) {
            // Reply machine will be developed latter.
            std::cout << "Do you like "
                      << std::accumulate(std::begin(pattern), std::end(pattern), std::string(""),
                                         [](const std::string &partial, const FeatureType &feature) {
                                             return partial + feature + ';';
                                         })
                      << " ?(Input 0/1)" << std::endl;
            bool answer;
            std::cin >> answer;
            feedback.push_back(answer);
        } else {
            feedback.push_back(_simulator->answer(pattern));
        }
    }

    return feedback;
}

void FITTER::_selectCoarsePatterns(std::vector<ColocationType> &candidatePatterns,
                                   const std::vector<ColocationType> &preferredPatterns) {
    // First, insert positive feedback pattern into coarse set.
    _coarsePatterns.insert(_coarsePatterns.end(), std::begin(preferredPatterns), std::end(preferredPatterns));

    // Then, calculate the nearest patterns of each preferred pattern.
    std::set<ColocationType> predictedPatterns;
    for(const auto &preferredPattern : preferredPatterns) {
        std::vector<std::pair<double, ColocationType>> similarities;
        for(auto &candidatePattern : candidatePatterns) {
            if(preferredPattern == candidatePattern) continue;

            double distance = 0;
            if(preferredPattern < candidatePattern) {
                distance = _calculatePatternDistance(preferredPattern, candidatePattern);
            } else {
                distance = _calculatePatternDistance(candidatePattern, preferredPattern);
            }
            similarities.emplace_back(distance, candidatePattern);
        }
        std::sort(similarities.begin(), similarities.end());
        similarities.erase(std::unique(similarities.begin(), similarities.end()), similarities.end());

//        std::cout << "Like: " << preferredPattern << ": " << std::endl;
        for(auto it = similarities.begin(); it != similarities.end() && it != similarities.begin() + _nearestSize; ++it) {
            predictedPatterns.insert((*it).second);
//            std::cout << (*it).second << std::endl;
        }
//        std::cout << std::endl;
    }

    auto candidateEnd = candidatePatterns.end();
    for(auto &predictedPattern : predictedPatterns) {
        candidateEnd = std::remove(std::begin(candidatePatterns), candidateEnd, predictedPattern);
        _coarsePatterns.push_back(predictedPattern);
    }

    candidatePatterns.erase(candidateEnd, candidatePatterns.end());
}

void FITTER::_filterDislikePatterns(std::vector<ColocationType> &candidatePatterns,
                                    const std::vector<ColocationType> &dislikePatterns) {
    std::set<ColocationType> predictedPatterns;
    for(const auto &dislikePattern : dislikePatterns) {
        std::vector<std::pair<double, ColocationType>> similarities;
        for(auto &candidatePattern : candidatePatterns) {
            if(dislikePattern == candidatePattern) continue;

            double distance = 0;
            if(dislikePattern < candidatePattern) {
                distance = _calculatePatternDistance(dislikePattern, candidatePattern);
            } else {
                distance = _calculatePatternDistance(candidatePattern, dislikePattern);
            }
            similarities.emplace_back(distance, candidatePattern);
        }
        std::sort(similarities.begin(), similarities.end());
        similarities.erase(std::unique(similarities.begin(), similarities.end()), similarities.end());

//        std::cout << "Dislike: " << dislikePattern << ": " << std::endl;
        for(auto it = similarities.begin(); it != similarities.end() && it != similarities.begin() + _nearestSize; ++it) {
            predictedPatterns.insert((*it).second);
//            std::cout << (*it).second << std::endl;
        }
//        std::cout << std::endl;
    }

    auto candidateEnd = candidatePatterns.end();
    for(auto &predictedPattern : predictedPatterns) {
        candidateEnd = std::remove(std::begin(candidatePatterns), candidateEnd, predictedPattern);
    }

    candidatePatterns.erase(candidateEnd, candidatePatterns.end());
}

std::vector<ColocationType> FITTER::_filterCoarsePatterns() {
    return _coarsePatterns;
}

Eigen::MatrixXd FITTER::_calculateRepresentationColumnWiseDistances(const Eigen::MatrixXd &N) {
    Eigen::MatrixXd distances(N.cols(), N.cols());
    for(int i = 0; i < N.cols(); ++i) {
        Eigen::VectorXd col1 = N.col(i);
        distances.row(i) = (N.colwise() - col1).colwise().lpNorm<2>();
//        for(int j = 0; j < N.cols(); ++j) {
//            Eigen::VectorXd col2 = N.col(j);
//            double distance = (col1 - col2).lpNorm<2>();
//            distances(i, j) = distance;
//        }
    }

    return distances;
}

double FITTER::_calculateBetaByMarkovInequality(const Eigen::MatrixXd &distances) {
    double mean = distances.mean();
    std::cout << "Means: " << mean << std::endl;
    double a = 1.0 / (1 - _markovBoundary);
    return mean * a;
}

Eigen::MatrixXd FITTER::_selectRepresentation(const Eigen::MatrixXd &representation,
                                              const Eigen::MatrixXd &columnWiseDistances,
                                              double beta) {
    Eigen::MatrixXd newRepresentation(representation.rows(), 1);
    // Choose the feature with the largest distances to others as initial.
    Eigen::Index initialRepresentationIndex;
    columnWiseDistances.rowwise().mean().maxCoeff(&initialRepresentationIndex);
    newRepresentation.col(0) = representation.col(initialRepresentationIndex);

    for(int i = 0; i < representation.cols(); ++i) {
        if(i == initialRepresentationIndex) continue;
        Eigen::VectorXd col = representation.col(i);
        Eigen::VectorXd newDistances = (newRepresentation.colwise() - col).colwise().lpNorm<2>();
        Eigen::Index maxDistanceIndex;
        double maxDistance = newDistances.maxCoeff(&maxDistanceIndex);
        if(maxDistance >= beta) {
            newRepresentation.conservativeResize(newRepresentation.rows(), newRepresentation.cols() + 1);
            newRepresentation.col(newRepresentation.cols() - 1) = col;
        }
//        Eigen::VectorXd candidateCol = representation(Eigen::indexing::all, i);
//        double maxDistance = -1;
//        for(int j = 0; j < newRepresentation.cols(); ++j) {
//            Eigen::VectorXd col2 = newRepresentation(Eigen::indexing::all, j);
//            double distance = (candidateCol - col2).lpNorm<2>();
//            if(maxDistance < 0 || distance > maxDistance) {
//                maxDistance = distance;
//            }
//        }
    }

    return newRepresentation;
}

void FITTER::printNearestFeature() {
    for(int i = 0; i < _features.size(); ++i) {
        const auto feature1 = _features[i];
        Eigen::VectorXd embedding1 = _embeddingRepresentation.row(i);
        double bestDistance = -1;
        FeatureType bestFeature;
        for(int j = 0; j < _features.size(); ++j) {
            if(i == j) continue;
            const auto feature2 = _features[j];
            Eigen::VectorXd embedding2 = _embeddingRepresentation.row(j);
            double distance = (embedding1 - embedding2).lpNorm<2>();
            if(bestDistance < 0 || distance < bestDistance) {
                bestDistance = distance;
                bestFeature = feature2;
            }
        }

        std::cout << feature1 << " : " << bestFeature << std::endl;
    }
}

std::vector<ColocationType> FITTER::execute() {
    std::vector<ColocationType> candidatePatterns = _prevalentPatterns;
    while (!candidatePatterns.empty()) {
        auto sample = _samplePatterns(candidatePatterns);
        auto feedback = _interactiveSurvey(sample);

        // Generate preferred and dislike patterns.
        std::vector<ColocationType> preferredPatterns;
        std::vector<ColocationType> dislikePatterns;
        for (int i = 0; i < feedback.size(); ++i) {
            if (feedback[i]) {
                preferredPatterns.push_back(sample[i]);
            } else {
                dislikePatterns.push_back(sample[i]);
            }
        }

        _selectCoarsePatterns(candidatePatterns, preferredPatterns);
        _filterDislikePatterns(candidatePatterns, dislikePatterns);
    }

    std::sort(_coarsePatterns.begin(), _coarsePatterns.end());

//    printNearestFeature();

    return _filterCoarsePatterns();
}