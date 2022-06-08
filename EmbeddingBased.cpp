//
// Created by jiang on 2022/5/31.
//

#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "EmbeddingBased.h"
#include "Utils.h"

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>

EmbeddingBased::EmbeddingBased(const std::vector<ColocationType> &prevalentPatterns,
                               unsigned int sampleSize, Simulator *simulator)
    : _prevalentPatterns(prevalentPatterns),
      _sampleSize(sampleSize),
      _simulator(simulator) {
    for(auto &prevalentPattern : _prevalentPatterns) {
        for(int i = 0; i < prevalentPattern.size(); ++i) {
            _features.push_back(prevalentPattern[i]);
        }
    }

    std::sort(_features.begin(), _features.end());
    _features.erase(std::unique(_features.begin(), _features.end()), _features.end());

    for(int i = 0; i < _features.size(); ++i) {
        _featureIndex[_features[i]] = i;
    }

    _indicatorMatrix = Eigen::MatrixXi(_features.size(), 0);

    _constructCoOccurrenceMatrix();
    _constructIndicatorMatrix();
    _constructEmbeddingRepresentation();

    _generateSampleRank();

    // Construct simulator.
    if(_simulator == nullptr) {
        _simulator = new Simulator(_prevalentPatterns, _features, _embeddingRepresentation);
    }
}

double EmbeddingBased::_calculateCoOccurrenceValue(const FeatureType &feature1, const FeatureType &feature2) {
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

void EmbeddingBased::_constructCoOccurrenceMatrix() {
    _coOccurrenceMatrix = Eigen::MatrixXd(_features.size(), _features.size());
    for(int i = 0; i < _features.size(); ++i) {
        for(int j = 0; j < _features.size(); ++j) {
            _coOccurrenceMatrix(i, j) = _calculateCoOccurrenceValue(_features[i], _features[j]);
        }
    }
}

void EmbeddingBased::_constructIndicatorMatrix() {
    cv::Mat cvCoOccurrenceMatrix;
    Eigen::MatrixXf eigenCoOccurrenceMatrix = _coOccurrenceMatrix.cast<float>();

    cv::eigen2cv(eigenCoOccurrenceMatrix, cvCoOccurrenceMatrix);

    for(int k = 2; _features.size() >= k; k++) {
        cv::Mat labels, centers;
        cv::kmeans(cvCoOccurrenceMatrix, k, labels,
                   cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
                   3, cv::KMEANS_PP_CENTERS, centers);

        Eigen::MatrixXi eigenLabels;
        cv::cv2eigen(labels, eigenLabels);
        Eigen::MatrixXi eigenOneHot = oneHot(eigenLabels, centers.rows);

        // Erase small clusters.
        int smallClusterCount = 0;
        Eigen::VectorXi pointNum = eigenOneHot.colwise().sum();
        std::vector<int> bigClusterIndices;
        for(int i = 0; i < pointNum.size(); ++i) {
            if(pointNum(i) <= 1) {
                ++smallClusterCount;
            } else {
                bigClusterIndices.push_back(i);
            }
        }

        // Concat the indicator matrix.
        Eigen::MatrixXi indicatorMatrix(_features.size(), _indicatorMatrix.cols() + bigClusterIndices.size());
        indicatorMatrix << _indicatorMatrix, eigenOneHot(Eigen::all, bigClusterIndices);

        _indicatorMatrix = indicatorMatrix;

        // Quit loop condition.
        if(smallClusterCount > std::ceil(k / _alpha)) {
            break;
        }
    }
}

void EmbeddingBased::_constructEmbeddingRepresentation() {
    Eigen::MatrixXd centered = _indicatorMatrix.cast<double>().rowwise() - _indicatorMatrix.cast<double>().colwise().mean();
    Eigen::MatrixXd cov = (centered.transpose() * centered) / double(centered.cols() - 1);

    Eigen::BDCSVD<Eigen::MatrixXd> svd(cov, Eigen::ComputeFullV);
    Eigen::MatrixXd V = svd.matrixV();

    Eigen::MatrixXd N = centered * V.transpose();
    std::vector<int> goodNColumnIndex;
    for(int i = 0; i < N.cols(); ++i) {
        Eigen::VectorXd col1 = N(Eigen::all, i);
        double maxDistance = 0;
        for(int j = 0; j < N.cols(); ++j) {
            Eigen::VectorXd col2 = N(Eigen::all, j);
            double distance = (col1 - col2).lpNorm<2>();
            maxDistance = std::max(maxDistance, distance);
        }

        if(maxDistance >= _beta) {
            goodNColumnIndex.push_back(i);
        }
    }

    _embeddingRepresentation = N(Eigen::all, goodNColumnIndex);
}

double EmbeddingBased::_calculateSemanticDistance(const ColocationType &pattern1, const ColocationType &pattern2) {
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

double EmbeddingBased::_calculateSingularity(const ColocationType &pattern) {
    // $D(P) = max_{f \in P}(1 / \Vert c(f) \Vert)$
    // c(f) is the co-occurrence vector of feature f.

    std::vector<unsigned int> featureIndices;
    for(auto &feature : pattern) {
        featureIndices.push_back(_featureIndex[feature]);
    }

    auto coOccurrenceMatrix = _coOccurrenceMatrix(featureIndices, Eigen::all);
    double singularity = coOccurrenceMatrix.rowwise().lpNorm<2>().cwiseInverse().maxCoeff();

    return singularity;
}

double EmbeddingBased::_calculateUniversality(const ColocationType &pattern) {
    std::vector<unsigned int> featureIndices;
    for(auto &feature : pattern) {
        featureIndices.push_back(_featureIndex[feature]);
    }

    auto coOccurrenceMatrix = _coOccurrenceMatrix(featureIndices, featureIndices);

    return coOccurrenceMatrix.norm() / pattern.size();
}

void EmbeddingBased::_generateSampleRank() {
    for(auto &pattern : _prevalentPatterns) {
        double singularity = _calculateSingularity(pattern);
        double universality = _calculateUniversality(pattern);
        double sampleRank = singularity * universality;

        _singularity[pattern] = singularity;
        _universality[pattern] = universality;
        _sampleRank[pattern] = sampleRank;
    }
}


std::vector<ColocationType> EmbeddingBased::_samplePatterns(std::vector<ColocationType> &candidatePatterns) {
    if(_sampleSize >= candidatePatterns.size()) {
        auto sample = candidatePatterns;
        candidatePatterns.clear();
        return sample;
    }

    // Store the final sample result.
    std::vector<ColocationType> sample;
    std::vector<std::pair<double, ColocationType>> candidateSampleRank;
    for(auto &pattern : candidatePatterns) {
        candidateSampleRank.push_back({_sampleRank[pattern], pattern});
    }
    std::sort(candidateSampleRank.begin(), candidateSampleRank.end(), std::less());

    for(int i = 0; i < _sampleSize; ++i) {
        auto &pattern = candidateSampleRank[i].second;
        sample.push_back(pattern);
        candidatePatterns.erase(std::remove(std::begin(candidatePatterns), std::end(candidatePatterns), pattern), std::end(candidatePatterns));
    }

    return sample;
}

std::vector<bool> EmbeddingBased::_interactiveSurvey(const std::vector<ColocationType> &samplePatterns) {
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

void EmbeddingBased::_selectCoarsePatterns(std::vector<ColocationType> &candidatePatterns,
                                           const std::vector<ColocationType> &preferredPatterns) {
    // First, insert positive feedback pattern into coarse set.
    _coarsePatterns.insert(_coarsePatterns.end(), std::begin(preferredPatterns), std::end(preferredPatterns));

    // Then, find the similarity of each preferred pattern.
    std::vector<ColocationType> similarities;
    for(int i = 0; i < preferredPatterns.size(); ++i) {
        double minDistance = -1;
        ColocationType similarity;
        for(int j = 0; j < candidatePatterns.size(); ++j) {
            double distance = _calculateSemanticDistance(preferredPatterns[i], candidatePatterns[j]);
            if(minDistance < 0 || distance < minDistance) {
                minDistance = distance;
                similarity = candidatePatterns[j];
            }
        }
        similarities.push_back(similarity);
    }

    std::sort(similarities.begin(), similarities.end());
    similarities.erase(std::unique(similarities.begin(), similarities.end()), similarities.end());

    auto candidateEnd = candidatePatterns.end();
    for(auto &similarity : similarities) {
        candidateEnd = std::remove(std::begin(candidatePatterns), candidateEnd, similarity);
        _coarsePatterns.push_back(similarity);
    }

    candidatePatterns.erase(candidateEnd, candidatePatterns.end());
}

void EmbeddingBased::_filterDislikePatterns(std::vector<ColocationType> &candidatePatterns,
                                            const std::vector<ColocationType> &dislikePatterns) {
    std::vector<ColocationType> similarities;
    for(int i = 0; i < dislikePatterns.size(); ++i) {
        double minDistance = -1;
        ColocationType similarity;
        for(int j = 0; j < candidatePatterns.size(); ++j) {
            double distance = _calculateSemanticDistance(dislikePatterns[i], candidatePatterns[j]);
            if(minDistance < 0 || distance < minDistance) {
                minDistance = distance;
                similarity = candidatePatterns[j];
            }
        }
        similarities.push_back(similarity);
    }

    auto candidateEnd = candidatePatterns.end();
    for(auto &similarity : similarities) {
        candidateEnd = std::remove(std::begin(candidatePatterns), candidateEnd, similarity);
    }
    candidatePatterns.erase(candidateEnd, candidatePatterns.end());
}

std::vector<ColocationType> EmbeddingBased::_filterCoarsePatterns() {
    return _coarsePatterns;
}

Simulator *EmbeddingBased::getSimulator() {
    return _simulator;
}

std::vector<ColocationType> EmbeddingBased::execute() {
    std::vector<ColocationType> candidatePatterns = _prevalentPatterns;
    while(!candidatePatterns.empty()) {
        auto sample = _samplePatterns(candidatePatterns);
        auto feedback = _interactiveSurvey(sample);

        // Generate preferred and dislike patterns.
        std::vector<ColocationType> preferredPatterns;
        std::vector<ColocationType> dislikePatterns;
        for(int i = 0; i < feedback.size(); ++i) {
            if(feedback[i]) {
                preferredPatterns.push_back(sample[i]);
            } else {
                dislikePatterns.push_back(sample[i]);
            }
        }

        _selectCoarsePatterns(candidatePatterns, preferredPatterns);
        _filterDislikePatterns(candidatePatterns, dislikePatterns);
    }

    std::sort(_coarsePatterns.begin(), _coarsePatterns.end());

    return _filterCoarsePatterns();
}

EmbeddingBased::~EmbeddingBased() {
    delete _simulator;
}