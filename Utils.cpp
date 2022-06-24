//
// Created by jiang on 2022/6/5.
//

#include <random>

#include "Utils.h"

#include <Eigen/Dense>

Eigen::MatrixXi oneHot(Eigen::VectorXi indices, int depth) {
    Eigen::MatrixXi oneHotMatrix = Eigen::MatrixXi::Zero(indices.size(), depth);

    for(auto i = 0; i < indices.size(); ++i) {
        //oneHotMatrix(i, i) = 1;
        oneHotMatrix(i, Eigen::indexing::all)[indices(i)] = 1;
    }

    return oneHotMatrix;
}

unsigned int randomInt(unsigned int low, unsigned int high) {
    std::mt19937 mt(std::random_device{}());
    auto dist = std::uniform_int_distribution<unsigned int>(low, high);
    return dist(mt);
}