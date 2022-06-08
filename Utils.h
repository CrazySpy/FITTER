//
// Created by jiang on 2022/6/5.
//

#ifndef EMBEDDINGBASED_UTILS_H
#define EMBEDDINGBASED_UTILS_H

#include <vector>
#include <Eigen/Dense>

Eigen::MatrixXi oneHot(Eigen::VectorXi indices, int depth);

unsigned int randomInt(unsigned int low, unsigned int high);

#endif //EMBEDDINGBASED_UTILS_H
