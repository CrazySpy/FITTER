//
// Created by jiang on 2022/5/31.
//

#ifndef EMBEDDINGBASED_TYPES_H
#define EMBEDDINGBASED_TYPES_H

#include <vector>
#include <string>
#include <fstream>
#include <numeric>

using InstanceIdType = std::string;
using FeatureType = std::string;
using LocationType = std::pair<double, double>;

struct InstanceType {
    FeatureType feature;
    InstanceIdType id;
    LocationType location;
    bool operator == (const InstanceType &rhs) const {
        return feature == rhs.feature && id == rhs.id;
    }

    bool operator < (const InstanceType &rhs) const {
        return feature == rhs.feature ? id < rhs.id : feature < rhs.feature;
    }
};

using InstanceNameType = std::pair<FeatureType, InstanceIdType>;

using ColocationType = std::vector<FeatureType>;
using ColocationSetType = std::vector<ColocationType>;

using GeneralizedConceptType = std::string;
using GeneralizedConceptSetType = std::vector<GeneralizedConceptType>;

struct ConfusionMatrixType {
    unsigned int TP, TN, FP, FN;
    friend std::ostream& operator << (std::ostream &ofs, const ConfusionMatrixType &confusionMatrix) {
        ofs << "\t\t\t" << "Actual:like" << "\t\tActual:dislike" << std::endl;
        ofs << "predict:like\t\t" << confusionMatrix.TP << "\t\t\t" << confusionMatrix.FP << std::endl;
        ofs << "predict:dislike\t\t" << confusionMatrix.FN << "\t\t\t" << confusionMatrix.TN << std::endl;

        return ofs;
    }
};

#endif //EMBEDDINGBASED_TYPES_H
