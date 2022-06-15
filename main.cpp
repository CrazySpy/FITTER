#include <iostream>
#include <string>

#include "EmbeddingBased.h"
#include "Types.h"

using namespace std;

int main(int argc, char **argv) {
    if(argc != 4) {
        cout << "Parameter format :\n" << argv[0] << " <prevalent pattern dataset path> <preferred pattern dataset path> <sample size>\n";
        return 0;
    }

    string prevalentPatternDataset(argv[1]);
    string preferredPatternDataset(argv[2]);
    unsigned int sampleSize = stoul(argv[3]);

    // Read prevalent patterns.
    std::vector<ColocationType> prevalentPatterns;

    ifstream prevalentPatternReader(prevalentPatternDataset);
    string line;
    while(getline(prevalentPatternReader, line)) {
        ColocationType prevalentPattern;

        int start = 0;
        while(start < line.size()) {
            int end = line.find_first_of(',', start);
            if(end == string::npos) end = line.size();
            string feature = line.substr(start, end - start);
            prevalentPattern.push_back(feature);
            start = end + 1;
        }

        prevalentPatterns.push_back(prevalentPattern);
    }
    std::sort(prevalentPatterns.begin(), prevalentPatterns.end());
    prevalentPatterns.erase(std::unique(prevalentPatterns.begin(), prevalentPatterns.end()), prevalentPatterns.end());
    cout << "Number of prevalent pattern: " << prevalentPatterns.size() << endl;

    // Read preferred patterns.
    std::vector<std::vector<FeatureType>> preferredPatterns;

    ifstream preferredPatternReader(preferredPatternDataset);
    while(getline(preferredPatternReader, line)) {
        std::vector<FeatureType> preferredPattern;

        int start = 0;
        while(start < line.size()) {
            int end = line.find_first_of(',', start);
            if(end == string::npos) end = line.size();
            string feature = line.substr(start, end - start);
            preferredPattern.push_back(feature);
            start = end + 1;
        }

        preferredPatterns.push_back(preferredPattern);
    }

    Simulator simulator = Simulator(prevalentPatterns, preferredPatterns);

    EmbeddingBased embedding(prevalentPatterns, sampleSize, &simulator);
    auto predictPatterns = embedding.execute();

    ConfusionMatrixType confusionMatrix = simulator.evaluate(predictPatterns);
    double precision = confusionMatrix.TP * 1.0 / (confusionMatrix.TP + confusionMatrix.FP);
    double recall = confusionMatrix.TP * 1.0 / (confusionMatrix.TP + confusionMatrix.FN);
    double fScore = 2 * precision * recall / (precision + recall);

    cout << "Answer time : " << simulator.getAnswerTime() << endl;
    cout << "Answer interesting time : " << simulator.getPositiveAnswerTime() << endl;
    cout << confusionMatrix << endl;
    cout << "precision : " << precision << endl;
    cout << "recall : " << recall << endl;
    cout << "F-score : " << fScore << endl;

    return 0;
}
