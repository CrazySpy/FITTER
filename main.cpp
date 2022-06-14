#include <iostream>
#include <string>
#include <sstream>

#include "EmbeddingBased.h"
#include "Types.h"

using namespace std;

int main(int argc, char **argv) {
    if(argc != 3) {
        cout << "Parameter format :\n" << argv[0] << " <prevalent pattern dataset path> <sample size>\n";
        return 0;
    }

    string prevalentPatternDataset(argv[1]);
    unsigned int sampleSize = stoul(argv[2]);

    // Read prevalent patterns.
    std::vector<ColocationType> prevalentPatterns;

    ifstream ifs(prevalentPatternDataset);
    string line;
    while(getline(ifs, line)) {
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
    cout << "Number of prevalent pattern: " << prevalentPatterns.size() << endl;

    EmbeddingBased embedding(prevalentPatterns, sampleSize);
    auto preferredPatterns = embedding.execute();

    Simulator * simulator = embedding.getSimulator();

    ConfusionMatrixType confusionMatrix = simulator->evaluate(preferredPatterns);
    double precision = confusionMatrix.TP * 1.0 / (confusionMatrix.TP + confusionMatrix.FP);
    double recall = confusionMatrix.TP * 1.0 / (confusionMatrix.TP + confusionMatrix.FN);
    double fScore = 2 * precision * recall / (precision + recall);

    cout << "Answer time : " << simulator->getAnswerTime() << endl;
    cout << "Answer interesting time : " << simulator->getPositiveAnswerTime() << endl;
    cout << confusionMatrix << endl;
    cout << "precision : " << precision << endl;
    cout << "recall : " << recall << endl;
    cout << "F-score : " << fScore << endl;

    return 0;
}
