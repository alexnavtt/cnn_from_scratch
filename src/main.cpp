#include <iostream>
#include <iomanip>
// #include "cnn_from_scratch/ModelDescription.h"
#include "cnn_from_scratch/Kernel.h"

int main(int argc, char* argv[]){

    my_cnn::SimpleMatrix<float> M({5, 5, 2});
    int val = 0;
    for (int j = 0; j < 5; j++){
        for (int i = 0; i < 5; i++){
            M(i, j) = val++;
        }
    }

    std::cout << "Initial Matrix:\n";
    std::cout << M;

    my_cnn::SimpleMatrix<float> SM(M, {1, 2, 0}, {3, 3, 1});

    std::cout << "Sub-Matrix:\n" << SM;

    my_cnn::SimpleMatrix<float> OSM(M, {0, 0, 0}, {3, 3, 1});

    std::cout << "Before Modification:\n" << OSM;
    OSM += SM;

    std::cout << "After Modification:\n" << OSM;

    std::cout << "Initial Matrix Again:\n" << M;

    return 0;
}