#include <iostream>
#include <iomanip>
// #include "cnn_from_scratch/ModelDescription.h"
#include "cnn_from_scratch/Kernel.h"

int main(int argc, char* argv[]){

    my_cnn::SimpleMatrix<float> M({5, 5, 2});
    int val = 0;
    for (float& v : M){
        v = val++;
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

    my_cnn::SimpleMatrix<float> SSM(M, {0, 0, 0}, {2, 3, 2});
    std::cout << "Rectangular matrix:\n" << SSM;
    SSM *= SSM;
    std::cout << "After squaring:\n" << SSM;
    SSM += 10;
    std::cout << "After offset: \n" << SSM;

    std::cout << "After invalid size operation:\n";
    try{
        std::cout << SSM*OSM;
    }catch(my_cnn::MatrixSizeException& e){
        std::cout << "Failed opeartion: " << e.what();
    }

    return 0;
}