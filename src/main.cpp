#include <iostream>
#include <iomanip>
#include <cnn_from_scratch/Kernel.h>

int main(int argc, char* argv[]){

    my_cnn::Kernel K({6, 6, 1}, 1);
    std::cout << "Kernel weights are\n" << K.weights << "\n";

    return 0;
}