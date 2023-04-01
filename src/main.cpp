#include <limits>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <assert.h>
#include <iostream>
#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/Kernel.h"

int main(int argc, char* argv[]){

    std::cout << "\033[38;2;0;0;0m" << "Black text\033[0m\n";

    // Grayscale image for testing
    my_cnn::SimpleMatrix<unsigned char> input_image({25, 25, 1});
    
    // Let's make it a 4 for fun
    input_image.subMatView({ 5,  4,  0}, {10,  2,  1}) = 255;
    input_image.subMatView({15,  4,  0}, { 2, 12,  1}) = 255;
    input_image.subMatView({ 7, 12,  0}, {15,  1,  1}) = 255;
    printImage(input_image);
    std::cout << input_image << "\n";

    // Convert it to a floating point representation
    my_cnn::SimpleMatrix<float> other_image = input_image/255;
    other_image.conditionallySet(0.5, my_cnn::SimpleMatrix<float>::GREATER, 0);
    // other_image.subMatView() -= 0.25;
    std::cout << other_image << "\n";
    printImage(other_image);

    // Radial Gradient
    my_cnn::SimpleMatrix<float> radial_grad({25, 25, 3});
    for (uint x = 0; x < radial_grad.dim(0); x++){
        for (uint y = 0; y < radial_grad.dim(1); y++){
            radial_grad(x, y, 0) = sqrt(x*x + y*y)/40.0f;
            radial_grad(x, y, 1) = 0.5*float(x)/radial_grad.dim(0);
            radial_grad(x, y, 2) = 0.5*float(y)/radial_grad.dim(1);
        }
    }
    printImage(radial_grad);

    my_cnn::Kernel K({6, 6, 1}, 1);
    std::cout << "Kernel weights are\n" << K.weights << "\n";

    return 0;
}