#include <limits>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <assert.h>
#include <iostream>
#include "cnn_from_scratch/Kernel.h"
#include "cnn_from_scratch/Pooling.h"
#include "cnn_from_scratch/imageUtil.h"

int main(int argc, char* argv[]){

    // Grayscale image for testing
    my_cnn::SimpleMatrix<unsigned char> input_image({25, 25, 2});
    
    // Let's make it a 4 for fun
    input_image.subMatView({ 5,  4,  0}, {10,  2,  2}) = 255;
    input_image.subMatView({15,  4,  0}, { 2, 12,  2}) = 255;
    input_image.subMatView({ 7, 12,  0}, {15,  1,  2}) = 255;

    // Create an edge detection kernel
    my_cnn::Kernel K({3, 3, 2}, 1);
    K.weights.subMatView({0, 0, 0}, {3, 1, 1}) = +1;
    K.weights.subMatView({0, 1, 0}, {3, 1, 1}) =  0;
    K.weights.subMatView({0, 2, 0}, {3, 1, 1}) = -1;
    K.weights.subMatView({0, 0, 1}, {1, 3, 1}) = +1;
    K.weights.subMatView({1, 0, 1}, {1, 3, 1}) =  0;
    K.weights.subMatView({2, 0, 1}, {1, 3, 1}) = -1;
    std::cout << "Kernel weights are\n" << K.weights << "\n";

    // No bias in this case
    K.biases[0] = 0;
    K.biases[1] = 0;
    std::cout << "Kernel biases are " << K.biases[0] << " and " << K.biases[1] << "\n";

    // Convert the byte image to float data
    my_cnn::SimpleMatrix<float> floating_point_image = input_image;
    floating_point_image /= 255.0; 

    // Invert the second layer
    auto im2 = floating_point_image.slice(1);
    im2 = im2*-1 + 1; 
    floating_point_image.subMatView({0, 0, 1}, {25, 25, 1}) = im2.data();

    // Display both layers
    std::cout << "Input images: \n";
    K.setInputData(&floating_point_image);
    my_cnn::printImage(floating_point_image.slice(0));
    my_cnn::printImage(floating_point_image.slice(1));

    // Apply the convolution kernel
    my_cnn::SimpleMatrix<float> convolved_image = K.convolve();
    my_cnn::printImage(convolved_image.slice(0));
    my_cnn::printImage(convolved_image.slice(1));

    // Run a 2x2 mean pooling
    my_cnn::Pooling pool;
    pool.dim0 = 2;
    pool.dim1 = 2;
    pool.stride0 = 2;
    pool.stride1 = 2;
    pool.type = my_cnn::AVG;
    my_cnn::SimpleMatrix<float> pooled_image = my_cnn::pooledMatrix(convolved_image, pool);
    my_cnn::printImage(pooled_image.slice(0));
    my_cnn::printImage(pooled_image.slice(1));

    return 0;
}