#include <limits>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <assert.h>
#include <iostream>
#include "cnn_from_scratch/Kernel.h"
#include "cnn_from_scratch/Pooling.h"
#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/ModelDescription.h"

int main(int argc, char* argv[]){

    // Grayscale image for testing
    my_cnn::SimpleMatrix<unsigned char> input_image({25, 25, 2});
    
    // Let's make it a 4 for fun
    input_image[input_image.subMatIdx({ 5,  4,  0}, {10,  2,  1})] = 255;
    input_image[input_image.subMatIdx({15,  4,  0}, { 2, 12,  1})] = 255;
    input_image[input_image.subMatIdx({ 7, 12,  0}, {15,  1,  1})] = 255;
    input_image[input_image.slice(1)] = input_image[input_image.slice(0)];
    // my_cnn::printImage(input_image);

    // Create a model to put the image through
    my_cnn::ModelDescription<unsigned char> model;

    // Create an edge detection kernel
    my_cnn::Kernel K({3, 3, 2}, 1, 1);
    K.weights[K.weights.subMatIdx({0, 0, 0}, {3, 1, 1})] = +1;
    K.weights[K.weights.subMatIdx({0, 1, 0}, {3, 1, 1})] =  0;
    K.weights[K.weights.subMatIdx({0, 2, 0}, {3, 1, 1})] = -1;
    K.weights[K.weights.subMatIdx({0, 0, 1}, {1, 3, 1})] = +1;
    K.weights[K.weights.subMatIdx({1, 0, 1}, {1, 3, 1})] =  0;
    K.weights[K.weights.subMatIdx({2, 0, 1}, {1, 3, 1})] = -1;
    std::cout << "Kernel weights are\n" << K.weights << "\n";

    // No bias in this case
    K.setBias(0, 0);
    // K.setBias(1, 0);
    // std::cout << "Kernel biases are " << K.getBias(0) << " and " << K.getBias(1) << "\n";
    model.addKernel(K, "FirstConvolutionLayer");

    // Convert the byte image to float data
    // my_cnn::SimpleMatrix<float> floating_point_image = input_image;
    // floating_point_image /= 255.0; 

    // Invert the second layer
    // my_cnn::dim3 slice_dim{floating_point_image.dim(0), floating_point_image.dim(1), 1};
    // floating_point_image[floating_point_image.slice(1)] *= my_cnn::SimpleMatrix<float>(slice_dim, -1);
    // floating_point_image[floating_point_image.slice(1)] += my_cnn::SimpleMatrix<float>(slice_dim,  1);

    // Display both layers
    // std::cout << "Input images: \n";
    // my_cnn::printImage(floating_point_image.sliceCopy(0));
    // my_cnn::printImage(floating_point_image.sliceCopy(1));

    // Apply the convolution kernel
    // K.setInputData(&floating_point_image);
    // my_cnn::SimpleMatrix<float> convolved_image = K.convolve();
    // std::cout << "After convolution: \n";
    // my_cnn::printImage(convolved_image.sliceCopy(0));
    // my_cnn::printImage(convolved_image.sliceCopy(1));

    // Run a 2x2 mean pooling
    my_cnn::Pooling pool;
    pool.dim0 = 2;
    pool.dim1 = 2;
    pool.stride0 = 2;
    pool.stride1 = 2;
    pool.type = my_cnn::AVG;
    model.addPooling(pool, "FirstPoolingLayer");
    // my_cnn::SimpleMatrix<float> pooled_image = my_cnn::pooledMatrix(convolved_image, pool);
    // std::cout << "After pooling: \n";
    // my_cnn::printImage(pooled_image.sliceCopy(0));
    // my_cnn::printImage(pooled_image.sliceCopy(1));

    model.run(input_image);

    return 0;
}