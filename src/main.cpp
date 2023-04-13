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
    my_cnn::SimpleMatrix<unsigned char> input_image({28, 28, 1});
    
    // Let's make it a 4 for fun
    input_image[input_image.subMatIdx({ 5,  4,  0}, {10,  2,  1})] = 255;
    input_image[input_image.subMatIdx({15,  4,  0}, { 2, 12,  1})] = 255;
    input_image[input_image.subMatIdx({ 7, 12,  0}, {15,  1,  1})] = 255;

    // Create a model to put the image through
    my_cnn::ModelDescription<unsigned char, std::string> model;

    // Create an edge detection kernel
    my_cnn::Kernel K1({5, 5, 1}, 2, 1);
    K1.weights.subMatView({0, 0, 0}, {5, 1, 1}) = +1;
    K1.weights.subMatView({0, 1, 0}, {5, 3, 1}) =  0;
    K1.weights.subMatView({0, 4, 0}, {5, 1, 1}) = -1;
    K1.weights.subMatView({0, 0, 1}, {1, 5, 1}) = +1;
    K1.weights.subMatView({1, 0, 1}, {3, 5, 1}) =  0;
    K1.weights.subMatView({4, 0, 1}, {1, 5, 1}) = -1;
    std::cout << "Kernel weights are\n" << K1.weights << "\n";

    // No bias in this case
    K1.biases = {0, 0};
    K1.pad_inputs = false;
    K1.activation = my_cnn::RELU;

    // Run a 2x2 max pooling
    my_cnn::Pooling pool;
    pool.dim0 = 2;
    pool.dim1 = 2;
    pool.stride0 = 2;
    pool.stride1 = 2;
    pool.type = my_cnn::MAX;

    // Add another filter layer
    my_cnn::Kernel K2({3, 3, 2}, 4, 1);
    K2.pad_inputs = false;
    K2.activation = my_cnn::SIGMOID;
    K2.biases = {0, 0, 0, 0};
    std::cout << "Second kernel weights are\n" << K2.weights << "\n";

    // Full model description
    model.addKernel(K1, "FirstConvolutionLayer");
    model.addPooling(pool, "FirstPoolingLayer");
    model.addKernel(K2, "SecondConvolutionLayer");
    model.addPooling(pool, "SecondPoolingLayer");
    model.addConnectedLayer(10, "ConnectedLayer");
    model.setOutputLabels({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"});

    my_cnn::ModelResults result = model.forwardPropagation(input_image);
    std::cout << "Model predicted that the image was a " << result.label << "\n";

    return 0;
}