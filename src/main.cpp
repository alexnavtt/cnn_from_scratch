#include <limits>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <assert.h>
#include <iostream>
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"
#include "cnn_from_scratch/Matrix/MatrixMath.h"
#include "cnn_from_scratch/Kernel.h"
#include "cnn_from_scratch/Pooling.h"
#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/ModelDescription.h"
#include "cnn_from_scratch/MNISTReader.h"

int main(int argc, char* argv[]){ 

    my_cnn::MNISTReader db
        ("../data/MNIST Train Images.ubyte-img", 
        "../data/MNIST Train Labels.ubyte-label");

    for (int i = 0; i < 10; i++){
        my_cnn::Image next_img = db.nextImage();
        std::cout << "Label :" << +next_img.label << "\n";
        my_cnn::printImage(next_img.data);
    }

    my_cnn::Image img_1 = db.getImage(1);
    my_cnn::printImage(img_1.data);
    return 0;

    // Grayscale image for testing
    my_cnn::SimpleMatrix<unsigned char> input_image = db.nextImage().data;
    input_image = db.nextImage().data;
    

    // Create a model to put the image through
    my_cnn::ModelDescription<unsigned char, std::string> model;

    // Create an edge detection kernel
    my_cnn::Kernel K1({5, 5, 1}, 2, 1);
    // K1.weights.subMatView({0, 0, 0}, {5, 1, 1}) = +1;
    // K1.weights.subMatView({0, 1, 0}, {5, 3, 1}) =  0;
    // K1.weights.subMatView({0, 4, 0}, {5, 1, 1}) = -1;
    // K1.weights.subMatView({0, 0, 1}, {1, 5, 1}) = +1;
    // K1.weights.subMatView({1, 0, 1}, {3, 5, 1}) =  0;
    // K1.weights.subMatView({4, 0, 1}, {1, 5, 1}) = -1;
    std::cout << "Kernel weights are\n" << K1.weights << "\n";

    // No bias in this case
    // K1.biases = {0, 0};
    K1.pad_inputs = false;
    K1.activation = my_cnn::RELU;

    // Run a 2x2 max pooling with a stride of 2x2
    my_cnn::Pooling pool({2, 2}, {2, 2}, my_cnn::MAX);

    // Add another filter layer
    my_cnn::Kernel K2({3, 3, 2}, 4, 1);
    K2.pad_inputs = false;
    K2.activation = my_cnn::SIGMOID;
    // K2.biases = {0, 0, 0, 0};
    std::cout << "Second kernel weights are\n" << K2.weights << "\n";

    // Full model description
    model.addKernel(K1, "FirstConvolutionLayer");
    model.addPooling(pool, "FirstPoolingLayer");
    model.addKernel(K2, "SecondConvolutionLayer");
    model.addPooling(pool, "SecondPoolingLayer");
    model.addConnectedLayer(10, "ConnectedLayer");
    model.setOutputLabels({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"});

    my_cnn::ModelResults result = model.forwardPropagation(input_image, &model.output_labels[4]);
    std::cout << "Model predicted that the image was a " << result.label << "\n";

    model.backwardsPropagation(result, 0.05);

    return 0;
}