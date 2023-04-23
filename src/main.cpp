#include <limits>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <assert.h>
#include <iostream>
#include <chrono>
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"
#include "cnn_from_scratch/Matrix/MatrixMath.h"
#include "cnn_from_scratch/Kernel.h"
#include "cnn_from_scratch/Pooling.h"
#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/ModelDescription.h"
#include "cnn_from_scratch/MNISTReader.h"

cpp_timer::Timer global_timer;

int main(int argc, char* argv[]){ 

    my_cnn::MNISTReader db
        ("../data/MNIST Train Images.ubyte-img", 
        "../data/MNIST Train Labels.ubyte-label");        

    // Create a model to put the image through
    my_cnn::ModelDescription<unsigned char, unsigned char> model;

    // Create a kernel 5x5 kernel with 2 output channels and a stride of 1
    auto K1 = std::make_shared<my_cnn::Kernel>(my_cnn::dim3(5, 5, 1), 2, 1);
    K1->activation = my_cnn::RELU;

    // Run a 2x2 max pooling with a stride of 2x2
    auto pool = std::make_shared<my_cnn::Pooling>(my_cnn::dim2(2, 2), my_cnn::dim2(2, 2), my_cnn::MAX);

    // Add another filter layer
    auto K2 = std::make_shared<my_cnn::Kernel>(my_cnn::dim3(3, 3, 2), 4, 1);
    K2->activation = my_cnn::SIGMOID;

    // Full model description
    model.addKernel(K1, "FirstConvolutionLayer");
    model.addPooling(pool, "FirstPoolingLayer");
    model.addKernel(K2, "SecondConvolutionLayer");
    model.addPooling(pool, "SecondPoolingLayer");
    model.addConnectedLayer(10, "ConnectedLayer");
    model.setOutputLabels({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    int batch_size = 1;
    for (int j = 0; j < 2; j++){
        int correct_count = 0;
        for (int i = 0; i < batch_size; i++){
            global_timer.tic("getImage");
            auto Data = db.nextImage();
            global_timer.toc("getImage");

            my_cnn::ModelResults result = model.forwardPropagation(Data.data, &Data.label);
            model.backwardsPropagation(result, 0.01);

            correct_count += result.label == Data.label;
        }
        std::cout << ": Accuracy was " << 100.0*correct_count/batch_size << "%\n";
    }

    auto Data = db.nextImage();
    auto result = model.forwardPropagation(Data.data);
    std::cout << "This image was predicted to be a " << +result.label << "\n";
    my_cnn::printImage(Data.data);

    global_timer.summary();

    // std::cout << "Model predicted that the image was a " << result.label << "\n";


    return 0;
}