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
#include "cnn_from_scratch/LoadingBar.h"

// Define global timer used by all translation units
cpp_timer::Timer global_timer;

template<typename ModelType, typename ImageSource>
size_t runBatch(ModelType& model, ImageSource& image_source, size_t batch_size, double learning_rate){
    size_t correct_count = 0;
    for (size_t i = 0; i < batch_size; i++){
        auto Data = image_source.getImage(i);

        // Run forwards and backwards propagation
        my_cnn::ModelResults result = model.forwardPropagation(Data.data, &Data.label);
        model.backwardsPropagation(result, learning_rate);

        // Try to catch where it happens
        for (const auto& layer : model.layers){
            auto w_norm = my_cnn::l2Norm(layer->weights);
            auto b_norm = my_cnn::l2Norm(layer->biases);

            if  (std::isnan(w_norm) || std::isinf(w_norm) || std::isnan(b_norm) || std::isinf(b_norm)){
                std::cout << "Layer " << layer->name << " has weights of norm " << w_norm << "\n";
                std::cout << "Layer " << layer->name << " has biases of norm " << b_norm << "\n";
                throw(std::runtime_error(""));
            }
        }

        correct_count += model.output_labels[result.label_idx] == Data.label;
        loadingBar(i, batch_size);
    }
    std::cout << "\n";
    return correct_count;
}

int main(int argc, char* argv[]){ 

    my_cnn::MNISTReader db
        ("../data/MNIST Train Images.ubyte-img", 
        "../data/MNIST Train Labels.ubyte-label");        

    // Create a model to put the image through
    my_cnn::ModelDescription<float, unsigned char> model;

    // Full model description
    model.addKernel ({5, 5, 1},  2    , my_cnn::SIGMOID);
    model.addPooling({2, 2}   , {2, 2}, my_cnn::MAX);
    model.addKernel ({3, 3, 2},  4    , my_cnn::SIGMOID);
    model.addPooling({2, 2}   , {2, 2}, my_cnn::MAX);
    model.addConnectedLayer(10);
    model.setOutputLabels({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    int num_epochs = 10;
    int batch_size = 500;
    for (int j = 0; j < num_epochs; j++){
        
        int correct_count = runBatch(model, db, batch_size, 0.001);

        auto max_element = my_cnn::max(model.layers.back()->weights);
        auto min_element = my_cnn::min(model.layers.back()->weights);

        for (const auto& layer : model.layers){
            for (size_t ii = 0; ii < layer->weights.dim().z; ii++)
                my_cnn::printImage(layer->weights.slice(ii));
            std::cout << "\n";
        }
        std::cout << "Connected layer weights range from " << min_element << " to " << max_element << "\n";

        int ii = 0;
        for (const auto& layer : model.layers){
            auto w_norm = my_cnn::l2Norm(layer->weights);
            auto b_norm = my_cnn::l2Norm(layer->biases);

            std::cout << "Layer " << ii << " has weights of norm " << w_norm << "\n";
            std::cout << "Layer " << ii++ << " has biases of norm " << b_norm << "\n";
        }

        std::cout << "Accuracy was " << 100.0*correct_count/batch_size << "%\n";

        db.getImage(0);
    }

    auto Data = db.getImage(25001);
    my_cnn::printImage(Data.data);
    auto result = model.forwardPropagation(Data.data);

    for (const auto& layer : model.layers){
        if (layer->weights.dim().z == 0) continue;
        std::cout << "Layer " << layer->name << "\n";
        std::cout << "Max: " << my_cnn::max(layer->weights) << " and min: " << my_cnn::min(layer->weights) << "\n";
        std::cout << layer->weights;
        for (size_t z = 0; z < layer->weights.dim().z; z++)
            my_cnn::printImage(layer->weights.slice(z));
    }

    for (const auto& res : result.layer_inputs){
        for (size_t z = 0; z < res.dim().z; z++)
            my_cnn::printImage(res.slice(z));
    }
    std::cout << "This image was predicted to be a " << +model.output_labels[result.label_idx] << " with confidence of " << result.confidence << "\n";
    // my_cnn::printImage(Data.data);

    global_timer.summary();

    // std::cout << "Model predicted that the image was a " << result.label << "\n";


    return 0;
}