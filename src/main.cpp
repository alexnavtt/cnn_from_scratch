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

int main(int argc, char* argv[]){ 

    my_cnn::MNISTReader db
        ("../data/MNIST Train Images.ubyte-img", 
        "../data/MNIST Train Labels.ubyte-label");        

    // Create a model to put the image through
    my_cnn::ModelDescription<float, unsigned char> model;

    // Full model description
    model.addKernel ({5, 5, 1},  2    , my_cnn::RELU);
    model.addPooling({2, 2}   , {2, 2}, my_cnn::MAX);
    model.addKernel ({3, 3, 2},  4    , my_cnn::LEAKY_RELU);
    model.addPooling({2, 2}   , {2, 2}, my_cnn::MAX);
    model.addConnectedLayer(10);
    model.setOutputLabels({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    int num_epochs = 1;
    int batch_size = 100;
    bool failed = false;
    for (int j = 0; j < num_epochs; j++){
        int correct_count = 0;
        // std::cout << "Starting epoch " << j << "\n";
        for (int i = 0; i < batch_size; i++){
            auto Data = db.getImage(i);

            my_cnn::ModelResults result = model.forwardPropagation(Data.data, &Data.label);
            if(std::isnan(result.loss) || std::isinf(result.loss)){
                failed = true;
                break;
            }
            // std::cout << "Loss was " << result.loss << "\n";
            model.backwardsPropagation(result, 0.01);

            // std::cout << "This image was a " << +Data.label << " and was predicted to be a " << +model.output_labels[result.label_idx] << "\n";

            correct_count += model.output_labels[result.label_idx] == Data.label;
            loadingBar(i, batch_size);
        }
        std::cout << "\n";

        int ii = 0;
        for (const auto& layer : model.layers){
            std::cout << "Layer " << ii << " has weights of norm " << my_cnn::l2Norm(layer->weights) << "\n";
            std::cout << "Layer " << ii++ << " has biases of norm " << my_cnn::l2Norm(layer->biases) << "\n";
        }

        std::cout << "Accuracy was " << 100.0*correct_count/batch_size << "%\n";

        db.getImage(0);
        if (failed) break;
    }

    auto Data = db.getImage(25001);
    my_cnn::printImage(Data.data);
    auto result = model.forwardPropagation(Data.data);
    std::cout << "This image was predicted to be a " << +model.output_labels[result.label_idx] << " with confidence of " << result.confidence << "\n";
    // my_cnn::printImage(Data.data);

    global_timer.summary();

    // std::cout << "Model predicted that the image was a " << result.label << "\n";


    return 0;
}