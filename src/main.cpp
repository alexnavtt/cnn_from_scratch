#include <limits>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <assert.h>
#include <iostream>
#include <chrono>
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"
#include "cnn_from_scratch/Matrix/MatrixMath.h"
#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/ModelDescription.h"
#include "cnn_from_scratch/MNISTReader.h"
#include "cnn_from_scratch/LoadingBar.h"

// Define global timer used by all translation units
cpp_timer::Timer global_timer;

using namespace std::string_literals;
static const std::string data_dir(DATA_DIR); 

template<typename ModelType>
void debugLayerWeights(const ModelType& model){
    int ii = 0;
    for (const auto& layer : model.layers){
        auto w_norm = my_cnn::l2Norm(layer->weights);
        auto b_norm = my_cnn::l2Norm(layer->biases);

        std::cout << "Layer " << ii << " has weights of norm " << w_norm << "\n";
        std::cout << "Layer " << ii++ << " has biases of norm " << b_norm << "\n";
    }
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

template<typename T>
std::ostream& operator<<(std::ostream& os, const my_cnn::ModelResults<T>& mr){
    os << "Layer Results:\n";
    os << "\tLabel Index: " << mr.label_idx << "\n";
    os << "\tConfidence:  " << mr.confidence << "\n";
    if (mr.knows_true_value){
        os << "\tLoss:        " << mr.loss << "\n";
        os << "\tTrue value:  " << std::to_string(mr.true_label) << "\n";
        os << "\tTrue index:  " << mr.true_label_idx << "\n";
    }
    os << "\n";
    return os;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

template<typename ModelType>
size_t runEpoch(ModelType& model, my_cnn::MNISTReader& image_source, size_t training_size, double learning_rate){
    size_t correct_count = 0;
    double total_loss = 0.0;
    for (size_t i = 0; i < training_size; i++){
        auto Data = image_source.getImage(i);

        // Run forwards and backwards propagation
        my_cnn::ModelResults result = model.forwardPropagation(Data.data);
        model.backwardsPropagation(result, Data.label, learning_rate);
        total_loss += result.loss;

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
        loadingBar(i, training_size);
    }
    std::cout << "\n";
    std::cout << "Average loss was " << total_loss/training_size << " and ";
    return correct_count;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

template<typename ModelType>
void validate(ModelType& model){
    STIC;

    my_cnn::MNISTReader db
        (DATA_DIR + "/MNIST Test Images.ubyte-img"s, 
         DATA_DIR + "/MNIST Test Labels.ubyte-label"s);  

    std::cout << "Running model validation test...\n";

    int correct_count = 0;
    int num_images = db.numImages();
    for (int i = 0; i < num_images; i++){
        auto Data = db.getImage(i);

        // Run forwards and backwards propagation
        my_cnn::ModelResults result = model.forwardPropagation(Data.data);
        correct_count += model.output_labels[result.label_idx] == Data.label;
        loadingBar(i, num_images);
    }

    std::cout << "\n\n";

    for (int i = 0; i < 5; i++){
        int idx = (double)(rand())/RAND_MAX * num_images;
        auto Data = db.getImage(idx);

        my_cnn::ModelResults result = model.forwardPropagation(Data.data);
        std::cout << "This image is a " << +model.output_labels[result.label_idx] << " (actual label " << +Data.label << ")\n";
        my_cnn::printImage(Data.data);
        std::cout << "\n\n";
    }

    std::cout << "\nValidation accuracy was " << 100.0f * correct_count/num_images << "%\n";
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

int main(int argc, char* argv[]){ 

    // Load the database images and labels
    my_cnn::MNISTReader db
        (data_dir + "/MNIST Train Images.ubyte-img"s, 
         data_dir + "/MNIST Train Labels.ubyte-label"s);        

    // Create a model to put the images through
    my_cnn::ModelDescription<float, unsigned char> model;

    // Full model description
    model.addKernel ({5, 5, 1},  2);
    model.addActivation(my_cnn::RELU);
    model.addPooling({2, 2}, {2, 2}, my_cnn::MAX);
    model.addKernel ({3, 3, 2},  4);
    model.addActivation(my_cnn::RELU);
    model.addPooling({2, 2}, {2, 2}, my_cnn::MAX);
    model.addConnectedLayer(10);
    model.setOutputLabels({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, my_cnn::SOFTMAX);

    int num_epochs = 1;
    int training_size = 60000;
    for (int j = 0; j < num_epochs; j++){
        std::cout << "Starting epoch " << j << ":\n";
        int correct_count = runEpoch(model, db, training_size, 0.0001);
        std::cout << "Accuracy was " << 100.0*correct_count/training_size << "%\n";
    }

    validate(model);
    debugLayerWeights(model);

    if (argc < 2){
        std::cout << "No file provided, model training results will not be saved\n";
    }else{
        std::string filename = data_dir + "/" + std::string(argv[1]) + ".model_data";
        std::cout << "Saving model to \"" << filename << "\"\n";
        model.saveModel(filename);
    }
    global_timer.summary();

    return 0;
}