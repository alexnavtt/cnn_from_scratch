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
#include "cnn_from_scratch/DataGenerator.h"

using namespace std::string_literals;
static const std::string data_dir(DATA_DIR); 

#ifdef TIMEIT
extern cpp_timer::Timer global_timer;
#endif

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
    if (mr.true_label){
        os << "\tLoss:        " << mr.loss << "\n";
        os << "\tTrue value:  " << mr.true_label.value() << "\n";
        os << "\tTrue index:  " << mr.true_label_idx << "\n";
    }
    os << "\n";
    return os;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

// Run through the MNIST test dataset for validation
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
        correct_count += model.output_labels[result.label_idx] == std::to_string(Data.label);
        loadingBar(i, num_images);
    }

    std::cout << "\n\n";

    for (int i = 0; i < 5; i++){
        int idx = (double)(rand())/RAND_MAX * num_images;
        auto Data = db.getImage(idx);

        my_cnn::ModelResults result = model.forwardPropagation(Data.data);
        std::cout << "This image is a " << model.output_labels[result.label_idx] << " (actual label " << +Data.label << ")\n";
        my_cnn::printImage(Data.data);
        std::cout << result << "\n\n";
    }

    std::cout << "\nValidation accuracy was " << 100.0f * correct_count/num_images << "%\n";
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

// Derive a class from DataGenerator to interface with the model for getting the data
class MNistGenerator : public my_cnn::DataGenerator<float> {
public:
    MNistGenerator(std::string image_filepath, std::string label_filepath) :
    db_(image_filepath, label_filepath)
    {
        reset();
    }

    my_cnn::LabeledInput<float> getNextDataPoint() override {
        my_cnn::LabeledInput<float> labeled_input;
        auto Data = db_.nextImage();
        labeled_input.data = Data.data;
        labeled_input.label = std::to_string(Data.label);
        return labeled_input;
    }

    size_t size() override {
        return db_.numImages() - 1;
    }

    bool hasAvailableData() override {
        return db_.imageIndex() < db_.numImages();
    }

    void reset() override {
        db_.getImage(0);
    }

private:
    my_cnn::MNISTReader db_;
};

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

int main(int argc, char* argv[]){   

    // Load the MNIST training data from disk
    MNistGenerator data_source(
         data_dir + "/MNIST Train Images.ubyte-img"s, 
         data_dir + "/MNIST Train Labels.ubyte-label"s);   

    // Create a model to put the images through
    using Model = my_cnn::ModelDescription<float, std::string>;
    Model model;

    // Full model description
    model.addKernel ({5, 5, 1},  2);
    model.addActivation(my_cnn::SIGMOID);
    model.addPooling({2, 2}, {2, 2}, my_cnn::MAX);
    model.addKernel ({3, 3, 2},  4);
    model.addActivation(my_cnn::RELU);
    model.addPooling({2, 2}, {2, 2}, my_cnn::MAX);
    model.addConnectedLayer(10);
    model.setOutputLabels({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}, my_cnn::SOFTMAX);

    // Set the hyperparameters
    Model::Hyperparameters params;
    params.learning_rate = 0.03;
    params.num_epochs    = 2;
    params.batch_size    = 32;
    params.num_threads   = 8; 

    // Train the model
    model.train(data_source, params);

    // Validate on the MNIST test data set
    validate(model);

    // If applicable, save the resulting model to a file
    if (argc < 2){
        std::cout << "No file provided, model training results will not be saved\n";
    }else{
        std::string filename = data_dir + "/" + std::string(argv[1]) + ".model_data";
        std::cout << "Saving model to \"" << filename << "\"\n";
        model.saveModel(filename);
    }

    #ifdef TIMEIT
    global_timer.summary();
    #endif
    
    return 0;
}