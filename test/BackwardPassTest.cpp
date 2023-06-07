#include <random>
#include <gtest/gtest.h>
#include "cnn_from_scratch/ModelDescription.h"

cpp_timer::Timer global_timer;

class SingleKernelFixture : public testing::Test {
public:
    SingleKernelFixture() :
    layer_input(my_cnn::dim3(3, 3, 1),
        {1.0, 2.0, 3.0,
         4.0,-5.0, 6.0,
         7.0, 8.0, 0.9}),
    layer_loss_gradient(my_cnn::dim3(2, 2, 1), 
        {-0.1, 0.3,
          0.4, 0.6}),
    layer_output(my_cnn::dim3(2, 2, 1), 
        { 0.0, 77.2,
         35.4,  0.6})
    {
        my_cnn::Kernel& K = model.addKernel(my_cnn::dim3(2, 2, 1), 1, my_cnn::RELU);
        K.weights.setEntries(
            {0.1, 3.0,
            -3.0, 9.0}
        );
        K.biases.setEntries({1.0});
    }

    my_cnn::SimpleMatrix<double> layer_input;
    my_cnn::SimpleMatrix<double> layer_loss_gradient;
    my_cnn::SimpleMatrix<double> layer_output;
    my_cnn::ModelDescription<double, double> model;
};

TEST_F(SingleKernelFixture, activationGradient){

    // Get the kernel
    my_cnn::Kernel& kernel = *std::dynamic_pointer_cast<my_cnn::Kernel>(model.layers.back());    

    // Get the output loss gradient from the layer loss gradient
    my_cnn::SimpleMatrix<double> dLdY = kernel.getdLdY(layer_output, layer_loss_gradient);

    // Expected output gradient
    my_cnn::SimpleMatrix<double> expected_output_gradient(my_cnn::dim3(2, 2, 1),
        { 0.0, 0.3, 
          0.4, 0.6}
    );

    ASSERT_TRUE(my_cnn::matrixEqual(expected_output_gradient, dLdY, 1e-5));
}

TEST_F(SingleKernelFixture, weightGradient){
    // Get the kernel
    my_cnn::Kernel& kernel = *std::dynamic_pointer_cast<my_cnn::Kernel>(model.layers.back());

    // Get the output loss gradient from the layer loss gradient
    my_cnn::SimpleMatrix<double> dLdY = kernel.getdLdY(layer_output, layer_loss_gradient);

    // Get the weight gradient
    my_cnn::SimpleMatrix<double> dLdW = kernel.getdLdW(layer_input, dLdY);

    my_cnn::SimpleMatrix<double> expected_weight_gradient(my_cnn::dim3(2, 2, 1));
    expected_weight_gradient(0, 0, 0) = dLdY(0, 0, 0) * layer_input(0, 0, 0)
                                      + dLdY(0, 1, 0) * layer_input(0, 1, 0)
                                      + dLdY(1, 0, 0) * layer_input(1, 0, 0)
                                      + dLdY(1, 1, 0) * layer_input(1, 1, 0);

    expected_weight_gradient(0, 1, 0) = dLdY(0, 0, 0) * layer_input(0, 1, 0)
                                      + dLdY(0, 1, 0) * layer_input(0, 2, 0)
                                      + dLdY(1, 0, 0) * layer_input(1, 1, 0)
                                      + dLdY(1, 1, 0) * layer_input(1, 2, 0);

    expected_weight_gradient(1, 0, 0) = dLdY(0, 0, 0) * layer_input(1, 0, 0)
                                      + dLdY(0, 1, 0) * layer_input(1, 1, 0)
                                      + dLdY(1, 0, 0) * layer_input(2, 0, 0)
                                      + dLdY(1, 1, 0) * layer_input(2, 1, 0);

    expected_weight_gradient(1, 1, 0) = dLdY(0, 0, 0) * layer_input(1, 1, 0)
                                      + dLdY(0, 1, 0) * layer_input(1, 2, 0)
                                      + dLdY(1, 0, 0) * layer_input(2, 1, 0)
                                      + dLdY(1, 1, 0) * layer_input(2, 2, 0);

    ASSERT_TRUE(my_cnn::matrixEqual(dLdW, expected_weight_gradient, 1e-5));
}

TEST_F(SingleKernelFixture, inputGradient){
    // Get the kernel
    my_cnn::Kernel& kernel = *std::dynamic_pointer_cast<my_cnn::Kernel>(model.layers.back());
    my_cnn::SimpleMatrix<double>& F = kernel.weights;

    // Get the output loss gradient from the layer loss gradient
    my_cnn::SimpleMatrix<double> dLdY = kernel.getdLdY(layer_output, layer_loss_gradient);

    // Get the weight gradient
    my_cnn::SimpleMatrix<double> dLdX = kernel.getdLdX(layer_input, dLdY);

    my_cnn::SimpleMatrix<double> expected_input_gradient(my_cnn::dim3(3, 3, 1));
    expected_input_gradient(0, 0, 0) = dLdY(0, 0, 0) * F(0, 0, 0);
    expected_input_gradient(0, 1, 0) = dLdY(0, 0, 0) * F(0, 1, 0) + dLdY(0, 1, 0) * F(0, 0, 0);
    expected_input_gradient(0, 2, 0) = dLdY(0, 1, 0) * F(0, 1, 0);
    expected_input_gradient(1, 0, 0) = dLdY(0, 0, 0) * F(1, 0, 0) + dLdY(1, 0, 0) * F(0, 0, 0);
    expected_input_gradient(1, 1, 0) = dLdY(0, 0, 0) * F(1, 1, 0) + dLdY(0, 1, 0) * F(1, 0, 0)
                                     + dLdY(1, 0, 0) * F(0, 1, 0) + dLdY(1, 1, 0) * F(0, 0, 0);
    expected_input_gradient(1, 2, 0) = dLdY(0, 1, 0) * F(1, 1, 0) + dLdY(1, 1, 0) * F(0, 1, 0);
    expected_input_gradient(2, 0, 0) = dLdY(1, 0, 0) * F(1, 0, 0);
    expected_input_gradient(2, 1, 0) = dLdY(1, 0, 0) * F(1, 1, 0) + dLdY(1, 1, 0) * F(1, 0, 0);
    expected_input_gradient(2, 2, 0) = dLdY(1, 1, 0) * F(1, 1, 0);

    std::cout << dLdX;
    std::cout << expected_input_gradient;
    
    ASSERT_TRUE(my_cnn::matrixEqual(dLdX, expected_input_gradient, 1e-5));
}

TEST_F(SingleKernelFixture, biasGradient){
    // Get the kernel
    my_cnn::Kernel& kernel = *std::dynamic_pointer_cast<my_cnn::Kernel>(model.layers.back());
    const my_cnn::SimpleMatrix<double>& B = kernel.biases;
}

int main(int argc, char* argv[]){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}