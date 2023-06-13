#include <random>
#include <gtest/gtest.h>
#include "cnn_from_scratch/ModelDescription.h"

TEST(Kernel, singleChannelSingleLayer){
    my_cnn::SimpleMatrix<float> input(my_cnn::dim3(4, 4, 1),
        {1, 2, 3, 4,
         5, 6, 7, 8,
         2, 4, 6, 8,
         1, 3, 5, 7}
    );

    my_cnn::ModelDescription<float, float> model;
    model.addKernel(my_cnn::dim3(3, 3, 1), 1);
    my_cnn::Kernel& kernel = *std::dynamic_pointer_cast<my_cnn::Kernel>(model.layers[0]);

    // Make sure their size is as expected
    EXPECT_EQ(kernel.weights.dim(), my_cnn::dim3(3, 3, 1));
    EXPECT_EQ(kernel.biases.dim() , my_cnn::dim3(1, 1, 1));

    // Set the values to known quantities
    kernel.weights.setEntries(
        {1.0f, -2.0f, 0.0f,
         4.0f,  2.0f, 1.0f,
         0.0f , 0.2f, 5.0f}
    );
    kernel.biases.setEntries({1});

    // Get the output and set the expected output
    my_cnn::SimpleMatrix<float> output = kernel.propagateForward(input);
    my_cnn::SimpleMatrix<float> unactivated_output(my_cnn::dim3(2, 2, 1),
        {66.8f + 1, 83.2f + 1,
         40.6f + 1, 64.0f + 1}
    );

    // Those two should be equal
    EXPECT_TRUE(my_cnn::matrixEqual(output, unactivated_output, 1e-5));
}

TEST(Kernel, singleChannelMultipleLayer){
    my_cnn::SimpleMatrix<float> input(my_cnn::dim3(4, 4, 1),
        {1, 2, 3, 4,
         5, 6, 7, 8,
         2, 4, 6, 8,
         1, 3, 5, 7}
    );

    my_cnn::ModelDescription<float, float> model;
    model.addKernel(my_cnn::dim3(3, 3, 1), 2);
    my_cnn::Kernel& kernel = *std::dynamic_pointer_cast<my_cnn::Kernel>(model.layers[0]);

    // Make sure their size is as expected
    EXPECT_EQ(kernel.weights.dim(), my_cnn::dim3(3, 3, 2));
    EXPECT_EQ(kernel.biases.dim() , my_cnn::dim3(2, 1, 1));

    // Set the values to known quantities
    kernel.weights.setEntries(
        {1.0f, -2.0f, 0.0f,
         4.0f,  2.0f, 1.0f,
         0.0f , 0.2f, 5.0f,
         
        -1.0f,  2.0f,  0.0f,
        -4.0f, -2.0f, -1.0f,
         0.0f, -0.2f, -5.0f}
    );
    kernel.biases.setEntries({1, -1});

    // Get the output and set the expected output
    my_cnn::SimpleMatrix<float> output = kernel.propagateForward(input);
    my_cnn::SimpleMatrix<float> unactivated_output(my_cnn::dim3(2, 2, 2),
        { 66.8f + 1,  83.2f + 1,
          40.6f + 1,  64.0f + 1,
         
         -66.8f - 1, -83.2f - 1,
         -40.6f - 1, -64.0f - 1}
    );

    // Those two should be equal
    EXPECT_TRUE(my_cnn::matrixEqual(output, unactivated_output, 1e-5));
}

TEST(Kernel, multipleChannelSingleLayer){
    my_cnn::SimpleMatrix<float> input(my_cnn::dim3(4, 4, 2),
        {1, 2, 3, 4,
         5, 6, 7, 8,
         2, 4, 6, 8,
         1, 3, 5, 7,
         
         5.0, 7.0, 2.0, 6.0,
        65.0, 7.0, 0.0, 0.2,
         0.2, 0.5, 0.3,-4.0,
        -6.2, 0.0, 2.3, 7.0}
    );

    my_cnn::ModelDescription<float, float> model;
    model.addKernel(my_cnn::dim3(3, 3, 2), 1);
    my_cnn::Kernel& kernel = *std::dynamic_pointer_cast<my_cnn::Kernel>(model.layers[0]);

    // Make sure their size is as expected
    EXPECT_EQ(kernel.weights.dim(), my_cnn::dim3(3, 3, 2));
    EXPECT_EQ(kernel.biases.dim() , my_cnn::dim3(1, 1, 1));

    // Set the values to known quantities
    kernel.weights.setEntries(
        {1.0f, -2.0f, 0.0f,
         4.0f,  2.0f, 1.0f,
         0.0f , 0.2f, 5.0f,
         
        -1.0f,  2.0f,  0.0f,
        -4.0f, -2.0f, -1.0f,
         0.0f, -0.2f, -5.0f}
    );
    kernel.biases.setEntries({1});

    // Get the output and set the expected output
    my_cnn::SimpleMatrix<float> output = kernel.propagateForward(input);
    my_cnn::SimpleMatrix<float> unactivated_output(my_cnn::dim3(2, 2, 1),
        { 66.8 - 266.6 + 1,  83.2 - 11.26 + 1,
          40.6 -  64.6 + 1,  64.0 - 41.06 + 1}
    );

    // Those two should be equal
    EXPECT_TRUE(my_cnn::matrixEqual(output, unactivated_output, 1e-5));
}

TEST(Pooling, singleChannelSingleStride){
    my_cnn::SimpleMatrix<float> input(my_cnn::dim3(4, 4, 1),
        {1, 2, 3, 4,
         5, 6, 7, 8,
         2, 4, 6, 8,
         1, 3, 5, 7}
    );

    my_cnn::ModelDescription<float, float> model;
    model.addPooling(my_cnn::dim2(2, 2), my_cnn::dim2(1, 1), my_cnn::PoolingType::MAX);
    my_cnn::Pooling& pool = *std::dynamic_pointer_cast<my_cnn::Pooling>(model.layers[0]);

    my_cnn::SimpleMatrix<float> result = pool.propagateForward(input);
    my_cnn::SimpleMatrix<float> expected(my_cnn::dim3(3, 3, 1),
        {6, 7, 8,
         6, 7, 8,
         4, 6, 8}
    );

    EXPECT_TRUE(my_cnn::matrixEqual(expected, result));
}

TEST(Pooling, multipleChannelSingleStride){
    my_cnn::SimpleMatrix<float> input(my_cnn::dim3(4, 4, 2),
        {1, 2, 3, 4,
         5, 6, 7, 8,
         2, 4, 6, 8,
         1, 3, 5, 7,
         
         5.0, 7.0, 2.0, 6.0,
        65.0, 7.0, 0.0, 0.2,
         0.2, 0.5, 0.3,-4.0,
        -6.2, 0.0, 2.3, 7.0}
    );

    my_cnn::ModelDescription<float, float> model;
    model.addPooling(my_cnn::dim2(2, 2), my_cnn::dim2(1, 1), my_cnn::PoolingType::MIN);
    my_cnn::Pooling& pool = *std::dynamic_pointer_cast<my_cnn::Pooling>(model.layers[0]);

    my_cnn::SimpleMatrix<float> result = pool.propagateForward(input);
    my_cnn::SimpleMatrix<float> expected(my_cnn::dim3(3, 3, 2),
        {1, 2, 3,
         2, 4, 6,
         1, 3, 5,
         
         5, 0, 0, 
       0.2, 0,-4,
      -6.2, 0,-4}
    );

    EXPECT_TRUE(my_cnn::matrixEqual(expected, result));
}

TEST(Pooling, singleChannelLargeStride){
    my_cnn::SimpleMatrix<float> input(my_cnn::dim3(4, 4, 1),
        {1, 2, 3, 4,
         5, 6, 7, 8,
         2, 4, 6, 8,
         1, 3, 5, 7}
    );

    my_cnn::ModelDescription<float, float> model;
    model.addPooling(my_cnn::dim2(2, 2), my_cnn::dim2(2, 2), my_cnn::PoolingType::AVG);
    my_cnn::Pooling& pool = *std::dynamic_pointer_cast<my_cnn::Pooling>(model.layers[0]);

    my_cnn::SimpleMatrix<float> result = pool.propagateForward(input);
    my_cnn::SimpleMatrix<float> expected(my_cnn::dim3(2, 2, 1),
        {3.5, 5.5,
         2.5, 6.5}
    );

    EXPECT_TRUE(my_cnn::matrixEqual(expected, result));
}

int main(int argc, char* argv[]){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}