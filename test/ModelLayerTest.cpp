#include <random>
#include <gtest/gtest.h>
#include "cnn_from_scratch/ModelDescription.h"

class FixtureBase : public testing::Test {
public:
    using type_t = typename decltype(my_cnn::Kernel::weights)::type;

    FixtureBase() :
    M(my_cnn::Dim3(5, 5, 1))
    {
        my_cnn::modify(M, [](type_t){return 1 - 2*(type_t)(rand())/RAND_MAX;});
    }

protected:
    my_cnn::SimpleMatrix<type_t> M;
};

// ============================================================================

using ActivationTest = FixtureBase;

TEST_F(ActivationTest, linear){
    my_cnn::Activation A(my_cnn::ModelActivationFunction::LINEAR);
    auto N = A.propagateForward(std::move(M));

    for (auto it = N.begin(); it != N.end(); ++it){
        EXPECT_EQ(M(it.idx()), N(it.idx()));
    }
}

TEST_F(ActivationTest, relu){
    my_cnn::Activation A(my_cnn::ModelActivationFunction::RELU);
    auto N = A.propagateForward(std::move(M));

    for (auto it = M.begin(); it != M.end(); ++it){
        auto val = *it;
        if (val > 0)
            EXPECT_EQ(N(it.idx()), val);
        else
            EXPECT_EQ(N(it.idx()), 0);
    }
}

TEST_F(ActivationTest, sigmoid){
    my_cnn::Activation A(my_cnn::ModelActivationFunction::SIGMOID);
    auto N = A.propagateForward(std::move(M));

    for (auto it = M.begin(); it != M.end(); ++it){
        auto val = *it;
        auto expected = 1.0/(1.0 + std::exp(-val));
        EXPECT_FLOAT_EQ(N(it.idx()), expected);
    }
}

TEST_F(ActivationTest, tangent){
    my_cnn::Activation A(my_cnn::ModelActivationFunction::TANGENT);
    auto N = A.propagateForward(std::move(M));

    for (auto it = M.begin(); it != M.end(); ++it){
        auto val = *it;
        auto expected = std::tanh(val);
        EXPECT_FLOAT_EQ(N(it.idx()), expected);
    }
}

TEST_F(ActivationTest, leakyRelu){
    my_cnn::Activation A(my_cnn::ModelActivationFunction::LEAKY_RELU);
    auto N = A.propagateForward(std::move(M));

    for (auto it = M.begin(); it != M.end(); ++it){
        auto val = *it;
        auto expected = val > 0 ? val : 0.1*val;
        EXPECT_FLOAT_EQ(N(it.idx()), expected);
    }
}

// ============================================================================

using ActivationGradient = FixtureBase;

TEST_F(ActivationGradient, linear){
    my_cnn::Activation A(my_cnn::ModelActivationFunction::LINEAR);
    auto N = A.propagateForward(std::move(M));
    auto grad = A.activationGradient(M);
    EXPECT_EQ(grad.dim(), M.dim());

    for (auto v : grad){
        EXPECT_EQ(v, 1);
    }
}

TEST_F(ActivationGradient, relu){
    my_cnn::Activation A(my_cnn::ModelActivationFunction::RELU);
    auto N = A.propagateForward(std::move(M));
    auto grad = A.activationGradient(M);
    EXPECT_EQ(grad.dim(), M.dim());

    for (auto it = M.begin(); it != M.end(); ++it){
        type_t expected = *it <= 0 ? 0 : 1;
        EXPECT_EQ(expected, grad(it.idx()));
    }
}

TEST_F(ActivationGradient, sigmoid){
    my_cnn::Activation A(my_cnn::ModelActivationFunction::SIGMOID);
    auto N = A.propagateForward(std::move(M));
    auto grad = A.activationGradient(M);
    EXPECT_EQ(grad.dim(), M.dim());

    for (auto it = M.begin(); it != M.end(); ++it){
        type_t sig = 1/(1 + std::exp(-*it));
        type_t expected = (sig)*(1 - sig);
        EXPECT_EQ(expected, grad(it.idx()));
    }
}

TEST_F(ActivationGradient, tangent){
    my_cnn::Activation A(my_cnn::ModelActivationFunction::TANGENT);
    auto N = A.propagateForward(std::move(M));
    auto grad = A.activationGradient(M);
    EXPECT_EQ(grad.dim(), M.dim());

    for (auto it = M.begin(); it != M.end(); ++it){
        type_t tanh = std::tanh(*it);
        type_t expected = 1 - tanh*tanh;
        EXPECT_EQ(expected, grad(it.idx()));
    }
}

TEST_F(ActivationGradient, leakyRelu){
    my_cnn::Activation A(my_cnn::ModelActivationFunction::LEAKY_RELU);
    auto N = A.propagateForward(std::move(M));
    auto grad = A.activationGradient(M);
    EXPECT_EQ(grad.dim(), M.dim());

    for (auto it = M.begin(); it != M.end(); ++it){
        type_t expected = *it <= 0 ? 0.1 : 1;
        EXPECT_EQ(expected, grad(it.idx()));
    }
}

// ============================================================================

int main(int argc, char* argv[]){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}