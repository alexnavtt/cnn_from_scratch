#include <random>
#include <gtest/gtest.h>
#include "cnn_from_scratch/ModelDescription.h"

cpp_timer::Timer global_timer;

class FixtureBase : public testing::Test, public my_cnn::ModelLayer {
public:
    using type_t = typename decltype(my_cnn::Kernel::weights)::type;

    FixtureBase() :
    M(my_cnn::dim3(5, 5, 1))
    {
        my_cnn::modify(M, [](type_t){return 1 - 2*(type_t)(rand())/RAND_MAX;});
        M_copy = M;
    }

    bool checkSize(const my_cnn::SimpleMatrix<double>& input){return {};}
    my_cnn::SimpleMatrix<double> propagateForward(my_cnn::SimpleMatrix<double>&& input){return {};}
    my_cnn::SimpleMatrix<double> propagateBackward(
        const my_cnn::SimpleMatrix<double>& input, 
        const my_cnn::SimpleMatrix<double>& output, 
        const my_cnn::SimpleMatrix<double>& output_grad, 
        double learning_rate, 
        bool
    ) override {return {};}

protected:
    my_cnn::SimpleMatrix<type_t> M;
    my_cnn::SimpleMatrix<type_t> M_copy;
};

// ============================================================================

using Activation = FixtureBase;

TEST_F(Activation, linear){
    activation = my_cnn::ModelActivationFunction::LINEAR;
    activate(M);

    for (auto it = M_copy.begin(); it != M_copy.end(); ++it){
        EXPECT_EQ(M(it.idx()), M_copy(it.idx()));
    }
}

TEST_F(Activation, relu){
    activation = my_cnn::ModelActivationFunction::RELU;
    activate(M);

    for (auto it = M_copy.begin(); it != M_copy.end(); ++it){
        auto val = *it;
        if (val > 0)
            EXPECT_EQ(M(it.idx()), val);
        else
            EXPECT_EQ(M(it.idx()), 0);
    }
}

TEST_F(Activation, sigmoid){
    activation = my_cnn::ModelActivationFunction::SIGMOID;
    activate(M);

    for (auto it = M_copy.begin(); it != M_copy.end(); ++it){
        auto val = *it;
        auto expected = 1.0/(1.0 + std::exp(-val));
        EXPECT_FLOAT_EQ(M(it.idx()), expected);
    }
}

TEST_F(Activation, tangent){
    activation = my_cnn::ModelActivationFunction::TANGENT;
    activate(M);

    for (auto it = M_copy.begin(); it != M_copy.end(); ++it){
        auto val = *it;
        auto expected = std::tanh(val);
        EXPECT_FLOAT_EQ(M(it.idx()), expected);
    }
}

TEST_F(Activation, leakyRelu){
    activation = my_cnn::ModelActivationFunction::LEAKY_RELU;
    activate(M);

    for (auto it = M_copy.begin(); it != M_copy.end(); ++it){
        auto val = *it;
        auto expected = val > 0 ? val : 0.1*val;
        EXPECT_FLOAT_EQ(M(it.idx()), expected);
    }
}

// ============================================================================

using ActivationGradient = FixtureBase;

TEST_F(ActivationGradient, linear){
    activation = my_cnn::ModelActivationFunction::LINEAR;
    activate(M);
    auto grad = activationGradient(M);
    EXPECT_EQ(grad.dim(), M.dim());

    for (auto v : grad){
        EXPECT_EQ(v, 1);
    }
}

TEST_F(ActivationGradient, relu){
    activation = my_cnn::ModelActivationFunction::RELU;
    activate(M);
    auto grad = activationGradient(M);
    EXPECT_EQ(grad.dim(), M.dim());

    for (auto it = M.begin(); it != M.end(); ++it){
        type_t expected = *it <= 0 ? 0 : 1;
        EXPECT_EQ(expected, grad(it.idx()));
    }
}

TEST_F(ActivationGradient, sigmoid){
    activation = my_cnn::ModelActivationFunction::SIGMOID;
    activate(M);
    auto grad = activationGradient(M);
    EXPECT_EQ(grad.dim(), M.dim());

    for (auto it = M.begin(); it != M.end(); ++it){
        type_t sig = 1/(1 + std::exp(-*it));
        type_t expected = (sig)*(1 - sig);
        EXPECT_EQ(expected, grad(it.idx()));
    }
}

TEST_F(ActivationGradient, tangent){
    activation = my_cnn::ModelActivationFunction::TANGENT;
    activate(M);
    auto grad = activationGradient(M);
    EXPECT_EQ(grad.dim(), M.dim());

    for (auto it = M.begin(); it != M.end(); ++it){
        type_t tanh = std::tanh(*it);
        type_t expected = 1 - tanh*tanh;
        EXPECT_EQ(expected, grad(it.idx()));
    }
}

TEST_F(ActivationGradient, leakyRelu){
    activation = my_cnn::ModelActivationFunction::LEAKY_RELU;
    activate(M);
    auto grad = activationGradient(M);
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