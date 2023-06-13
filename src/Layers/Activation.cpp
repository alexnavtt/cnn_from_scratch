#include "cnn_from_scratch/Layers/Activation.h"

namespace my_cnn{

/* ---  Activation Function Definitions --- */
double reluFcn(double val)          {return val > 0 ? val : 0;}
double sigmoidFcn(double val)       {return 1.0/(1.0 + std::exp(-val));};
double linearFcn(double val)        {return val;}
double tangentFcn(double val)       {return std::tanh(val);}
double leakyRelyFcn(double val)     {return std::max(0.1f*val, val);}

/* ---  Activation Gradient Definitions --- */
double reluGradFcn(double val)      {return val > 0;}
double sigmoidGradFcn(double val)   {return sigmoidFcn(val)*(1 - sigmoidFcn(val));}
double linearGradFcn(double)        {return 1.0;}
double tangentGradFcn(double val)   {return 1.0 - tangentFcn(val)*tangentFcn(val);}
double leakyRelyGradFcn(double val) {return val > 0 ? 1.0 : 0.1;}

// Array of activation functions for easy access via enum
static constexpr std::array<double(*)(double), 5> activation_functions(){
    std::array<double(*)(double), 5> functions{};
    functions[RELU]       = reluFcn;
    functions[SIGMOID]    = sigmoidFcn;
    functions[LINEAR]     = linearFcn;
    functions[TANGENT]    = tangentFcn;
    functions[LEAKY_RELU] = leakyRelyFcn;
    return functions;
};

// Array of activation gradient functions for easy access via enum
static constexpr std::array<double(*)(double), 5> activation_gradients(){
    std::array<double(*)(double), 5> functions{};
    functions[RELU]       = reluGradFcn;
    functions[SIGMOID]    = sigmoidGradFcn;
    functions[LINEAR]     = linearGradFcn;
    functions[TANGENT]    = tangentGradFcn;
    functions[LEAKY_RELU] = leakyRelyGradFcn;
    return functions;
};

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

Activation::Activation(ModelActivationFunction activation) : 
activation_(activation),
activationFunction_{activation_functions()[activation_]},
activationGradientFunction_{activation_gradients()[activation_]}
{}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

ModelFlowMode Activation::getType() const {
    return ACTIVATION;
};

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

SimpleMatrix<double> Activation::propagateForward(SimpleMatrix<double>&& input) {
    return apply(input, activationFunction_);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

SimpleMatrix<double> Activation::activationGradient(const SimpleMatrix<double>& activated_output) const {
    return apply(activated_output, activationGradientFunction_);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

SimpleMatrix<double> Activation::propagateBackward(
    const SimpleMatrix<double>&, const SimpleMatrix<double>& Z, 
    const SimpleMatrix<double>& dLdZ, double, bool)
{
    return dLdZ * activationGradient(Z);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

std::string Activation::serialize() const {
    std::stringstream ss;
    ss << "Activation\n";
    ss << toString(activation_) << "\n";
    return ss.str();
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

bool Activation::deserialize(std::istream& is) {
    serialization::expect<void>(is, "Activation");

    std::string activation_string;
    std::getline(is, activation_string);
    activation_ = fromString(activation_string);
    return true;
}

} // namespace my_cnn
