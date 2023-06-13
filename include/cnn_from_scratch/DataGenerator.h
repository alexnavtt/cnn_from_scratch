#pragma once

#include <string>
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

namespace my_cnn{

template<typename ModelInputType>
struct LabeledInput{
    SimpleMatrix<ModelInputType> data;
    std::string label;
};

template<typename ModelInputType>
class DataGenerator {
    virtual LabeledInput<ModelInputType> getNextDataPoint() = 0;
};

} // namespace my_cnn
