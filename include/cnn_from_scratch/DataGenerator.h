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
public:
    virtual LabeledInput<ModelInputType> getNextDataPoint() = 0;
    virtual size_t size() = 0;
    virtual bool hasAvailableData() = 0;
};

} // namespace my_cnn
