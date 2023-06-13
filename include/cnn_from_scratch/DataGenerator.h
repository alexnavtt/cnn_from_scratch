#pragma once

#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

namespace my_cnn{
    
template<typename ModelInputType>
class DataGenerator {
    virtual SimpleMatrix<ModelInputType> getNextDataPoint() = 0;
};

} // namespace my_cnn
