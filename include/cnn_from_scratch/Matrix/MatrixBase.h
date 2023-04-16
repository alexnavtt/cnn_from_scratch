#pragma once

#include "cnn_from_scratch/Matrix/dim.h"

namespace my_cnn{
    
class MatrixBase{
public: 
    MatrixBase() = default;
    MatrixBase(const dim3& dim) : dim_(dim) {}

    size_t size() const{
        return dim_.x*dim_.y*dim_.z;
    }

    const dim3& dim() const{
        return dim_;
    }

protected:
    dim3 dim_;
};

} // namespace my_cnn
