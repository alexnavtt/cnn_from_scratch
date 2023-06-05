#pragma once

#include "cnn_from_scratch/Matrix/dim.h"

namespace my_cnn{
    
class MatrixBase{
public: 
    MatrixBase() = default;
    MatrixBase(const dim3& dim) : dim_(dim) {}

    bool isSquare() const{
        return dim_.x == dim_.y;
    }

    bool isFlat() const{
        return dim_.z == 1;
    }

    bool isRow() const{
        return isFlat() && dim_.x == 1;
    }

    bool isColumn() const{
        return isFlat() && dim_.y == 1;
    }

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
