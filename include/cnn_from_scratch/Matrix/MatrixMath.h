#pragma once

#include <type_traits>
#include <functional>
#include "cnn_from_scratch/Matrix/dim.h"

namespace my_cnn{
    
template<typename T>
class SimpleMatrix;

template<typename MatrixType1, typename MatrixType2, class BinaryOp>
class MatrixOperationResult{
public:
    using type = typename std::common_type_t<typename MatrixType1::type, typename MatrixType2::type>;

    MatrixOperationResult(const MatrixType1& M1, const MatrixType2& M2, BinaryOp Op) :
    m1_(&M1), m2_(&M2), op(Op) 
    {
        checkSize(M1, M2);
    }

    type operator()(dim3 idx) const{
        return std::invoke(op, m1_->operator()(idx), m2_->operator()(idx));
    }

    dim3 dim() const{
        return m1_->dim();
    }

    operator SimpleMatrix<type>() const; 

private:
    const MatrixType1* m1_;
    const MatrixType2* m2_;
    BinaryOp op;
};

template<typename Matrix1, typename Matrix2>
void checkSize(const Matrix1& M1, const Matrix2& M2){
    if (M1.dim() != M2.dim()){
        std::stringstream ss;
        ss << "Mismatched matrix sizes between matrices M1 " << 
        M1.dim() << " and M2 " << M2.dim();
        throw MatrixSizeException(ss.str());
    }
}

template<typename T1, typename T2>
typename std::common_type_t<T1, T2> binaryAdd(const T1& a, const T2& b){
    return a + b;
}

template<typename T1, typename T2>
typename std::common_type_t<T1, T2> binaryMultiply(const T1& a, const T2& b){
    return a * b;
}

template<typename T1, typename T2>
typename std::common_type_t<T1, T2> binaryDivide(const T1& a, const T2& b){
    return a / b;
}

template<typename T1, typename T2>
typename std::common_type_t<T1, T2> binarySubtract(const T1& a, const T2& b){
    return a - b;
}

template<typename MatrixType1, typename MatrixType2>
auto matrixAdd(const MatrixType1& M1, const MatrixType2& M2){
    using T1 = typename MatrixType1::type;
    using T2 = typename MatrixType2::type;
    return MatrixOperationResult(M1, M2, binaryAdd<T1, T2>);
}

template<typename MatrixType1, typename MatrixType2>
auto operator+(const MatrixType1& M1, const MatrixType2& M2){
    using T1 = typename MatrixType1::type;
    using T2 = typename MatrixType2::type;
    return MatrixOperationResult(M1, M2, binaryAdd<T1, T2>);
}

template<typename MatrixType1, typename MatrixType2>
auto operator-(const MatrixType1& M1, const MatrixType2& M2){
    using T1 = typename MatrixType1::type;
    using T2 = typename MatrixType2::type;
    return MatrixOperationResult(M1, M2, binarySubtract<T1, T2>);
}

template<typename MatrixType1, typename MatrixType2>
auto operator*(const MatrixType1& M1, const MatrixType2& M2){
    using T1 = typename MatrixType1::type;
    using T2 = typename MatrixType2::type;
    return MatrixOperationResult(M1, M2, binaryMultiply<T1, T2>);
}

template<typename MatrixType1, typename MatrixType2>
auto operator/(const MatrixType1& M1, const MatrixType2& M2){
    using T1 = typename MatrixType1::type;
    using T2 = typename MatrixType2::type;
    return MatrixOperationResult(M1, M2, binaryDivide<T1, T2>);
}

} // namespace my_cnn
