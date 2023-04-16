#pragma once

#include <type_traits>
#include <functional>
#include "cnn_from_scratch/Matrix/dim.h"

namespace my_cnn{
    
template<typename T>
class SimpleMatrix;

// ================================================================================================
// Matrix - Matrix math result
// ================================================================================================

template<typename MatrixType1, typename MatrixType2, class BinaryOp>
class MatrixOperationResult{
public:
    using type = typename std::common_type_t<typename MatrixType1::type, typename MatrixType2::type>;

    MatrixOperationResult(const MatrixType1& M1, const MatrixType2& M2, BinaryOp Op) :
    m1_(&M1), m2_(&M2), op(Op) 
    {
        checkSize(M1, M2);
    }

    type operator()(const dim3& idx) const{
        return std::invoke(op, *m1_, *m2_, idx);
    }

    dim3 dim() const{
        return m1_->dim();
    }

private:
    const MatrixType1* m1_;
    const MatrixType2* m2_;
    BinaryOp op;
};

template<typename MatrixType1, typename MatrixType2>
auto elementWiseAdd(const MatrixType1& M1, const MatrixType2& M2, const dim3& idx){
    return M1(idx) + M2(idx);
}

template<typename MatrixType1, typename MatrixType2>
auto elementWiseMultiply(const MatrixType1& M1, const MatrixType2& M2, const dim3& idx){
    return M1(idx) * M2(idx);
}

template<typename MatrixType1, typename MatrixType2>
auto elementWiseDivide(const MatrixType1& M1, const MatrixType2& M2, const dim3& idx){
    return M1(idx) / M2(idx);
}

template<typename MatrixType1, typename MatrixType2>
auto elementWiseSubtract(const MatrixType1& M1, const MatrixType2& M2, const dim3& idx){
    return M1(idx) - M2(idx);
}

template<typename MatrixType1, typename MatrixType2,
    std::enable_if_t<std::is_convertible_v<MatrixType2, SimpleMatrix<typename MatrixType2::type>>, bool> = true>
auto operator+(const MatrixType1& M1, const MatrixType2& M2){
    return MatrixOperationResult(M1, M2, elementWiseAdd<MatrixType1, MatrixType2>);
}

template<typename MatrixType1, typename MatrixType2,
    std::enable_if_t<std::is_convertible_v<MatrixType2, SimpleMatrix<typename MatrixType2::type>>, bool> = true>
auto operator-(const MatrixType1& M1, const MatrixType2& M2){
    return MatrixOperationResult(M1, M2, elementWiseSubtract<MatrixType1, MatrixType2>);
}

template<typename MatrixType1, typename MatrixType2,
    std::enable_if_t<std::is_convertible_v<MatrixType2, SimpleMatrix<typename MatrixType2::type>>, bool> = true>
auto operator*(const MatrixType1& M1, const MatrixType2& M2){
    return MatrixOperationResult(M1, M2, elementWiseMultiply<MatrixType1, MatrixType2>);
}

template<typename MatrixType1, typename MatrixType2,
    std::enable_if_t<std::is_convertible_v<MatrixType2, SimpleMatrix<typename MatrixType2::type>>, bool> = true>
auto operator/(const MatrixType1& M1, const MatrixType2& M2){
    return MatrixOperationResult(M1, M2, elementWiseDivide<MatrixType1, MatrixType2>);
}


// ================================================================================================
// Matrix - Scalar math result
// ================================================================================================

template<typename MatrixType, typename ScalarType, class BinaryOp,
    typename = std::enable_if_t<std::is_arithmetic_v<ScalarType>>>
class ScalarOperationResult{
public:
    using type = typename std::common_type_t<typename MatrixType::type, ScalarType>;

    ScalarOperationResult(const MatrixType& M, ScalarType S, BinaryOp Op) :
    m_(&M), s_(S), op(Op) 
    {}

    type operator()(dim3 idx) const{
        return std::invoke(op, *m_, s_, idx);
    }

    dim3 dim() const{
        return m_->dim();
    }

private:
    const MatrixType* m_;
    const ScalarType s_;
    BinaryOp op;
};

template<typename MatrixType1, typename Scalar>
auto scalarAdd(const MatrixType1& M1, const Scalar& S, const dim3& idx){
    return M1(idx) + S;
}

template<typename MatrixType1, typename Scalar>
auto scalarMultiply(const MatrixType1& M1, const Scalar& S, const dim3& idx){
    return M1(idx) * S;
}

template<typename MatrixType1, typename Scalar>
auto scalarDivide(const MatrixType1& M1, const Scalar& S, const dim3& idx){
    return M1(idx) / S;
}

template<typename MatrixType1, typename Scalar>
auto scalarSubtract(const MatrixType1& M1, const Scalar& S, const dim3& idx){
    return M1(idx) - S;
}

template<typename MatrixType1, typename Scalar,
    std::enable_if_t<std::is_arithmetic_v<Scalar>, bool> = true>
auto operator+(const MatrixType1& M1, const Scalar& S){
    return ScalarOperationResult(M1, S, scalarAdd<MatrixType1, Scalar>);
}

template<typename MatrixType1, typename Scalar,
    std::enable_if_t<std::is_arithmetic_v<Scalar>, bool> = true>
auto operator-(const MatrixType1& M1, const Scalar& S){
    return ScalarOperationResult(M1, S, scalarSubtract<MatrixType1, Scalar>);
}

template<typename MatrixType1, typename Scalar,
    std::enable_if_t<std::is_arithmetic_v<Scalar>, bool> = true>
auto operator*(const MatrixType1& M1, const Scalar& S){
    return ScalarOperationResult(M1, S, scalarMultiply<MatrixType1, Scalar>);
}

template<typename MatrixType1, typename Scalar,
    std::enable_if_t<std::is_arithmetic_v<Scalar>, bool> = true>
auto operator/(const MatrixType1& M1, const Scalar& S){
    return ScalarOperationResult(M1, S, scalarDivide<MatrixType1, Scalar>);
}


// ================================================================================================
// 
// ================================================================================================

template<typename Matrix1, typename Matrix2>
void checkSize(const Matrix1& M1, const Matrix2& M2){
    if (M1.dim() != M2.dim()){
        std::stringstream ss;
        ss << "Mismatched matrix sizes between matrices M1 " << 
        M1.dim() << " and M2 " << M2.dim();
        throw MatrixSizeException(ss.str());
    }
}

} // namespace my_cnn
