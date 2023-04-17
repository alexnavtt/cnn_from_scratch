#pragma once

#include <type_traits>
#include <functional>
#include "cnn_from_scratch/Matrix/MatrixBase.h"

namespace my_cnn{
    
template<typename T>
class SimpleMatrix;

// ================================================================================================
// Self modification math
// ================================================================================================

template<typename MatrixType, typename Other, std::enable_if_t<std::is_convertible_v<Other, typename std::remove_reference_t<MatrixType>::type>, bool> = true>
MatrixType& operator+=(MatrixType&& M, const Other& other){
    std::for_each(std::begin(M), std::end(M), [&](auto& val){val += other;});
    return M;
}

template<typename MatrixType, typename Other, std::enable_if_t<std::is_base_of_v<MatrixBase, Other>, bool> = true>
MatrixType& operator+=(MatrixType&& M, const Other& other){
    checkSize(M, other);
    for (DimIterator<3> idx(M.dim(), {0, 0, 0}); idx.idx.z < M.dim().z; idx++){
        M(idx.idx) += other(idx.idx);
    }
    return M;
}

template<typename MatrixType, typename Other, std::enable_if_t<std::is_convertible_v<Other, typename std::remove_reference_t<MatrixType>::type>, bool> = true>
MatrixType& operator-=(MatrixType&& M, const Other& other){
    std::for_each(std::begin(M), std::end(M), [&](auto& val){val -= other;});
    return M;
}

template<typename MatrixType, typename Other, std::enable_if_t<std::is_base_of_v<MatrixBase, Other>, bool> = true>
MatrixType& operator-=(MatrixType&& M, const Other& other){
    checkSize(M, other);
    for (DimIterator<3> idx(M.dim(), {0, 0, 0}); idx.idx.z < M.dim().z; idx++){
        M(idx.idx) -= other(idx.idx);
    }
    return M;
}

template<typename MatrixType, typename Other, std::enable_if_t<std::is_convertible_v<Other, typename std::remove_reference_t<MatrixType>::type>, bool> = true>
MatrixType& operator*=(MatrixType&& M, const Other& other){
    std::for_each(std::begin(M), std::end(M), [&](auto& val){val *= other;});
    return M;
}

template<typename MatrixType, typename Other, std::enable_if_t<std::is_base_of_v<MatrixBase, Other>, bool> = true>
MatrixType& operator*=(MatrixType&& M, const Other& other){
    checkSize(M, other);
    for (DimIterator<3> idx(M.dim(), {0, 0, 0}); idx.idx.z < M.dim().z; idx++){
        M(idx.idx) *= other(idx.idx);
    }
    return M;
}

template<typename MatrixType, typename Other, std::enable_if_t<std::is_convertible_v<Other, typename std::remove_reference_t<MatrixType>::type>, bool> = true>
MatrixType& operator/=(MatrixType&& M, const Other& other){
    std::for_each(std::begin(M), std::end(M), [&](auto& val){val /= other;});
    return M;
}

template<typename MatrixType, typename Other, std::enable_if_t<std::is_base_of_v<MatrixBase, Other>, bool> = true>
MatrixType& operator/=(MatrixType&& M, const Other& other){
    checkSize(M, other);
    for (DimIterator<3> idx(M.dim(), {0, 0, 0}); idx.idx.z < M.dim().z; idx++){
        M(idx.idx) /= other(idx.idx);
    }
    return M;
}

// ================================================================================================
// Matrix - Matrix math result
// ================================================================================================

template<typename MatrixType1, typename MatrixType2, class BinaryOp, 
    typename = std::enable_if_t<std::is_base_of_v<MatrixBase, MatrixType1> && std::is_base_of_v<MatrixBase, MatrixType2>>>
class MatrixOperationResult : public MatrixBase{
public:
    using type = typename std::common_type_t<typename MatrixType1::type, typename MatrixType2::type>;

    MatrixOperationResult(const MatrixType1& M1, const MatrixType2& M2, BinaryOp Op) :
    MatrixBase(M1.dim()),
    m1_(&M1), m2_(&M2), op(Op) 
    {
        checkSize(M1, M2);
    }

    type operator()(const dim3& idx) const{
        return std::invoke(op, *m1_, *m2_, idx);
    }

    auto begin() {return MatrixIterator<MatrixOperationResult<MatrixType1, MatrixType2, BinaryOp>>(this, {0, 0, 0});}
    auto end() {return MatrixIterator<MatrixOperationResult<MatrixType1, MatrixType2, BinaryOp>>(this, {0, 0, dim_.z});}
    auto begin() const {return MatrixIterator<const MatrixOperationResult<MatrixType1, MatrixType2, BinaryOp>>(this, {0, 0, 0});}
    auto end() const {return MatrixIterator<const MatrixOperationResult<MatrixType1, MatrixType2, BinaryOp>>(this, {0, 0, dim_.z});}

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

template<typename MatrixType1, typename MatrixType2,
    std::enable_if_t<std::is_base_of_v<MatrixBase, MatrixType1> && std::is_base_of_v<MatrixBase, MatrixType2>, bool> = true>
bool operator==(const MatrixType1& M1, const MatrixType2& M2){
    if (M1.dim() != M2.dim()) return false;
    bool equal = true;
    for (auto it = std::begin(M1); it != std::end(M1) && equal; it++){
        equal = equal && (*it == M2(it.idx()));
    }
    return equal;
}

// ================================================================================================
// Matrix - Scalar math result
// ================================================================================================

template<typename MatrixType, typename ScalarType, class BinaryOp>
class ScalarOperationResult : public MatrixBase{
public:
    using type = typename std::common_type_t<typename MatrixType::type, ScalarType>;

    ScalarOperationResult(const MatrixType& M, ScalarType S, BinaryOp Op) :
    MatrixBase(M.dim()),
    m_(&M), s_(S), op(Op) 
    {}

    type operator()(dim3 idx) const{
        return std::invoke(op, *m_, s_, idx);
    }

    auto begin() {return MatrixIterator<ScalarOperationResult<MatrixType, ScalarType, BinaryOp>>(this, {0, 0, 0});}
    auto end() {return MatrixIterator<ScalarOperationResult<MatrixType, ScalarType, BinaryOp>>(this, {0, 0, dim_.z});}
    auto begin() const {return MatrixIterator<const ScalarOperationResult<MatrixType, ScalarType, BinaryOp>>(this, {0, 0, 0});}
    auto end() const {return MatrixIterator<const ScalarOperationResult<MatrixType, ScalarType, BinaryOp>>(this, {0, 0, dim_.z});}

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
// Matrix unary operation
// ================================================================================================

template<typename MatrixType, class UnaryOp>
class UnaryOperationResult : public MatrixBase{
public:
    using type = typename std::remove_const_t<typename MatrixType::type>;

    UnaryOperationResult(const MatrixType& M, UnaryOp Op) :
    MatrixBase(M.dim()),
    m_(&M), op(Op) 
    {}

    type operator()(dim3 idx) const{
        return std::invoke(op, *m_, idx);
    }

    auto begin() {return MatrixIterator<UnaryOperationResult<MatrixType, UnaryOp>>(this, {0, 0, 0});}
    auto end() {return MatrixIterator<UnaryOperationResult<MatrixType, UnaryOp>>(this, {0, 0, dim_.z});}
    auto begin() const {return MatrixIterator<const UnaryOperationResult<MatrixType, UnaryOp>>(this, {0, 0, 0});}
    auto end() const {return MatrixIterator<const UnaryOperationResult<MatrixType, UnaryOp>>(this, {0, 0, dim_.z});}

private:
    const MatrixType* m_;
    UnaryOp op;
};

template<typename MatrixType, typename UnaryOp>
auto unaryOperation(const MatrixType& M, const dim3& idx, UnaryOp op){
    return std::invoke(op, M(idx));
}

template<typename MatrixType, typename UnaryOp>
auto apply(const MatrixType& M, UnaryOp op){
    return UnaryOperationResult(M, [op](const MatrixType& M_, const dim3& idx_){return std::invoke(op, M_(idx_));});
}

template<typename MatrixType>
auto abs(const MatrixType& M){
    using input_type = typename MatrixType::type;
    using return_type = decltype(std::abs(input_type{}));
    return apply<MatrixType, return_type(*)(input_type)>(M, std::abs);
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
