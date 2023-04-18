#pragma once

#include <numeric>
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

    type operator()(uint x, uint y, uint z) const{
        return operator()(dim3(x,y,z));
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
    std::enable_if_t<std::is_base_of_v<MatrixBase, MatrixType1> && std::is_base_of_v<MatrixBase, MatrixType2>, bool> = true>
auto operator+(const MatrixType1& M1, const MatrixType2& M2){
    return MatrixOperationResult(M1, M2, elementWiseAdd<MatrixType1, MatrixType2>);
}

template<typename MatrixType1, typename MatrixType2,
    std::enable_if_t<std::is_base_of_v<MatrixBase, MatrixType1> && std::is_base_of_v<MatrixBase, MatrixType2>, bool> = true>
auto operator-(const MatrixType1& M1, const MatrixType2& M2){
    return MatrixOperationResult(M1, M2, elementWiseSubtract<MatrixType1, MatrixType2>);
}

template<typename MatrixType1, typename MatrixType2,
    std::enable_if_t<std::is_base_of_v<MatrixBase, MatrixType1> && std::is_base_of_v<MatrixBase, MatrixType2>, bool> = true>
auto operator*(const MatrixType1& M1, const MatrixType2& M2){
    return MatrixOperationResult(M1, M2, elementWiseMultiply<MatrixType1, MatrixType2>);
}

template<typename MatrixType1, typename MatrixType2,
    std::enable_if_t<std::is_base_of_v<MatrixBase, MatrixType1> && std::is_base_of_v<MatrixBase, MatrixType2>, bool> = true>
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

template<typename MatrixType1, typename MatrixType2>
auto dot(const MatrixType1& M1, const MatrixType2& M2){
    // Dot is only valid if both a row/column vectors of the same size
    if (M1.size() != M2.size() 
    || (M1.dim().z != 1) 
    || (M2.dim().z != 1) 
    || (M1.dim().x != 1 && M1.dim().y != 1)
    || (M2.dim().x != 1 && M2.dim().y != 1))
    {
        std::stringstream ss;
        ss << "Cannot get dot product between matrices of size " << M1.dim() << " and " << M2.dim();
        throw MatrixSizeException(ss.str());
    }

    return std::inner_product(M1.begin(), M1.end(), M2.begin(), 0);
}

template<typename MatrixType1, typename MatrixType2>
class MatrixMultiplyResult : public MatrixBase{

    template<typename T1, typename T2>
    friend auto matrixMultiply(const T1&, const T2&);

public:

    using type = typename std::common_type_t<typename MatrixType1::type, typename MatrixType2::type>;
    type operator()(const dim3& idx) const{
        dim3 row_start_idx(idx.x, 0, idx.z);
        const dim3 col_start_idx(0, idx.y, idx.z);
        auto matrix_2_it = MatrixIterator<const MatrixType2>(m2_, col_start_idx);
        type sum{};
        for (size_t i = 0; i < row_size_; ++i, ++row_start_idx.y, ++matrix_2_it){
            sum += *matrix_2_it * m1_->operator()(row_start_idx);
        }
        return sum;
    }

    type operator()(uint x, uint y, uint z) const{
        return operator()(dim3(x,y,z));
    }

    auto begin() {return MatrixIterator<MatrixMultiplyResult<MatrixType1, MatrixType2>>(this, {0, 0, 0});}
    auto end() {return MatrixIterator<MatrixMultiplyResult<MatrixType1, MatrixType2>>(this, {0, 0, dim_.z});}
    auto begin() const {return MatrixIterator<const MatrixMultiplyResult<MatrixType1, MatrixType2>>(this, {0, 0, 0});}
    auto end() const {return MatrixIterator<const MatrixMultiplyResult<MatrixType1, MatrixType2>>(this, {0, 0, dim_.z});}

private:
    const MatrixType1* m1_;
    const MatrixType2* m2_;
    size_t row_size_;

    static auto makeMatrixMultiplyResult(const MatrixType1& M1, const MatrixType2& M2){
        return MatrixMultiplyResult(M1, M2);
    }

    MatrixMultiplyResult(const MatrixType1& M1, const MatrixType2& M2) :
    MatrixBase({M1.dim().x, M2.dim().y, M1.dim().z}),
    m1_(&M1), m2_(&M2), row_size_(M1.dim().y)
    {}
};

template<typename MatrixType1, typename MatrixType2>
auto matrixMultiply(const MatrixType1& M1, const MatrixType2& M2){
    if ((M1.dim().z != M2.dim().z) || (M1.dim().y != M2.dim().x)){
        std::stringstream ss;
        ss << "Cannot perform matrix multiplication between matrices of size " << M1.dim() << " and " << M2.dim();
        throw MatrixSizeException(ss.str());
    }

    return MatrixMultiplyResult<MatrixType1, MatrixType2>::makeMatrixMultiplyResult(M1, M2);
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

    type operator()(uint x, uint y, uint z) const{
        return operator()(dim3(x,y,z));
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
    std::enable_if_t<std::is_arithmetic_v<Scalar> && std::is_base_of_v<MatrixBase, MatrixType1>, bool> = true>
auto operator+(const MatrixType1& M1, const Scalar& S){
    return ScalarOperationResult(M1, S, scalarAdd<MatrixType1, Scalar>);
}

template<typename MatrixType1, typename Scalar,
    std::enable_if_t<std::is_arithmetic_v<Scalar> && std::is_base_of_v<MatrixBase, MatrixType1>, bool> = true>
auto operator+(const Scalar& S, const MatrixType1& M1){
    return ScalarOperationResult(M1, S, scalarAdd<MatrixType1, Scalar>);
}

template<typename MatrixType1, typename Scalar,
    std::enable_if_t<std::is_arithmetic_v<Scalar> && std::is_base_of_v<MatrixBase, MatrixType1>, bool> = true>
auto operator-(const MatrixType1& M1, const Scalar& S){
    return ScalarOperationResult(M1, S, scalarSubtract<MatrixType1, Scalar>);
}

template<typename MatrixType1, typename Scalar,
    std::enable_if_t<std::is_arithmetic_v<Scalar> && std::is_base_of_v<MatrixBase, MatrixType1>, bool> = true>
auto operator-(const Scalar& S, const MatrixType1& M1){
    return ScalarOperationResult(M1, S, scalarSubtract<MatrixType1, Scalar>);
}

template<typename MatrixType1, typename Scalar,
    std::enable_if_t<std::is_arithmetic_v<Scalar> && std::is_base_of_v<MatrixBase, MatrixType1>, bool> = true>
auto operator*(const MatrixType1& M1, const Scalar& S){
    return ScalarOperationResult(M1, S, scalarMultiply<MatrixType1, Scalar>);
}

template<typename MatrixType1, typename Scalar,
    std::enable_if_t<std::is_arithmetic_v<Scalar> && std::is_base_of_v<MatrixBase, MatrixType1>, bool> = true>
auto operator*(const Scalar& S, const MatrixType1& M1){
    return ScalarOperationResult(M1, S, scalarMultiply<MatrixType1, Scalar>);
}


template<typename MatrixType1, typename Scalar,
    std::enable_if_t<std::is_arithmetic_v<Scalar> && std::is_base_of_v<MatrixBase, MatrixType1>, bool> = true>
auto operator/(const MatrixType1& M1, const Scalar& S){
    return ScalarOperationResult(M1, S, scalarDivide<MatrixType1, Scalar>);
}

template<typename MatrixType1, typename Scalar,
    std::enable_if_t<std::is_arithmetic_v<Scalar> && std::is_base_of_v<MatrixBase, MatrixType1>, bool> = true>
auto operator/(const Scalar& S, const MatrixType1& M1){
    return ScalarOperationResult(M1, S, scalarDivide<MatrixType1, Scalar>);
}


// ================================================================================================
// Matrix unary operation
// ================================================================================================

template<typename MatrixType, class UnaryOp>
class UnaryOperationResult : public MatrixBase{
public:

    template<typename T1>
    friend auto transpose(const T1&);

    using type = typename std::remove_const_t<typename MatrixType::type>;

    UnaryOperationResult(const MatrixType& M, UnaryOp Op) :
    MatrixBase(M.dim()),
    m_(&M), op(Op) 
    {}

    type operator()(dim3 idx) const{
        return std::invoke(op, *m_, idx);
    }

    type operator()(uint x, uint y, uint z) const{
        return operator()(dim3(x,y,z));
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
auto apply(const MatrixType& M, UnaryOp op){
    return UnaryOperationResult(M, [op](const MatrixType& M_, const dim3& idx_){return std::invoke(op, M_(idx_));});
}

template<typename MatrixType, typename UnaryOp>
void modify(MatrixType& M, UnaryOp op){
    std::for_each(M.begin(), M.end(), [&](auto& in){in = op(in);});
}

template<typename MatrixType>
auto abs(const MatrixType& M){
    using input_type = typename MatrixType::type;
    using return_type = decltype(std::abs(input_type{}));
    return apply<MatrixType, return_type(*)(input_type)>(M, std::abs);
}

template<typename MatrixType>
auto exp(const MatrixType& M){
    using input_type = typename MatrixType::type;
    using return_type = decltype(std::exp(input_type{}));
    return apply<MatrixType, return_type(*)(input_type)>(M, std::exp);
}

template<typename MatrixType>
auto transpose(const MatrixType& M){
    auto ret_val = UnaryOperationResult(M, [](const MatrixType& M, const dim3& idx){return M(dim3(idx.y, idx.x, idx.z));});
    ret_val.dim_.x = M.dim().y;
    ret_val.dim_.y = M.dim().x;
    return ret_val;
}

template<size_t NumSteps, typename MatrixType>
auto rotate(const MatrixType& M){
    static_assert(NumSteps > 0 && NumSteps < 4);
    if (not M.isSquare()){
        std::stringstream ss;
        ss << "Cannot rotate matrix of size " << M.dim() << " because it is not square";
        throw MatrixTransformException(ss.str());
    }

    auto rotatedIndex = [](const MatrixType& M, const dim3& idx){
        dim3 new_idx;
        new_idx.z = idx.z;
        if constexpr (NumSteps == 1){
            new_idx.x = M.dim().y - idx.y - 1;
            new_idx.y = idx.x;
        }else if (NumSteps == 2){
            new_idx.x = M.dim().x - idx.x - 1;
            new_idx.y = M.dim().y - idx.y - 1;
        }else if (NumSteps == 3){
            new_idx.x = idx.y;
            new_idx.y = M.dim().x - idx.x - 1;
        }

        return M(new_idx);
    };

    return UnaryOperationResult(M, rotatedIndex);
}

// ================================================================================================
// Matrix reduction operations
// ================================================================================================

template<typename MatrixType>
auto sum(const MatrixType& M){
    return std::accumulate(M.begin(), M.end(), 0);
}

template<typename MatrixType>
auto mean(const MatrixType& M){
    return sum(M) / M.size();
}

template<typename MatrixType>
auto max(const MatrixType& M){
    return *std::max_element(M.begin(), M.end());
}

template<typename MatrixType>
auto min(const MatrixType& M){
    return *std::min_element(M.begin(), M.end());
}

template<typename MatrixType>
auto minIndex(const MatrixType& M){
    return std::min_element(M.begin(), M.end()).idx();
}

template<typename MatrixType>
auto maxIndex(const MatrixType& M){
    return std::max_element(M.begin(), M.end()).idx();
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
