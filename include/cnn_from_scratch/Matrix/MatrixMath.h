#pragma once

#include <assert.h>
#include <numeric>
#include <type_traits>
#include <functional>
#include "cnn_from_scratch/Matrix/MatrixBase.h"

#define bad_access_error R"|||(                                                 \
                                                                                \
    MATRIX ACCESS ERROR                                                         \
    You tried to use a temporary Matrix operation result as an lvalue           \
    This can lead to dangling references and so it has been disabled            \
                                                                                \
    auto myMat = /** Matrix Operation **/                                       \
    ^ ^                                                                         \
    (This saves the temporary instead of creating a new matrix)                 \
                                                                                \
    my_cnn::SimpleMatrix<T> = /** Matrix Operation **/                          \
    ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^                                                       \
    (Do this instead to get the final result)                                   \
                                                                                \
    If you really know what you're doing, you can still use std::move           \
    to access the result as an rvalue. This is only valid if a single           \
    binary matrix-matrix operation, a single matrix-scalar opration,            \
    or a single unary matrix function was used                                  \
                                                                                \
)|||"

namespace my_cnn{
    
template<typename T>
class SimpleMatrix;

// ================================================================================================
// Self modification math
// ================================================================================================

template<typename MatrixType, typename UnaryOp>
void modify(MatrixType&& M, UnaryOp op){
    std::for_each(std::forward<MatrixType>(M).begin(), std::forward<MatrixType>(M).end(), [&](auto& in){in = op(in);});
}

template<typename MatrixType, typename Scalar, std::enable_if_t<std::is_convertible_v<Scalar, typename std::remove_reference_t<MatrixType>::type>, bool> = true>
MatrixType& operator+=(MatrixType&& M, const Scalar& other){
    std::for_each(std::begin(M), std::end(M), [&](auto& val){val += other;});
    return M;
}

template<typename MatrixType, typename Other, std::enable_if_t<std::is_base_of_v<MatrixBase, Other>, bool> = true>
MatrixType& operator+=(MatrixType&& M, Other&& other){
    checkSize(M, other);
    for (auto it = std::begin(M); it != std::end(M); it++){
        *it += std::forward<Other>(other)(it.idx());
    }
    return M;
}

template<typename MatrixType, typename Scalar, std::enable_if_t<std::is_convertible_v<Scalar, typename std::remove_reference_t<MatrixType>::type>, bool> = true>
MatrixType& operator-=(MatrixType&& M, const Scalar& other){
    std::for_each(std::begin(M), std::end(M), [&](auto& val){val -= other;});
    return M;
}

template<typename MatrixType, typename Other, std::enable_if_t<std::is_base_of_v<MatrixBase, Other>, bool> = true>
MatrixType& operator-=(MatrixType&& M, Other&& other){
    checkSize(M, other);
    for (auto it = std::begin(M); it != std::end(M); it++){
        *it -= std::forward<Other>(other)(it.idx());
    }
    return M;
}

template<typename MatrixType, typename Scalar, std::enable_if_t<std::is_convertible_v<Scalar, typename std::remove_reference_t<MatrixType>::type>, bool> = true>
MatrixType& operator*=(MatrixType&& M, const Scalar& other){
    std::for_each(std::begin(M), std::end(M), [&](auto& val){val *= other;});
    return M;
}

template<typename MatrixType, typename Other, std::enable_if_t<std::is_base_of_v<MatrixBase, Other>, bool> = true>
MatrixType& operator*=(MatrixType&& M, Other&& other){
    checkSize(M, other);
    for (auto it = std::begin(M); it != std::end(M); it++){
        *it *= std::forward<Other>(other)(it.idx());
    }
    return M;
}

template<typename MatrixType, typename Scalar, std::enable_if_t<std::is_convertible_v<Scalar, typename std::remove_reference_t<MatrixType>::type>, bool> = true>
MatrixType& operator/=(MatrixType&& M, const Scalar& other){
    std::for_each(std::begin(M), std::end(M), [&](auto& val){val /= other;});
    return M;
}

template<typename MatrixType, typename Other, std::enable_if_t<std::is_base_of_v<MatrixBase, Other>, bool> = true>
MatrixType& operator/=(MatrixType&& M, Other&& other){
    checkSize(M, other);
    for (auto it = std::begin(M); it != std::end(M); it++){
        *it /= std::forward<Other>(other)(it.idx());
    }
    return M;
}

// ================================================================================================
// Matrix - Matrix math result
// ================================================================================================

template<typename MatrixType1, typename MatrixType2, class BinaryOp, 
    typename = std::enable_if_t<
        std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType1>> 
     && std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType2>>
    >>
class MatrixOperationResult : public MatrixBase{
public:
    using this_type = MatrixOperationResult<MatrixType1, MatrixType2, BinaryOp, std::enable_if_t<
        std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType1>> 
     && std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType2>>
    >>;

    using MT1 = typename std::remove_reference_t<MatrixType1>;
    using MT2 = typename std::remove_reference_t<MatrixType2>;

    using type = typename std::common_type_t<typename MT1::type, typename MT2::type>;

    template<typename = std::enable_if_t<std::is_reference_v<MatrixType1> && std::is_reference_v<MatrixType2>>>
    MatrixOperationResult(MatrixType1 M1, MatrixType2 M2, BinaryOp Op) :
    MatrixBase(M1.dim()),
    m1_(&M1), m2_(&M2), op(Op) 
    {
        checkSize(M1, M2);
    }

    type operator()(const dim3& idx) && {
        return std::invoke(op, std::forward<MatrixType1>(*m1_), std::forward<MatrixType2>(*m2_), idx);
    }

    type operator()(uint x, uint y, uint z) && {
        return operator()(dim3(x,y,z));
    }

    template<typename U>
    type operator()(U, U = {}, U = {}) const &{
        static_assert(std::is_same_v<U, void>, bad_access_error);
        static_assert(not std::is_same_v<U, void>);
        return type{};
    }

    auto begin() && {return MatrixIterator<this_type&&>(std::move(*this), {0, 0, 0});}
    auto end() && {return MatrixIterator<this_type&&>(std::move(*this), {0, 0, dim_.z});}
    auto begin() const && {return MatrixIterator<const this_type&&>(*this, {0, 0, 0});}
    auto end() const && {return MatrixIterator<const this_type&&>(*this, {0, 0, dim_.z});}


private:
    MT1* m1_;
    MT2* m2_;
    BinaryOp op;
};

template<typename MatrixType1, typename MatrixType2>
auto elementWiseAdd(MatrixType1&& M1, MatrixType2&& M2, const dim3& idx){
    return std::forward<MatrixType1>(M1)(idx) + std::forward<MatrixType2>(M2)(idx);
}

template<typename MatrixType1, typename MatrixType2>
auto elementWiseMultiply(MatrixType1&& M1, MatrixType2&& M2, const dim3& idx){
    return std::forward<MatrixType1>(M1)(idx) * std::forward<MatrixType2>(M2)(idx);
}

template<typename MatrixType1, typename MatrixType2>
auto elementWiseDivide(MatrixType1&& M1, MatrixType2&& M2, const dim3& idx){
    return std::forward<MatrixType1>(M1)(idx) / std::forward<MatrixType2>(M2)(idx);
}

template<typename MatrixType1, typename MatrixType2>
auto elementWiseSubtract(MatrixType1&& M1, MatrixType2&& M2, const dim3& idx){
    return std::forward<MatrixType1>(M1)(idx) - std::forward<MatrixType2>(M2)(idx);
}

template<typename MatrixType1, typename MatrixType2,
    std::enable_if_t<
        std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType1>> 
     && std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType2>>, bool> = true>
auto operator+(MatrixType1&& M1, MatrixType2&& M2){
    return MatrixOperationResult<
        decltype(std::forward<MatrixType1>(M1)),
        decltype(std::forward<MatrixType2>(M2)),
        decltype(&elementWiseAdd<MatrixType1, MatrixType2>)>
    (std::forward<MatrixType1>(M1), std::forward<MatrixType2>(M2), elementWiseAdd<MatrixType1, MatrixType2>);
}

template<typename MatrixType1, typename MatrixType2,
    std::enable_if_t<
        std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType1>> 
     && std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType2>>, bool> = true>
auto operator-(MatrixType1&& M1, MatrixType2&& M2){
    return MatrixOperationResult<
        decltype(std::forward<MatrixType1>(M1)),
        decltype(std::forward<MatrixType2>(M2)),
        decltype(&elementWiseSubtract<MatrixType1, MatrixType2>)>
    (std::forward<MatrixType1>(M1), std::forward<MatrixType2>(M2), elementWiseSubtract<MatrixType1, MatrixType2>);
}

template<typename MatrixType1, typename MatrixType2,
    std::enable_if_t<
        std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType1>> 
     && std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType2>>, bool> = true>
auto operator*(MatrixType1&& M1, MatrixType2&& M2){
    return MatrixOperationResult<
        decltype(std::forward<MatrixType1>(M1)),
        decltype(std::forward<MatrixType2>(M2)),
        decltype(&elementWiseMultiply<MatrixType1, MatrixType2>)>
    (std::forward<MatrixType1>(M1), std::forward<MatrixType2>(M2), elementWiseMultiply<MatrixType1, MatrixType2>);
}

template<typename MatrixType1, typename MatrixType2,
    std::enable_if_t<
        std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType1>> 
     && std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType2>>, bool> = true>
auto operator/(MatrixType1&& M1, MatrixType2&& M2){
    return MatrixOperationResult<
        decltype(std::forward<MatrixType1>(M1)),
        decltype(std::forward<MatrixType2>(M2)),
        decltype(&elementWiseDivide<MatrixType1, MatrixType2>)>
    (std::forward<MatrixType1>(M1), std::forward<MatrixType2>(M2), elementWiseDivide<MatrixType1, MatrixType2>);
}

template<typename MatrixType1, typename MatrixType2,
    std::enable_if_t<
        std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType1>> 
     && std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType2>>, bool> = true>
bool operator==(MatrixType1&& M1, MatrixType2&& M2){
    if (M1.dim() != M2.dim()) return false;
    bool equal = true;
    for (auto it = std::forward<MatrixType1>(M1).begin(); it != std::forward<MatrixType1>(M1).end() && equal; it++){
        equal = equal && (*it == std::forward<MatrixType2>(M2)(it.idx()));
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
    friend auto matrixMultiply(T1&&, T2&&);

public:
    using MT1  = typename std::remove_reference_t<MatrixType1>;
    using MT2  = typename std::remove_reference_t<MatrixType2>;
    using type = typename std::common_type_t<typename MT1::type, typename MT2::type>;

    type operator()(const dim3& idx) const{
        dim3 row_start_idx(idx.x, 0, idx.z);
        const dim3 col_start_idx(0, idx.y, idx.z);
        auto matrix_2_it = MatrixIterator<MatrixType2>(std::forward<MatrixType2>(*m2_), col_start_idx);
        type sum{};
        for (size_t i = 0; i < row_size_; ++i, ++row_start_idx.y, ++matrix_2_it){
            sum += *matrix_2_it * std::forward<MatrixType1>(*m1_)(row_start_idx);
        }
        return sum;
    }

    type operator()(uint x, uint y, uint z) const{
        return operator()(dim3(x,y,z));
    }

    template<typename U>
    type operator()(U, U = {}, U = {}) const &{
        static_assert(std::is_same_v<U, void>, bad_access_error);
        static_assert(not std::is_same_v<U, void>);
        return type{};
    }

    auto begin() && {return MatrixIterator<MatrixMultiplyResult<MatrixType1, MatrixType2>&&>(std::move(*this), {0, 0, 0});}
    auto end() && {return MatrixIterator<MatrixMultiplyResult<MatrixType1, MatrixType2>&&>(std::move(*this), {0, 0, dim_.z});}
    auto begin() const && {return MatrixIterator<const MatrixMultiplyResult<MatrixType1, MatrixType2>&&>(std::move(*this), {0, 0, 0});}
    auto end() const && {return MatrixIterator<const MatrixMultiplyResult<MatrixType1, MatrixType2>&&>(std::move(*this), {0, 0, dim_.z});}

private:
    MT1* m1_;
    MT2* m2_;
    size_t row_size_;

    template<typename = std::enable_if_t<std::is_reference_v<MatrixType1> && std::is_reference_v<MatrixType2>>>
    MatrixMultiplyResult(MatrixType1 M1, MatrixType2 M2) :
    MatrixBase({M1.dim().x, M2.dim().y, M1.dim().z}),
    m1_(&M1), m2_(&M2), row_size_(M1.dim().y)
    {}
};

template<typename MatrixType1, typename MatrixType2>
auto matrixMultiply(MatrixType1&& M1, MatrixType2&& M2){
    if ((M1.dim().z != M2.dim().z) || (M1.dim().y != M2.dim().x)){
        std::stringstream ss;
        ss << "Cannot perform matrix multiplication between matrices of size " << M1.dim() << " and " << M2.dim();
        throw MatrixSizeException(ss.str());
    }

    return MatrixMultiplyResult<decltype(std::forward<MatrixType1>(M1)), decltype(std::forward<MatrixType2>(M2))>
            (std::forward<MatrixType1>(M1), std::forward<MatrixType2>(M2));
}

// ================================================================================================
// Matrix - Scalar math result
// ================================================================================================

template<typename MatrixType, typename ScalarType, class BinaryOp,
    typename = std::enable_if_t<std::is_reference_v<MatrixType>>>
class ScalarOperationResult : public MatrixBase{
public:
    using MT = std::remove_reference_t<MatrixType>;
    using type = typename std::common_type_t<typename MT::type, ScalarType>;

    ScalarOperationResult(MatrixType M, ScalarType S, BinaryOp Op) :
    MatrixBase(M.dim()),
    m_(&M), s_(S), op(Op) 
    {}

    type operator()(dim3 idx) const &&{
        return std::invoke(op, std::forward<MatrixType>(*m_), s_, idx);
    }

    type operator()(uint x, uint y, uint z) const &&{
        return operator()(dim3(x,y,z));
    }

    template<typename U>
    type operator()(U, U = {}, U = {}) const &{
        static_assert(std::is_same_v<U, void>, bad_access_error);
        static_assert(not std::is_same_v<U, void>);
        return type{};
    }

    auto begin() && {return MatrixIterator<ScalarOperationResult<MatrixType, ScalarType, BinaryOp>&&>(std::move(*this), {0, 0, 0});}
    auto end() && {return MatrixIterator<ScalarOperationResult<MatrixType, ScalarType, BinaryOp>&&>(std::move(*this), {0, 0, dim_.z});}
    auto begin() const && {return MatrixIterator<const ScalarOperationResult<MatrixType, ScalarType, BinaryOp>&&>(std::move(*this), {0, 0, 0});}
    auto end() const && {return MatrixIterator<const ScalarOperationResult<MatrixType, ScalarType, BinaryOp>&&>(std::move(*this), {0, 0, dim_.z});}

private:
    MT* m_;
    const ScalarType s_;
    BinaryOp op;
};

template<typename MatrixType, typename Scalar>
auto scalarAdd(MatrixType&& M1, const Scalar& S, const dim3& idx){
    return std::forward<MatrixType>(M1)(idx) + S;
}

template<typename MatrixType, typename Scalar>
auto scalarMultiply(MatrixType&& M1, const Scalar& S, const dim3& idx){
    return std::forward<MatrixType>(M1)(idx) * S;
}

template<typename MatrixType, typename Scalar>
auto scalarDivide(MatrixType&& M1, const Scalar& S, const dim3& idx){
    return std::forward<MatrixType>(M1)(idx) / S;
}

template<typename MatrixType, typename Scalar>
auto scalarSubtract(MatrixType&& M1, const Scalar& S, const dim3& idx){
    return std::forward<MatrixType>(M1)(idx) - S;
}

template<typename MatrixType, typename Scalar,
    std::enable_if_t<
        std::is_arithmetic_v<Scalar> 
     && std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType>>, bool> = true>
auto operator+(MatrixType&& M1, const Scalar& S){
    return ScalarOperationResult<
        decltype(std::forward<MatrixType>(M1)),
        Scalar,
        decltype(&scalarAdd<MatrixType, Scalar>)
    >
    (std::forward<MatrixType>(M1), S, scalarAdd<MatrixType, Scalar>);
}

template<typename MatrixType, typename Scalar,
    std::enable_if_t<
        std::is_arithmetic_v<Scalar> 
     && std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType>>, bool> = true>
auto operator+(const Scalar& S, MatrixType&& M1){
    return ScalarOperationResult<
        decltype(std::forward<MatrixType>(M1)),
        Scalar,
        decltype(&scalarAdd<MatrixType, Scalar>)
    >
    (std::forward<MatrixType>(M1), S, scalarAdd<MatrixType, Scalar>);
}

template<typename MatrixType, typename Scalar,
    std::enable_if_t<
        std::is_arithmetic_v<Scalar> 
     && std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType>>, bool> = true>
auto operator-(MatrixType&& M1, const Scalar& S){
    return ScalarOperationResult<
        decltype(std::forward<MatrixType>(M1)),
        Scalar,
        decltype(&scalarSubtract<MatrixType, Scalar>)
    >
    (std::forward<MatrixType>(M1), S, scalarSubtract<MatrixType, Scalar>);
}

template<typename MatrixType, typename Scalar,
    std::enable_if_t<
        std::is_arithmetic_v<Scalar> 
     && std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType>>, bool> = true>
auto operator-(const Scalar& S, MatrixType&& M1){
    return ScalarOperationResult<
        decltype(std::forward<MatrixType>(M1)),
        Scalar,
        decltype(&scalarSubtract<MatrixType, Scalar>)
    >
    (std::forward<MatrixType>(M1), S, scalarSubtract<MatrixType, Scalar>);
}

template<typename MatrixType, typename Scalar,
    std::enable_if_t<
        std::is_arithmetic_v<Scalar> 
     && std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType>>, bool> = true>
auto operator*(MatrixType&& M1, const Scalar& S){
    return ScalarOperationResult<
        decltype(std::forward<MatrixType>(M1)),
        Scalar,
        decltype(&scalarMultiply<MatrixType, Scalar>)
    >
    (std::forward<MatrixType>(M1), S, scalarMultiply<MatrixType, Scalar>);
}

template<typename MatrixType, typename Scalar,
    std::enable_if_t<
        std::is_arithmetic_v<Scalar> 
     && std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType>>, bool> = true>
auto operator*(const Scalar& S, MatrixType&& M1){
    return ScalarOperationResult<
        decltype(std::forward<MatrixType>(M1)),
        Scalar,
        decltype(&scalarMultiply<MatrixType, Scalar>)
    >
    (std::forward<MatrixType>(M1), S, scalarMultiply<MatrixType, Scalar>);
}


template<typename MatrixType, typename Scalar,
    std::enable_if_t<
        std::is_arithmetic_v<Scalar> 
     && std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType>>, bool> = true>
auto operator/(MatrixType&& M1, const Scalar& S){
    return ScalarOperationResult<
        decltype(std::forward<MatrixType>(M1)),
        Scalar,
        decltype(&scalarDivide<MatrixType, Scalar>)
    >
    (std::forward<MatrixType>(M1), S, scalarDivide<MatrixType, Scalar>);
}

template<typename MatrixType, typename Scalar,
    std::enable_if_t<
        std::is_arithmetic_v<Scalar> 
     && std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType>>, bool> = true>
auto operator/(const Scalar& S, MatrixType&& M1){
    return ScalarOperationResult<
        decltype(std::forward<MatrixType>(M1)),
        Scalar,
        decltype(&scalarDivide<MatrixType, Scalar>)
    >
    (std::forward<MatrixType>(M1), S, scalarDivide<MatrixType, Scalar>);
}


// ================================================================================================
// Matrix unary operation
// ================================================================================================

template<typename MatrixType, class UnaryOp>
class UnaryOperationResult : public MatrixBase{
public:

    template<typename T1>
    friend auto transpose(T1&&);

    using MT = typename std::remove_reference_t<MatrixType>;
    using type = typename std::remove_const_t<typename MT::type>;

    UnaryOperationResult(MatrixType&& M, UnaryOp Op) :
    MatrixBase(M.dim()),
    m_(&M), op(Op) 
    {}

    type operator()(dim3 idx) const && {
        return std::invoke(op, std::forward<MatrixType>(*m_), idx);
    }

    type operator()(uint x, uint y, uint z) const && {
        return operator()(dim3(x,y,z));
    }

    template<typename U>
    type operator()(U, U = {}, U = {}) const &{
        static_assert(std::is_same_v<U, void>, bad_access_error);
        static_assert(not std::is_same_v<U, void>);
        return type{};
    }

    auto begin() && {return MatrixIterator<UnaryOperationResult<MatrixType, UnaryOp>&&>(std::move(*this), {0, 0, 0});}
    auto end() && {return MatrixIterator<UnaryOperationResult<MatrixType, UnaryOp>&&>(std::move(*this), {0, 0, dim_.z});}
    auto begin() const && {return MatrixIterator<const UnaryOperationResult<MatrixType, UnaryOp>&&>(std::move(*this), {0, 0, 0});}
    auto end() const && {return MatrixIterator<const UnaryOperationResult<MatrixType, UnaryOp>&&>(std::move(*this), {0, 0, dim_.z});}

private:
    MT* m_;
    UnaryOp op;
};

template<typename MatrixType, typename UnaryOp>
auto apply(MatrixType&& M, UnaryOp op){
    auto invoker = [op](MatrixType&& M_, const dim3& idx_){
        return std::invoke(op, std::forward<MatrixType>(M_)(idx_));
    };

    return UnaryOperationResult<
        decltype(std::forward<MatrixType>(M)), 
        decltype(invoker)>
    (std::forward<MatrixType>(M), invoker);
}

template<typename MatrixType>
auto abs(MatrixType&& M){
    using input_type = typename MatrixType::type;
    using return_type = decltype(std::abs(input_type{}));
    return my_cnn::apply<MatrixType, return_type(*)(input_type)>(std::forward<MatrixType>(M), std::abs);
}

template<typename MatrixType>
auto exp(MatrixType&& M){
    using input_type = typename MatrixType::type;
    using return_type = decltype(std::exp(input_type{}));
    return my_cnn::apply<MatrixType, return_type(*)(input_type)>(std::forward<MatrixType>(M), std::exp);
}

template<typename MatrixType>
auto transpose(MatrixType&& M){
    auto transposeIdx = [](MatrixType&& M, const dim3& idx){
        return std::forward<MatrixType>(M)(dim3(idx.y, idx.x, idx.z));
    };
    auto&& ret_val = UnaryOperationResult<MatrixType, decltype(transposeIdx)>(std::forward<MatrixType>(M), transposeIdx);
    ret_val.dim_.x = M.dim().y;
    ret_val.dim_.y = M.dim().x;
    return ret_val;
}

template<size_t NumSteps, typename MatrixType,
    typename = std::enable_if_t<std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType>>>>
auto rotate(MatrixType&& M){
    static_assert(NumSteps > 0 && NumSteps < 4);
    if (not M.isSquare()){
        std::stringstream ss;
        ss << "Cannot rotate matrix of size " << M.dim() << " because it is not square";
        throw MatrixTransformException(ss.str());
    }

    auto rotatedIndex = [](MatrixType&& M, const dim3& idx){
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

        return std::forward<MatrixType>(M)(new_idx);
    };

    return UnaryOperationResult(std::forward<MatrixType>(M), rotatedIndex);
}

// ================================================================================================
// Matrix reduction operations
// ================================================================================================

template<typename MatrixType>
auto sum(MatrixType&& M){
    return std::accumulate(std::forward<MatrixType>(M).begin(), std::forward<MatrixType>(M).end(), 0);
}

template<typename MatrixType>
auto mean(MatrixType&& M){
    return sum(M) / M.size();
}

template<typename MatrixType>
auto max(MatrixType&& M){
    return *std::max_element(std::forward<MatrixType>(M).begin(), std::forward<MatrixType>(M).end());
}

template<typename MatrixType>
auto min(MatrixType&& M){
    return *std::min_element(std::forward<MatrixType>(M).begin(), std::forward<MatrixType>(M).end());
}

template<typename MatrixType>
auto minIndex(MatrixType&& M){
    return std::min_element(std::forward<MatrixType>(M).begin(), std::forward<MatrixType>(M).end()).idx();
}

template<typename MatrixType>
auto maxIndex(MatrixType&& M){
    return std::max_element(std::forward<MatrixType>(M).begin(), std::forward<MatrixType>(M).end()).idx();
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
