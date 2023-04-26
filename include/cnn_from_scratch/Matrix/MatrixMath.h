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

template<typename MatrixType1, typename MatrixType2>
using CommonMatrixType = typename std::common_type_t<
        typename std::remove_reference_t<MatrixType1>::type,
        typename std::remove_reference_t<MatrixType2>::type
    >;


template<typename MatrixType1, typename MatrixType2 = MatrixBase>
using IsMatrixBase = typename std::enable_if_t<
        std::is_base_of_v<
            MatrixBase,
            typename std::remove_reference_t<MatrixType1>
        >
        && 
        std::is_base_of_v<
            MatrixBase,
            typename std::remove_reference_t<MatrixType2>
        >
    >;

template<typename MatrixType>
using MatrixStorageType = typename std::conditional_t<
    std::is_lvalue_reference_v<MatrixType>,
    MatrixType,
    std::remove_reference_t<MatrixType>
>;

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

template<typename MatrixType1, typename MatrixType2, class BinaryOp>
class ElementWiseMatrixOperationResult : public MatrixBase{
public:
    using this_type = ElementWiseMatrixOperationResult<MatrixType1, MatrixType2, BinaryOp>;
    using type = CommonMatrixType<MatrixType1, MatrixType2>;

    using MT1 = MatrixStorageType<MatrixType1>;
    using MT2 = MatrixStorageType<MatrixType2>;

    // template<typename = std::enable_if_t<std::is_reference_v<MatrixType1> && std::is_reference_v<MatrixType2>>>
    ElementWiseMatrixOperationResult(MT1 M1, MT2 M2, BinaryOp Op) :
    MatrixBase(M1.dim()),
    m1_(std::forward<MatrixType1>(M1)), m2_(std::forward<MatrixType2>(M2)), op(Op) 
    {
        checkSize(m1_, m2_);
    }

    type operator()(const dim3& idx) const {
        return std::invoke(op, m1_, m2_, idx);
    }

    type operator()(uint x, uint y, uint z) const {
        return operator()(dim3(x,y,z));
    }

    auto begin() {return MatrixIterator<this_type&>(*this, {0, 0, 0});}
    auto end() {return MatrixIterator<this_type&>(*this, {0, 0, dim_.z});}
    auto begin() const {return MatrixIterator<const this_type&>(*this, {0, 0, 0});}
    auto end() const {return MatrixIterator<const this_type&>(*this, {0, 0, dim_.z});}


private:
    const MT1 m1_;
    const MT2 m2_;
    BinaryOp op;
};

#define ADD_MATRIX_MATRIX_OPERATOR(op)                                                                  \
                                                                                                        \
template<typename MatrixType1, typename MatrixType2, typename = IsMatrixBase<MatrixType1, MatrixType2>> \
auto operator op (MatrixType1&& M1, MatrixType2&& M2){                                                  \
    auto OP = [](const MatrixType1& m1, const MatrixType2& m2, const dim3& idx) {                       \
        return m1(idx) op m2(idx);                                                                      \
    };                                                                                                  \
                                                                                                        \
    return ElementWiseMatrixOperationResult<decltype(M1), decltype(M2), decltype(OP)>                   \
    (std::forward<MatrixType1>(M1), std::forward<MatrixType2>(M2), OP);                                 \
}

ADD_MATRIX_MATRIX_OPERATOR(+)
ADD_MATRIX_MATRIX_OPERATOR(-)
ADD_MATRIX_MATRIX_OPERATOR(/)
ADD_MATRIX_MATRIX_OPERATOR(*)
ADD_MATRIX_MATRIX_OPERATOR(<)
ADD_MATRIX_MATRIX_OPERATOR(<=)
ADD_MATRIX_MATRIX_OPERATOR(>)
ADD_MATRIX_MATRIX_OPERATOR(>=)
ADD_MATRIX_MATRIX_OPERATOR(!=)
ADD_MATRIX_MATRIX_OPERATOR(==)

#undef ADD_MATRIX_MATRIX_OPERATOR

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

template<typename MatrixType1, typename MatrixType2, typename BinaryOp>
class GeneralMatrixOperationResult : public MatrixBase{
public:
    using MT1  = MatrixStorageType<MatrixType1>;
    using MT2  = MatrixStorageType<MatrixType2>;
    using type = CommonMatrixType<MT1, MT2>;

    GeneralMatrixOperationResult(MT1 M1, MT2 M2, dim3 dim, BinaryOp op) :
    MatrixBase(dim),
    m1_(std::forward<MatrixType1>(M1)), 
    m2_(std::forward<MatrixType2>(M2)),
    op_(op)
    {}

    type operator()(const dim3& idx) const {
        return std::invoke(op_, m1_, m2_, idx);
    }

    type operator()(uint x, uint y, uint z) const {
        return operator()(dim3(x,y,z));
    }

    auto begin() {return MatrixIterator<GeneralMatrixOperationResult<MatrixType1, MatrixType2, BinaryOp>&>(*this, {0, 0, 0});}
    auto end() {return MatrixIterator<GeneralMatrixOperationResult<MatrixType1, MatrixType2, BinaryOp>&>(*this, {0, 0, dim_.z});}
    auto begin() const {return MatrixIterator<const GeneralMatrixOperationResult<MatrixType1, MatrixType2, BinaryOp>&>(*this, {0, 0, 0});}
    auto end() const {return MatrixIterator<const GeneralMatrixOperationResult<MatrixType1, MatrixType2, BinaryOp>&>(*this, {0, 0, dim_.z});}

private:
    const MT1 m1_;
    const MT2 m2_;
    BinaryOp op_;
};

template<typename MatrixType1, typename MatrixType2, typename = IsMatrixBase<MatrixType1, MatrixType2>>
auto matrixMultiply(MatrixType1&& M1, MatrixType2&& M2){
    if ((M1.dim().z != M2.dim().z) || (M1.dim().y != M2.dim().x)){
        std::stringstream ss;
        ss << "Cannot perform matrix multiplication between matrices of size " << M1.dim() << " and " << M2.dim();
        throw MatrixSizeException(ss.str());
    }

    using T = CommonMatrixType<MatrixType1, MatrixType2>;

    auto getMultipliedIndex = [](const MatrixType1& m1, const MatrixType2& m2, const dim3& idx) {
        dim3 m1_idx(idx.x, 0, idx.z);
        dim3 m2_idx(0, idx.y, idx.z);
                
        T sum{};
        for (size_t i = 0; i < m1.dim().y; ++i, ++m1_idx.y, ++m2_idx.x){
            sum += m1(m1_idx) * m2(m2_idx);
        }
        return sum;
    };

    return GeneralMatrixOperationResult<decltype(M1), decltype(M2),decltype(getMultipliedIndex)>
    (std::forward<MatrixType1>(M1), std::forward<MatrixType2>(M2), dim3(M1.dim().x, M2.dim().y, M1.dim().z), getMultipliedIndex);
}

template<typename MatrixType1, typename MatrixType2, typename = IsMatrixBase<MatrixType1, MatrixType2>>
auto convolve(MatrixType1&& M1, MatrixType2&& M2, dim2 stride){
    // We define convolutions only for 2D matrices
    assert(M1.dim().z == M2.dim().z == 1);

    const dim3 output_size(
        M1.dim().x - M2.dim().x + 1,
        M1.dim().y - M2.dim().y + 1,
        1
    );

    auto getConvolvedValue = [stride, output_size](const MatrixType1& m1, const MatrixType2& m2, const dim3& idx){
        dim3 view_idx(idx.x*stride.x, idx.y*stride.y, 0);
        CommonMatrixType<MatrixType1, MatrixType2> sum{};
        for (uint x = 0; x < m2.dim().x; x++){
            for (uint y = 0; y < m2.dim().y; y++){
                dim3 local_idx(x, y, 0);
                sum += m1(local_idx + view_idx) * m2(local_idx);
            }
        }
        return sum;
    };

    return GeneralMatrixOperationResult<decltype(M1), decltype(M2), decltype(getConvolvedValue)>
    (std::forward<MatrixType1>(M1), std::forward<MatrixType2>(M2), output_size, getConvolvedValue);
}

// ================================================================================================
// Matrix - Scalar math result
// ================================================================================================

template<typename MatrixType, typename ScalarType, class BinaryOp>
class ScalarOperationResult : public MatrixBase{
public:
    using MT = MatrixStorageType<MatrixType>;
    using type = typename std::common_type_t<typename std::remove_reference_t<MatrixType>::type, ScalarType>;

    ScalarOperationResult(MatrixType&& M, const ScalarType& S, BinaryOp op) :
    MatrixBase(M.dim()),
    m_(std::forward<MatrixType>(M)), s_(S), op_(op) 
    {}

    type operator()(dim3 idx) const {
        return std::invoke(op_, m_, s_, idx);
    }

    type operator()(uint x, uint y, uint z) const {
        return operator()(dim3(x,y,z));
    }

    auto begin() {return MatrixIterator<ScalarOperationResult<MatrixType, ScalarType, BinaryOp>&>(*this, {0, 0, 0});}
    auto end() {return MatrixIterator<ScalarOperationResult<MatrixType, ScalarType, BinaryOp>&>(*this, {0, 0, dim_.z});}
    auto begin() const {return MatrixIterator<const ScalarOperationResult<MatrixType, ScalarType, BinaryOp>&>(*this, {0, 0, 0});}
    auto end() const {return MatrixIterator<const ScalarOperationResult<MatrixType, ScalarType, BinaryOp>&>(*this, {0, 0, dim_.z});}

private:
    const MT m_;
    const ScalarType s_;
    BinaryOp op_;
};

#define ADD_MATRIX_SCALAR_OPERATOR(op)                                                          \
                                                                                                \
template<typename MatrixType, typename Scalar, typename = IsMatrixBase<MatrixType>>             \
auto operator op (MatrixType&& M1, const Scalar& S){                                            \
    auto comp = [](const MatrixType& m1, const Scalar& s, const dim3& idx) {                    \
        return std::forward<MatrixType>(m1)(idx) op s;                                          \
    };                                                                                          \
                                                                                                \
    return ScalarOperationResult<decltype(M1), Scalar, decltype(comp)>                          \
    (std::forward<MatrixType>(M1), S, comp);                                                    \
}                                                                                               \
template<typename MatrixType, typename Scalar, typename = IsMatrixBase<MatrixType>>             \
auto operator op (const Scalar& S, MatrixType&& M1){                                            \
    auto comp = [](const MatrixType& m1, const Scalar& s, const dim3& idx) {                    \
        return std::forward<MatrixType>(m1)(idx) op s;                                          \
    };                                                                                          \
                                                                                                \
    return ScalarOperationResult<decltype(M1), Scalar, decltype(comp)>                          \
    (std::forward<MatrixType>(M1), S, comp);                                                    \
}

ADD_MATRIX_SCALAR_OPERATOR(<)
ADD_MATRIX_SCALAR_OPERATOR(<=)
ADD_MATRIX_SCALAR_OPERATOR(>)
ADD_MATRIX_SCALAR_OPERATOR(>=)
ADD_MATRIX_SCALAR_OPERATOR(==)
ADD_MATRIX_SCALAR_OPERATOR(!=)
ADD_MATRIX_SCALAR_OPERATOR(+)
ADD_MATRIX_SCALAR_OPERATOR(-)
ADD_MATRIX_SCALAR_OPERATOR(*)
ADD_MATRIX_SCALAR_OPERATOR(/)

#undef ADD_MATRIX_SCALAR_OPERATOR

// ================================================================================================
// Matrix unary operation
// ================================================================================================

template<typename MatrixType, class UnaryOp>
class UnaryOperationResult : public MatrixBase{
public:

    template<typename T1>
    friend auto transpose(T1&&);

    using MT = MatrixStorageType<MatrixType>;
    using type = typename std::remove_reference_t<MatrixType>::type;

    UnaryOperationResult(MatrixType&& M, UnaryOp Op) :
    MatrixBase(M.dim()),
    m_(std::forward<MatrixType>(M)), op(Op) 
    {}

    type operator()(dim3 idx) const {
        return std::invoke(op, m_, idx);
    }

    type operator()(uint x, uint y, uint z) const {
        return operator()(dim3(x,y,z));
    }

    auto begin() {return MatrixIterator<UnaryOperationResult<MatrixType, UnaryOp>&>(*this, {0, 0, 0});}
    auto end() {return MatrixIterator<UnaryOperationResult<MatrixType, UnaryOp>&>(*this, {0, 0, dim_.z});}
    auto begin() const {return MatrixIterator<const UnaryOperationResult<MatrixType, UnaryOp>&>(*this, {0, 0, 0});}
    auto end() const {return MatrixIterator<const UnaryOperationResult<MatrixType, UnaryOp>&>(*this, {0, 0, dim_.z});}

private:
    const MT m_;
    UnaryOp op;
};

template<typename MatrixType, typename UnaryOp>
auto apply(MatrixType&& M, UnaryOp op){
    auto invoker = [op](const MatrixType& M_, const dim3& idx_){
        return std::invoke(op, M_(idx_));
    };

    return UnaryOperationResult<decltype(M), decltype(invoker)>(std::forward<MatrixType>(M), invoker);
}

template<typename MatrixType>
auto abs(MatrixType&& M){
    using input_type = typename MatrixType::type;
    using return_type = decltype(std::abs(input_type{}));
    return my_cnn::apply<decltype(M), return_type(*)(input_type)>(std::forward<MatrixType>(M), std::abs);
}

template<typename MatrixType>
auto exp(MatrixType&& M){
    using input_type = typename std::remove_reference_t<MatrixType>::type;
    using return_type = decltype(std::exp(input_type{}));
    return my_cnn::apply<decltype(M), return_type(*)(input_type)>(std::forward<MatrixType>(M), std::exp);
}

template<typename MatrixType>
auto transpose(MatrixType&& M){
    auto transposeIdx = [](const MatrixType& M, const dim3& idx){
        return M(dim3(idx.y, idx.x, idx.z));
    };
    auto&& ret_val = UnaryOperationResult<decltype(M), decltype(transposeIdx)>(std::forward<MatrixType>(M), transposeIdx);
    ret_val.dim_.x = M.dim().y;
    ret_val.dim_.y = M.dim().x;
    return ret_val;
}

template<size_t NumSteps, typename MatrixType, typename = IsMatrixBase<MatrixType>>
auto rotate(MatrixType&& M){
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

    return UnaryOperationResult<decltype(M), decltype(rotatedIndex)>(std::forward<MatrixType>(M), rotatedIndex);
}

// ================================================================================================
// Matrix reduction operations
// ================================================================================================

template<typename MatrixType>
auto sum(MatrixType&& M){
    using T = typename std::remove_reference_t<MatrixType>::type;
    return std::accumulate(std::forward<MatrixType>(M).begin(), std::forward<MatrixType>(M).end(), T());
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

template<typename MatrixType>
auto l2Norm(MatrixType&& M){
    return std::sqrt(my_cnn::sum(std::forward<MatrixType>(M)*std::forward<MatrixType>(M)));
}

// ================================================================================================
// 
// ================================================================================================

template<typename MatrixType1, typename MatrixType2>
bool matrixEqual(MatrixType1&& M1, MatrixType2&& M2){
    if (M1.dim() != M2.dim()) return false;

    decltype(auto) m1 = std::forward<MatrixType1>(M1);
    decltype(auto) m2 = std::forward<MatrixType2>(M2);

    bool equal = true;
    for (auto it = m1.begin(); equal && (it != m1.end()); it++){
        equal = equal && (*it == m2(it.idx()));
    }
    return equal;
}

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
