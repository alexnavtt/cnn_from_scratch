#include <string>
#include <type_traits>
#include "cnn_from_scratch/SubMatrixView.h"
#include "cnn_from_scratch/SimpleMatrix.h"

namespace my_cnn{

template<typename T, typename MatrixType>
const T& SubMatrixView<T, MatrixType>::operator()(dim3 idx) const{
    return mat_ptr_->operator()(start_ + idx);
}

template<typename T, typename MatrixType>
const T& SubMatrixView<T, MatrixType>::at(dim3 idx) const{
    if (idx.x >= dim_.x || idx.y >= dim_.y || idx.z >= dim_.z){
        std::stringstream ss;
        ss << "Failed to index SubMatrixView of size " << dim_ << " with index " << idx;
        throw std::out_of_range(ss.str());
    }
    return this->operator[](idx);
}

// template<typename T, typename MatrixType>
// template<std::enable_if_t<not std::is_const_v<MatrixType>, bool>>
// T& SubMatrixView<T, MatrixType>::operator()(dim3 idx){
//     return const_cast<T&>(const_cast<const SubMatrixView<T>&>(*this).operator()(idx)); 
// }

// template<typename T, typename MatrixType>
// template<std::enable_if_t<not std::is_const_v<MatrixType>, bool>>
// T& SubMatrixView<T, MatrixType>::at(dim3 idx){
//     return const_cast<T&>(const_cast<const SubMatrixView<T>&>(*this).at(idx)); 
// }

template<typename T, typename MatrixType>
SubMatrixView<T, MatrixType>::operator SimpleMatrix<T>() const{
    SimpleMatrix<T> mat(this->dim_);
    std::copy(begin(), end(), std::begin(mat));
    return mat;
}

template<typename T, typename MatrixType>
template<typename Other, std::enable_if_t<std::is_convertible_v<Other, T> && not std::is_const_v<MatrixType>, bool>>
SubMatrixView<T, MatrixType>& SubMatrixView<T, MatrixType>::operator=(const Other& o){
    std::fill(begin(), end(), o);
    return *this;
}

template<typename T, typename MatrixType>
template<typename Other, std::enable_if_t<std::is_convertible_v<Other, SimpleMatrix<T>> && not std::is_const_v<MatrixType>, bool>>
SubMatrixView<T, MatrixType>& SubMatrixView<T, MatrixType>::operator=(const Other& o){
    std::copy(std::begin(o), std::end(o), std::begin(*this));
    return *this;
}

#define DEFINE_MODIFYING_OPERATOR(op)                                                               \
                                                                                                    \
template<typename T, typename MatrixType>                                                                                \
template<typename Other, std::enable_if_t<std::is_convertible_v<Other, T> && not std::is_const_v<MatrixType>, bool>>                   \
SubMatrixView<T, MatrixType> SubMatrixView<T, MatrixType>::operator op(const Other& o){                                     \
    std::for_each(begin(), end(), [&](T& val){val op o;});                                          \
    return *this;                                                                                   \
}                                                                                                   \
                                                                                                    \
template<typename T, typename MatrixType>                                                                                \
template<typename Other, std::enable_if_t<std::is_convertible_v<Other, T> && not std::is_const_v<MatrixType>, bool>>                   \
SubMatrixView<T, MatrixType> SubMatrixView<T, MatrixType>::operator op(const SimpleMatrix<Other>& o){                       \
    assert(dim_ == o.dim_);                                                                         \
    auto it = begin();                                                                              \
    auto my_end = end();                                                                            \
    for (auto other_it = std::begin(o); it != my_end;){                                             \
        *it++ op *other_it++;                                                                       \
    }                                                                                               \
    return *this;                                                                                   \
}                                                                                                   \
                                                                                                    \
template<typename T, typename MatrixType>                                                                                \
template<typename Other, std::enable_if_t<std::is_convertible_v<Other, T> && not std::is_const_v<MatrixType>, bool>>                   \
SubMatrixView<T, MatrixType> SubMatrixView<T, MatrixType>::operator op(const SubMatrixView<Other>& o){                      \
    assert(dim_ == o.dim_);                                                                         \
    SubMatrixIterator<T, MatrixType> it = begin();                                                                              \
    auto my_end = end();                                                                            \
                                                                                                    \
    /* Check if we're copying from the range we're writing to */                                    \
    if (o.mat_ptr_ == mat_ptr_){                                                                    \
        /* If they do overlap, copy the data first */                                               \
        return (*this) op static_cast<SimpleMatrix<Other>>(o);                                      \
    }                                                                                               \
    /* If they don't overlap, just add as usual */                                                  \
    else{                                                                                           \
        for (auto other_it = std::begin(o); it != my_end;){                                         \
            *it++ op *other_it++;                                                                   \
        }                                                                                           \
    }                                                                                               \
    return *this;                                                                                   \
}

DEFINE_MODIFYING_OPERATOR(+=);
DEFINE_MODIFYING_OPERATOR(-=);
DEFINE_MODIFYING_OPERATOR(*=);
DEFINE_MODIFYING_OPERATOR(/=);

#define DEFINE_OUTPUT_OPERATOR(op)                                                                 \
                                                                                                   \
template<typename T, typename MatrixType>                                                          \
template<typename Other, std::enable_if_t<std::is_convertible_v<Other, T>, bool>>                  \
SimpleMatrix<typename std::common_type_t<T, Other>>                                                \
SubMatrixView<T, MatrixType>::operator op(const Other& o) const {                                  \
    using U = typename std::common_type_t<T, Other>;                                               \
    SimpleMatrix<U> out = *this;                                                                   \
    out op##= o;                                                                                   \
    return out;                                                                                    \
}                                                                                                  \
                                                                                                   \
template<typename T, typename MatrixType>                                                          \
template<typename Other, std::enable_if_t<std::is_convertible_v<Other, SimpleMatrix<T>>, bool>>    \
SimpleMatrix<typename std::common_type_t<T, Other>>                                                \
SubMatrixView<T, MatrixType>::operator op(const Other& o) const {                                  \
    using U = typename std::common_type_t<T, Other>;                                               \
    SimpleMatrix<U> out(dim_);                                                                     \
    auto it1 = std::begin(*this);                                                                  \
    auto it2 = std::begin(o);                                                                      \
    auto it3 = std::begin(out);                                                                    \
    for (; it1 != std::end(*this); it1++, it2++, it3++){                                           \
        *it3 = *it1 op *it2;                                                                       \
    }                                                                                              \
}

DEFINE_OUTPUT_OPERATOR(+);
DEFINE_OUTPUT_OPERATOR(-);
DEFINE_OUTPUT_OPERATOR(*);
DEFINE_OUTPUT_OPERATOR(/);

} // namespace my_cnn
