#include <string>
#include <type_traits>
#include "cnn_from_scratch/Matrix/SubMatrixView.h"
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

namespace my_cnn{

// Declare type for ease of use
template<typename MatrixType>
using type = typename MatrixType::type;

template<typename MatrixType>
const type<MatrixType>& SubMatrixView<MatrixType>::operator()(dim3 idx) const{
    return mat_ptr_->operator()(start_ + idx);
}

template<typename MatrixType>
const type<MatrixType>& SubMatrixView<MatrixType>::at(dim3 idx) const{
    if (idx.x >= dim_.x || idx.y >= dim_.y || idx.z >= dim_.z){
        std::stringstream ss;
        ss << "Failed to index SubMatrixView of size " << dim_ << " with index " << idx;
        throw std::out_of_range(ss.str());
    }
    return this->operator[](idx);
}

template<typename MatrixType>
SubMatrixView<MatrixType>::operator SimpleMatrix<SubMatrixView<MatrixType>::type>() const{
    SimpleMatrix<SubMatrixView<MatrixType>::type> mat(this->dim_);
    std::copy(begin(), end(), std::begin(mat));
    return mat;
}

template<typename MatrixType>
SimpleMatrix<type<MatrixType>> SubMatrixView<MatrixType>::matrix() const{
    return *this;
}

template<typename MatrixType>
template<typename Other, 
    std::enable_if_t<
        std::is_convertible_v<Other, type<MatrixType>> 
        && not std::is_const_v<MatrixType>,
    bool>>
SubMatrixView<MatrixType>& SubMatrixView<MatrixType>::operator=(const Other& o){
    std::fill(begin(), end(), o);
    return *this;
}

template<typename MatrixType>
template<typename Other, 
    std::enable_if_t<
        std::is_convertible_v<Other, SimpleMatrix<type<MatrixType>>> 
        && not std::is_const_v<MatrixType>, 
    bool>>
SubMatrixView<MatrixType>& SubMatrixView<MatrixType>::operator=(const Other& o){
    std::copy(std::begin(o), std::end(o), std::begin(*this));
    return *this;
}

#define DEFINE_MODIFYING_OPERATOR(op)                                                               \
                                                                                                    \
template<typename MatrixType>                                                                            \
template<typename Other, std::enable_if_t<std::is_convertible_v<Other, type<MatrixType>> && not std::is_const_v<MatrixType>, bool>>                   \
SubMatrixView<MatrixType>& SubMatrixView<MatrixType>::operator op(const Other& o){                                     \
    std::for_each(begin(), end(), [&](type& val){val op o;});                                          \
    return *this;                                                                                   \
}                                                                                                   \
                                                                                                    \
template<typename MatrixType>                                                                                \
template<typename Other, std::enable_if_t<std::is_convertible_v<Other, type<MatrixType>> && not std::is_const_v<MatrixType>, bool>>                   \
SubMatrixView<MatrixType>& SubMatrixView<MatrixType>::operator op(const SimpleMatrix<Other>& o){                       \
    assert(dim_ == o.dim_);                                                                         \
    auto it = begin();                                                                              \
    auto my_end = end();                                                                            \
    for (auto other_it = std::begin(o); it != my_end;){                                             \
        *it++ op *other_it++;                                                                       \
    }                                                                                               \
    return *this;                                                                                   \
}                                                                                                   \
                                                                                                    \
template<typename MatrixType>                                                                                \
template<typename Other, std::enable_if_t<std::is_convertible_v<Other, type<MatrixType>> && not std::is_const_v<MatrixType>, bool>>                   \
SubMatrixView<MatrixType>& SubMatrixView<MatrixType>::operator op(const SubMatrixView<Other>& o){                      \
    assert(dim_ == o.dim_);                                                                         \
    SubMatrixIterator<MatrixType> it = begin();                                                                              \
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

#define DEFINE_OUTPUT_OPERATOR(op)                                                                               \
                                                                                                                 \
template<typename MatrixType>                                                                                    \
template<typename Other, std::enable_if_t<std::is_convertible_v<Other, type<MatrixType>>, bool>>                 \
SimpleMatrix<typename std::common_type_t<type<MatrixType>, Other>>                                               \
SubMatrixView<MatrixType>::operator op(const Other& o) const {                                                   \
    using U = typename std::common_type_t<type, Other>;                                                          \
    SimpleMatrix<U> out = *this;                                                                                 \
    out op##= o;                                                                                                 \
    return out;                                                                                                  \
}                                                                                                                \
                                                                                                                 \
template<typename MatrixType>                                                                                    \
template<typename Other, std::enable_if_t<std::is_convertible_v<Other, SimpleMatrix<type<MatrixType>>>, bool>>   \
SimpleMatrix<typename std::common_type_t<type<MatrixType>, Other>>                                               \
SubMatrixView<MatrixType>::operator op(const Other& o) const {                                                   \
    using U = typename std::common_type_t<type, Other>;                                                          \
    SimpleMatrix<U> out(dim_);                                                                                   \
    auto it1 = std::begin(*this);                                                                                \
    auto it2 = std::begin(o);                                                                                    \
    auto it3 = std::begin(out);                                                                                  \
    for (; it1 != std::end(*this); it1++, it2++, it3++){                                                         \
        *it3 = *it1 op *it2;                                                                                     \
    }                                                                                                            \
}

DEFINE_OUTPUT_OPERATOR(+);
DEFINE_OUTPUT_OPERATOR(-);
DEFINE_OUTPUT_OPERATOR(*);
DEFINE_OUTPUT_OPERATOR(/);

} // namespace my_cnn
