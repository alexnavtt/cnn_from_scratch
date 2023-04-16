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
template<typename Other, std::enable_if_t<std::is_convertible_v<Other, type<MatrixType>>,bool>>
SubMatrixView<MatrixType>& SubMatrixView<MatrixType>::operator=(const Other& o){
    std::fill(begin(), end(), o);
    return *this;
}

template<typename MatrixType>
template<typename Other, std::enable_if_t<std::is_base_of_v<MatrixBase, Other>, bool>>
SubMatrixView<MatrixType>& SubMatrixView<MatrixType>::operator=(const Other& o){
    std::copy(std::begin(o), std::end(o), std::begin(*this));
    return *this;
}

} // namespace my_cnn
