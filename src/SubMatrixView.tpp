#include <string>
#include <type_traits>
#include "cnn_from_scratch/Matrix/SubMatrixView.h"
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

namespace my_cnn{

// Declare type for ease of use
template<typename MatrixType>
using type = typename MatrixType::type;

template<typename T>
const T& SubMatrixView<T>::operator()(const dim3& idx) const{
    return mat_ptr_->operator()(start_ + idx);
}

template<typename T>
const T& SubMatrixView<T>::at(const dim3& idx) const{
    if (idx.x >= dim_.x || idx.y >= dim_.y || idx.z >= dim_.z){
        std::stringstream ss;
        ss << "Failed to index SubMatrixView of size " << dim_ << " with index " << idx;
        throw std::out_of_range(ss.str());
    }
    return this->operator()(idx);
}

template<typename T>
T& SubMatrixView<T>::operator()(const dim3& idx) {
    return mat_ptr_->operator()(start_ + idx);
}

template<typename T>
T& SubMatrixView<T>::at(const dim3& idx) {
    if (idx.x >= dim_.x || idx.y >= dim_.y || idx.z >= dim_.z){
        std::stringstream ss;
        ss << "Failed to index SubMatrixView of size " << dim_ << " with index " << idx;
        throw std::out_of_range(ss.str());
    }
    return this->operator()(idx);
}

template<typename T>
SubMatrixView<T>::operator SimpleMatrix<typename std::remove_const_t<T>>() const{
    SimpleMatrix<typename std::remove_const_t<T>> mat(this->dim_);
    std::copy(begin(), end(), std::begin(mat));
    return mat;
}

template<typename T>
SimpleMatrix<typename std::remove_const_t<T>> SubMatrixView<T>::matrix() const{
    return *this;
}

template<typename T>
template<typename Other, std::enable_if_t<std::is_convertible_v<Other, T>,bool>>
SubMatrixView<T>& SubMatrixView<T>::operator=(const Other& o){
    std::fill(begin(), end(), o);
    return *this;
}

template<typename T>
template<typename Other, std::enable_if_t<std::is_base_of_v<MatrixBase, Other>, bool>>
SubMatrixView<T>& SubMatrixView<T>::operator=(const Other& o){
    std::copy(std::begin(o), std::end(o), std::begin(*this));
    return *this;
}

} // namespace my_cnn
