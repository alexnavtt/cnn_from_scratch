#include <string>
#include <type_traits>
#include "cnn_from_scratch/Matrix/SubMatrixView.h"
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

namespace my_cnn{

// Declare type for ease of use
template<typename MatrixType>
using type = typename MatrixType::type;

template<typename T>
const T& SubMatrixView<T>::operator()(uint x, uint y, uint z) const{
    return mat_ptr_->operator()(start_.x + x, start_.y + y, start_.z + z);
}

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
T& SubMatrixView<T>::operator()(uint x, uint y, uint z) {
    return mat_ptr_->operator()(start_.x + x, start_.y + y, start_.z + z);
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
SubMatrixView<T> SubMatrixView<T>::slices(int idx, int num){
    return SubMatrixView<T>(*this, {0, 0, idx}, {dim_.x, dim_.y, num});
}

template<typename T>
SubMatrixView<T> SubMatrixView<T>::slice(int idx){
    return slices(idx, 1);
}

template<typename T>
template<typename Other, std::enable_if_t<std::is_convertible_v<Other, T>,bool>>
SubMatrixView<T>& SubMatrixView<T>::operator=(const Other& o){
    std::fill(begin(), end(), o);
    return *this;
}

template<typename T>
template<typename Other, std::enable_if_t<std::is_base_of_v<MatrixBase, std::remove_reference_t<Other>>, bool>>
SubMatrixView<T>& SubMatrixView<T>::operator=(Other&& M){
    std::copy(std::forward<Other>(M).begin(), std::forward<Other>(M).end(), std::begin(*this));
    return *this;
}

} // namespace my_cnn
