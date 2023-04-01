#include <cstring>
#include "cnn_from_scratch/SimpleMatrix.h"

#define THROW_SUB_SIZE_EXCEPTION { \
    std::stringstream ss;      \
    ss << "subMatrix generation size mismatch. Max indices of (" \
       << idx.x-1 + sub_dim.x << ", " << idx.y-1 + sub_dim.y << ", " << idx.z-1 + sub_dim.z \
       << ") would exceed initial matrix size which is (" \
       << dim_.x << ", " << dim_.y << ", " << dim_.z \
       << ")\n";    \
       throw MatrixSizeException(ss.str()); \
}

namespace my_cnn{

template<typename T>
size_t SimpleMatrix<T>::getIndex(size_t x_idx, size_t y_idx, size_t z_idx) const{
    return z_idx * dim_.y * dim_.x + y_idx * dim_.x + x_idx;
}

template<typename T>
size_t SimpleMatrix<T>::getIndex(dim3 dim) const{
    return getIndex(dim.x, dim.y, dim.z);
}

template<typename T>
const T& SimpleMatrix<T>::operator()(size_t x_idx, size_t y_idx, size_t z_idx) const {
    return data_[getIndex(x_idx, y_idx, z_idx)];
}

template<typename T>
T& SimpleMatrix<T>::operator()(size_t x_idx, size_t y_idx, size_t z_idx) {
    return data_[getIndex(x_idx, y_idx, z_idx)];
}

template <typename T> 
template<typename Other>
SimpleMatrix<T>& SimpleMatrix<T>::operator=(const SimpleMatrix<Other>& M){
    dim_ = M.dim_;
    data_.resize(dim_.x*dim_.y*dim_.z);
    if constexpr (std::is_same_v<T, Other>){
        data_ = M.data_;
    }else{
        for (size_t i = 0; i < data_.size(); i++){
            data_[i] = static_cast<T>(M.data_[i]);
        }
    }
    return *this;
}
    
template<typename T>
SimpleMatrix<T> SimpleMatrix<T>::subMat(dim3 idx, dim3 sub_dim) const{
    // Verify that this is a valid submatrix
    if (idx.x-1 + sub_dim.x >= dim_.x || 
        idx.y-1 + sub_dim.y >= dim_.y || 
        idx.z-1 + sub_dim.z >= dim_.z)
        THROW_SUB_SIZE_EXCEPTION;

    SimpleMatrix<T> sub_mat(sub_dim);
    sub_mat.data_ = data_[
        std::gslice(
            getIndex(idx.x, idx.y, idx.z), 
            {sub_dim.z, sub_dim.y, sub_dim.x}, 
            {dim_.y*dim_.x, dim_.x, 1}
        )
    ];
    return sub_mat;
}

template<typename T>
SubMatrixView<T> SimpleMatrix<T>::subMatView(dim3 idx, dim3 sub_dim){
    // Verify that this is a valid submatrix
    if (idx.x-1 + sub_dim.x >= dim_.x || 
        idx.y-1 + sub_dim.y >= dim_.y || 
        idx.z-1 + sub_dim.z >= dim_.z)
        THROW_SUB_SIZE_EXCEPTION;

    return SubMatrixView<T>(data_, 
        std::gslice(
            getIndex(idx.x, idx.y, idx.z), 
            {sub_dim.z, sub_dim.y, sub_dim.x}, 
            {dim_.y*dim_.x, dim_.x, 1}
        )
    );
}

template<typename T> 
void SimpleMatrix<T>::conditionallySet(T val, Comparison pred, T other){
    switch (pred){
        case LESS:          data_[data_ <  other] = val; return;
        case LESS_EQUAL :   data_[data_ <= other] = val; return;
        case GREATER:       data_[data_ >  other] = val; return;
        case GREATER_EQUAL: data_[data_ >= other] = val; return;
        case EQUAL:         data_[data_ == other] = val; return;
        case NOT_EQUAL:     data_[data_ != other] = val; return;
    }
}

} // namespace my_cnn

template class my_cnn::SimpleMatrix<int>;
template class my_cnn::SimpleMatrix<unsigned int>;
template class my_cnn::SimpleMatrix<char>;
template class my_cnn::SimpleMatrix<unsigned char>;
template class my_cnn::SimpleMatrix<short>;
template class my_cnn::SimpleMatrix<unsigned short>;
template class my_cnn::SimpleMatrix<long>;
template class my_cnn::SimpleMatrix<unsigned long>;
template class my_cnn::SimpleMatrix<long long>;
template class my_cnn::SimpleMatrix<unsigned long long>;
template class my_cnn::SimpleMatrix<float>;
template class my_cnn::SimpleMatrix<double>;
