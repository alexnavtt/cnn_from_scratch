#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

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

// ================================== //
// ========== CONSTRUCTORS ========== //
// ================================== //

// Initial value based constructor
template<typename T>
SimpleMatrix<T>::SimpleMatrix(dim3 dim, T initial_val):
std::valarray<T>(initial_val, dim.x*dim.y*dim.z),
MatrixBase(dim)
{}

// From a Matrix-like object
template<typename T>
template<typename MatrixType, std::enable_if_t<std::is_base_of_v<MatrixBase, MatrixType>, bool>>
SimpleMatrix<T>::SimpleMatrix(const MatrixType& M) : 
std::valarray<T>(M.dim().x*M.dim().y*M.dim().z), 
MatrixBase(M.dim())
{
    for (DimIterator<3> idx(dim_, {0, 0, 0}); idx.idx.z < dim_.z; idx++){
        (*this)(idx.idx) = M(idx.idx);
    }
}

// Full matrix description constructor
template<typename T>
SimpleMatrix<T>::SimpleMatrix(dim3 dim, std::valarray<T>&& vals):
std::valarray<T>(dim.x*dim.y*dim.z),
MatrixBase(dim)
{
    setEntries(std::forward<std::valarray<T>>(vals));
}

// From a gslice_array
template<typename T>
SimpleMatrix<T>::SimpleMatrix(dim3 dim, std::gslice_array<T>&& vals):
std::valarray<T>(vals),
MatrixBase(dim)
{}

// Type conversion constructor
template<typename T>
template<typename Other, std::enable_if_t<not std::is_same_v<T, Other>, bool>>
SimpleMatrix<T>::SimpleMatrix(const SimpleMatrix<Other>& M) : 
std::valarray<T>(M.dim_.x*M.dim_.y*M.dim_.z),
MatrixBase(M.dim_)
{
    std::copy(std::begin(M), std::end(M), std::begin(*this));
}

// Copy constructor
template<typename T>
SimpleMatrix<T>::SimpleMatrix(const SimpleMatrix<T>& M) : 
std::valarray<T>(M), 
MatrixBase(M.dim_) 
{}

// Move copy constructor
template<typename T>
SimpleMatrix<T>::SimpleMatrix(SimpleMatrix<T>&& M) : 
std::valarray<T>(std::forward<std::valarray<T>>(M)), 
MatrixBase(M.dim_) 
{}

// ================================== //
// =========== ASSIGNMENT =========== //
// ================================== //

// Valarray setting
template<typename T>
SimpleMatrix<T>& SimpleMatrix<T>::operator=(std::valarray<T>&& v){
    if (v.size() == this->size())
        static_cast<std::valarray<T>&>(*this) = std::forward<std::valarray<T>>(v);
    else
        throw MatrixSizeException("Cannot assign value to matrix, size mismatch");
    return *this;
}

// Gslice setting
template<typename T>
SimpleMatrix<T>& SimpleMatrix<T>::operator=(std::gslice_array<T>&& v){
    std::valarray<T> arr(v);
    if (arr.size() == this->size())
        static_cast<std::valarray<T>&>(*this) = arr;
    else
        throw MatrixSizeException("Cannot assign value to matrix, size mismatch");
    return *this;
}

// Type conversion
template<typename T>
template<typename Other>
SimpleMatrix<T>& SimpleMatrix<T>::operator=(const SimpleMatrix<Other>& M){
    dim_ = M.dim_;
    std::valarray<T>::resize(dim_.x*dim_.y*dim_.z);
    std::copy(std::begin(M), std::end(M), std::begin(*this));
    return *this;
}

// Set entries via array
template<typename T>
void SimpleMatrix<T>::setEntries(std::valarray<T>&& v){
    if (v.size() == this->size())
        static_cast<std::valarray<T>&>(*this) = v[
            std::gslice(
                0, 
                {dim_.z, dim_.y, dim_.x},
                {dim_.x*dim_.y, 1, dim_.y}
            )
        ];
    else
        throw MatrixSizeException("Cannot assign value to matrix, size mismatch");
}

// ================================== //
// =========== DIMENSIONS =========== //
// ================================== //

template<typename T>
uint SimpleMatrix<T>::dim(size_t idx) const{
    return dim_.data[idx];
}

template<typename T>
const dim3& SimpleMatrix<T>::dims() const{
    return dim_;
}

template<typename T>
void SimpleMatrix<T>::reshape(int x, int y, int z){
    if(x*y*z != dim_.x*dim_.y*dim_.z){
        std::stringstream ss;
        ss << "Cannot transform matrix from size " << dim_ << " to size " << dim3(x,y,z) << "\n";
        throw MatrixTransformException(ss.str());
    }
    dim_ = dim3(x, y, z);
}

template<typename T>
SimpleMatrix<T> SimpleMatrix<T>::transpose(){
    dim3 new_dim(dim_.y, dim_.x, dim_.z);
    SimpleMatrix<T> output(new_dim);
    static_cast<std::valarray<T>&>(output) = (*this)[
        std::gslice(
            0, 
            {dim_.z, dim_.x, dim_.y},
            {dim_.x*dim_.y, 1, dim_.x}
        )
    ];
    return output;
}

// ================================== //
// ============ INDEXING ============ //
// ================================== //

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
    // return data_[getIndex(x_idx, y_idx, z_idx)];
    return std::valarray<T>::operator[](getIndex(x_idx, y_idx, z_idx));
}

template<typename T>
T& SimpleMatrix<T>::operator()(size_t x_idx, size_t y_idx, size_t z_idx) {
    // return data_[getIndex(x_idx, y_idx, z_idx)];
    return std::valarray<T>::operator[](getIndex(x_idx, y_idx, z_idx));
}

template<typename T>
const T& SimpleMatrix<T>::operator()(dim3 idx) const {
    // return data_[getIndex(idx.x, idx.y, idx.z)];
    return std::valarray<T>::operator[](getIndex(idx.x, idx.y, idx.z));
}

template<typename T>
T& SimpleMatrix<T>::operator()(dim3 idx) {
    // return data_[getIndex(idx.x, idx.y, idx.z)];
    return std::valarray<T>::operator[](getIndex(idx.x, idx.y, idx.z));
}
    
template<typename T>
SimpleMatrix<T> SimpleMatrix<T>::subMatCopy(dim3 idx, dim3 sub_dim) const{
    // Verify that this is a valid submatrix
    if (idx.x-1 + sub_dim.x >= dim_.x || 
        idx.y-1 + sub_dim.y >= dim_.y || 
        idx.z-1 + sub_dim.z >= dim_.z)
        THROW_SUB_SIZE_EXCEPTION;

    SimpleMatrix<T> sub_mat(sub_dim);
    static_cast<std::valarray<T>&>(sub_mat) = this->operator[](
        std::gslice(
            getIndex(idx.x, idx.y, idx.z), 
            {sub_dim.z, sub_dim.y, sub_dim.x}, 
            {dim_.y*dim_.x, dim_.x, 1}
        )
    );
    return sub_mat;
}

template<typename T>
SubMatrixView<SimpleMatrix<T>> SimpleMatrix<T>::subMatView(dim3 idx, dim3 sub_dim){
    return SubMatrixView(*this, idx, sub_dim);
}

template<typename T>
SubMatrixView<const SimpleMatrix<T>> SimpleMatrix<T>::subMatView(dim3 idx, dim3 sub_dim) const{
    return SubMatrixView(*this, idx, sub_dim);
}

template<typename T>
SubMatrixView<SimpleMatrix<T>> SimpleMatrix<T>::slices(int idx, int num){
    return SubMatrixView(*this, {0, 0, idx}, {dim_.x, dim_.y, num});
}

template<typename T>
SubMatrixView<SimpleMatrix<T>> SimpleMatrix<T>::slice(int idx) {
    return slices(idx, 1);
}

template<typename T>
SubMatrixView<const SimpleMatrix<T>> SimpleMatrix<T>::slices(int idx, int num) const{
    return SubMatrixView(*this, {0, 0, idx}, {dim_.x, dim_.y, num});
}

template<typename T>
SubMatrixView<const SimpleMatrix<T>> SimpleMatrix<T>::slice(int idx) const {
    return slices(idx, 1);
}

// =================================== //
// ======== ELEMENT WISE MATH ======== //
// =================================== //

template<typename T>
bool SimpleMatrix<T>::operator ==(const SimpleMatrix<T>& other) const{
    if (dim_ != other.dim_) return false;
    for (size_t i = 0; i < this->size(); i++){
        if (this->operator[](i) != other[i]) return false;
    }
    return true;
}

template<typename T>
SimpleMatrix<T> SimpleMatrix<T>::abs() const {
    if constexpr (std::is_unsigned_v<T>)
        return *this;
    else{
        SimpleMatrix<T> out(dim_);
        return out = std::abs(*this);
    }
}

template<typename T>
size_t SimpleMatrix<T>::minIndex() const{
    return std::distance(std::begin(*this), std::min_element(std::begin(*this), std::end(*this)));
}

template<typename T>
size_t SimpleMatrix<T>::maxIndex() const{
    return std::distance(std::begin(*this), std::max_element(std::begin(*this), std::end(*this)));
}

// =================================== //
// =========== MATRIX MATH =========== //
// =================================== //

template<typename T>
template<typename Other, std::enable_if_t<std::is_base_of_v<MatrixBase, Other>, bool>>
typename std::common_type_t<T, typename Other::type> SimpleMatrix<T>::dot(const Other& M) const{
    if (this->size() != M.size()){
        throw MatrixSizeException("Cannot get the dot product A.B, size mismatch. A has " 
            + std::to_string(this->size()) + " elements and B has " + std::to_string(M.size()) + " elements");
    }

    if constexpr (std::is_same_v<Other, SimpleMatrix<T>>){
        return (static_cast<const std::valarray<T>&>(*this) * static_cast<const std::valarray<T>&>(M)).sum();
    }else{
        std::common_type_t<T, typename Other::type> sum{};
        auto this_it = std::begin(*this);
        auto other_it = std::begin(M);
        for (; this_it != std::end(*this); this_it++, other_it++){
            sum += *this_it * *other_it;
        }
        return sum;
    }
}

template<typename T>
template<typename Other>
SimpleMatrix<typename std::common_type<T, Other>::type> SimpleMatrix<T>::matMul(const SimpleMatrix<Other> M) const{
    if (dim_.z != M.dim_.z){
        std::cout << "Matrix multiply error: Differing number of layers. ";
        std::cout << "This has " << dim_.z << " layers and the other has " << M.dim_.z << "\n";
        throw MatrixSizeException("Multiplication layer mismatch");
    }else if(dim_.y != M.dim_.x){
        std::cout << "Matrix multiply error: Incompatible matrix dimensions ";
        std::cout << dim_ << " and " << M.dim_ << "\n";
        throw MatrixSizeException("Multiplication dimension mismatch");
    }

    SimpleMatrix<typename std::common_type<T, Other>::type> out({dim_.x, M.dim_.y, dim_.z});
    for (uint row = 0; row < out.dim_.x; row++){
        for (uint col = 0; col < out.dim_.y; col++){
            for (uint layer = 0; layer < dim_.z; layer++){

                out(row, col, layer) = 
                    this->subMatCopy({row, 0, layer}, {1, dim_.y, 1})
                        .dot(M.subMatView({0, col, layer}, {M.dim_.x, 1, 1}));

            }
        }
    }

    return out;
}

} // namespace my_cnn

