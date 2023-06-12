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
MatrixBase(dim),
values_(dim.x*dim.y*dim.z, initial_val)
{}

// From a Matrix-like object
template<typename T>
template<typename MatrixType, std::enable_if_t<std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType>>, bool>>
SimpleMatrix<T>::SimpleMatrix(MatrixType&& M) : 
MatrixBase(M.dim()),
values_(M.size())
{
    std::copy(std::forward<MatrixType>(M).begin(), std::forward<MatrixType>(M).end(), begin());
}

// Full matrix description constructor
template<typename T>
SimpleMatrix<T>::SimpleMatrix(dim3 dim, std::vector<T>&& vals):
MatrixBase(dim),
values_(size())
{
    setEntries(std::forward<std::vector<T>>(vals));
}

// ================================== //
// =========== ASSIGNMENT =========== //
// ================================== //

// Vector setting
template<typename T>
SimpleMatrix<T>& SimpleMatrix<T>::operator=(std::vector<T>&& v){
    if (v.size() == values_.size())
        values_ = std::forward<std::vector<T>>(v);
    else
        throw MatrixSizeException("Cannot assign value to matrix, size mismatch");
    return *this;
}

// Type conversion
template<typename T>
template<typename Other>
SimpleMatrix<T>& SimpleMatrix<T>::operator=(const SimpleMatrix<Other>& M){
    dim_ = M.dim_;
    values_.resize(dim_.x*dim_.y*dim_.z);
    std::copy(M.values_.begin(), M.values_.end(), values_.begin());
    return *this;
}

// Set entries via array
template<typename T>
void SimpleMatrix<T>::setEntries(std::vector<T>&& v){
    if (v.size() == values_.size()){
        auto val= v.begin();
        for (uint layer = 0; layer < dim_.z; layer++){
            for (uint row = 0; row < dim_.x; row++){
                for (uint col = 0; col < dim_.y; col++){
                    (*this)(row, col, layer) = *val++;
                }
            }
        }
    }
    else
        throw MatrixSizeException("Cannot assign value to matrix, size mismatch");
}

// Assign to matrix like
template<typename T>
template<typename MatrixType, std::enable_if_t<std::is_base_of_v<MatrixBase, std::remove_reference_t<MatrixType>>, bool>>
SimpleMatrix<T>& SimpleMatrix<T>::operator=(MatrixType&& M){
    dim_ = M.dim();
    values_.resize(size());
    std::copy(std::forward<MatrixType>(M).begin(), std::forward<MatrixType>(M).end(), begin());
    return *this;
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
void SimpleMatrix<T>::resize(int x, int y, int z){
    dim_ = dim3(x, y, z);
    values_.resize(x*y*z, T{});
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
    return values_[getIndex(x_idx, y_idx, z_idx)];
}

template<typename T>
T& SimpleMatrix<T>::operator()(size_t x_idx, size_t y_idx, size_t z_idx) {
    return values_[getIndex(x_idx, y_idx, z_idx)];
}

template<typename T>
const T& SimpleMatrix<T>::operator()(dim3 idx) const {
    return values_[getIndex(idx.x, idx.y, idx.z)];
}

template<typename T>
T& SimpleMatrix<T>::operator()(dim3 idx) {
    return values_[getIndex(idx.x, idx.y, idx.z)];
}
    
template<typename T>
SimpleMatrix<T> SimpleMatrix<T>::subMatCopy(dim3 idx, dim3 sub_dim) const{
    return static_cast<SimpleMatrix<T>>(subMatView(idx, sub_dim));
}

template<typename T>
SubMatrixView<T> SimpleMatrix<T>::subMatView(dim3 idx, dim3 sub_dim){
    return SubMatrixView<T>(*this, idx, sub_dim);
}

template<typename T>
SubMatrixView<const T> SimpleMatrix<T>::subMatView(dim3 idx, dim3 sub_dim) const{
    return SubMatrixView<const T>(*this, idx, sub_dim);
}

template<typename T>
SubMatrixView<T> SimpleMatrix<T>::slices(int idx, int num){
    return SubMatrixView<T>(*this, {0, 0, idx}, {dim_.x, dim_.y, num});
}

template<typename T>
SubMatrixView<T> SimpleMatrix<T>::slice(int idx) {
    return slices(idx, 1);
}

template<typename T>
SubMatrixView<const T> SimpleMatrix<T>::slices(int idx, int num) const {
    return SubMatrixView<const T>(*this, {0, 0, idx}, {dim_.x, dim_.y, num});
}

template<typename T>
SubMatrixView<const T> SimpleMatrix<T>::slice(int idx) const {
    return slices(idx, 1);
}

// =================================== //
// ======== ELEMENT WISE MATH ======== //
// =================================== //

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
// ========== SERIALIZATION ========== //
// =================================== //

template<typename T>
void SimpleMatrix<T>::serialize(std::ostream& os) const {
    os << std::fixed;
    os << dim_.x << " " << dim_.y << " " << dim_.z << "\n";
    for (int z = 0; z < dim_.z; z++){
        for (int x = 0; x < dim_.x; x++){
            for (int y = 0; y < dim_.y; y++){
                const T& val = this->operator()(x, y, z);
                os << std::setw(16) << val << " ";
            }
            os << "\n";
        }
        os << "\n";
    }
    os << std::defaultfloat;
}

template<typename T>
bool SimpleMatrix<T>::deserialize(std::istream& is) {
    // Retreive the dimensions of the matrix
    dim3 dim;
    is >> dim.x;
    is >> dim.y;
    is >> dim.z;
    std::cout << "Matrix dimension was read as " << dim << "\n";

    if (is.fail()) return false;
    std::cout << "Success\n";
    this->resize(dim);

    // Retrieve the rows of the matrix
    for (int z = 0; z < dim.z; z++){
        for (int x = 0; x < dim.x; x++){
            // For each entry, write to the corresponding value
            for (int y = 0; y < dim.y; y++){
                is >> this->operator()(x, y, z);
                if (is.fail()) return false;
            }

            // Newline
            is.ignore(std::numeric_limits<int>::max(), '\n');
        }

        // Newline
        is.ignore(std::numeric_limits<int>::max(), '\n');
    }

    return true;
}

} // namespace my_cnn

