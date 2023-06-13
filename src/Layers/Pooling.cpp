#include "cnn_from_scratch/Layers/Pooling.h"

namespace my_cnn{
    
Pooling::Pooling(dim2 dim, dim2 stride, PoolingType type) :
    dim_(dim), 
    stride_(stride), 
    type_(type)
{

}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

ModelFlowMode Pooling::getType() const {
    return POOLING;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

bool Pooling::checkSize(const SimpleMatrix<double>& input)  {
    return input.dim(0) >= dim_.x && input.dim(1) >= dim_.y;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

dim3 Pooling::outputSize(const dim3& input_dim) const {
    const int x_size = std::ceil((input_dim.x - dim_.x + 1.0f)/stride_.x);
    const int y_size = std::ceil((input_dim.y - dim_.y + 1.0f)/stride_.y);
    const int z_size = input_dim.z;
    return dim3(x_size, y_size, z_size); 
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

SimpleMatrix<double> Pooling::propagateForward(SimpleMatrix<double>&& input) {
    // Create the approriately sized output
    SimpleMatrix<double> output(outputSize(input.dim()));

    // Reset the affected indices vector
    affected_indices_.resize(output.dim());

    const dim3 pool_size{dim_.x, dim_.y, 1};
    for (auto it = output.begin(); it != output.end(); ++it){

        dim3 out_idx = it.idx();
        dim3 in_idx(out_idx.x*stride_.x, out_idx.y*stride_.y, out_idx.z);

        const auto AoI = input.subMatView(in_idx, pool_size);

        switch (type_){
            case MIN:
            {
                const dim3 min_index = in_idx + minIndex(AoI);
                affected_indices_(out_idx) = min_index;
                output(out_idx) = input(min_index);
                break;
            }

            case MAX: 
            {
                const dim3 max_index= in_idx + maxIndex(AoI);
                affected_indices_(out_idx) = max_index;
                output(out_idx) = input(max_index);
                break;
            }

            case AVG:
                output(out_idx) = mean(AoI);
                break;
        }

    }
    return output;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

SimpleMatrix<double> Pooling::getdLdX(const SimpleMatrix<double>& X, const SimpleMatrix<double>& dLdY){
    SimpleMatrix<double> dLdx(X.dim());
    switch(type_){
        case MIN:
        case MAX:
        {
            for (auto it = affected_indices_.begin(); it != affected_indices_.end(); ++it){
                dLdx(*it) += dLdY(it.idx());
            }
            break;
        }

        case AVG:
        {
            const dim3 pool_size{dim_.x, dim_.y, 1};
            const size_t size = dim_.size();
            for (auto out_it = dLdY.begin(); out_it != dLdY.end(); ++out_it){
                dim3 in_idx(out_it.idx().x*stride_.x, out_it.idx().y*stride_.y, out_it.idx().z);
                dLdx.subMatView(in_idx, pool_size) += (*out_it)/size;
            }
            break;
        }
    }

    return dLdx;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

SimpleMatrix<double> Pooling::propagateBackward(
    const SimpleMatrix<double>& X,    const SimpleMatrix<double>&, 
    const SimpleMatrix<double>& dLdY, double , bool)
{        
    return getdLdX(X, dLdY);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

std::string Pooling::serialize() const {
    std::stringstream ss;
    
    // Record the layer type
    ss << "Pooling\n";
    ss << (type_ == MAX ? "MAX\n" : 
           type_ == MIN ? "MIN\n" : 
                          "AVG\n");

    // Record the pool size 
    serialization::place(ss, dim_.x, "x");
    serialization::place(ss, dim_.y, "y");

    // Record the stride
    ss << "stride\n";
    serialization::place(ss, dim_.x, "x");
    serialization::place(ss, dim_.y, "y");

    return ss.str();
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

bool Pooling::deserialize(std::istream& is) {
    // Check that the label is correct
    serialization::expect<void>(is, "Pooling");

    // Get the layer type
    std::string type_string;
    std::getline(is, type_string);
    if      (type_string == "MAX") type_ = MAX;
    else if (type_string == "MIN") type_ = MIN;
    else if (type_string == "AVG") type_ = AVG;
    else 
        throw std::runtime_error("Unknown pooling type: " + type_string);

    // Get the pool dim
    dim_.x = serialization::expect<int>(is, "x");
    dim_.y = serialization::expect<int>(is, "y");

    // Check for the stride label
    serialization::expect<void>(is, "stride");

    // Get the stride
    stride_.x = serialization::expect<int>(is, "x");
    stride_.y = serialization::expect<int>(is, "y");

    return true;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

} // namespace my_cnn
