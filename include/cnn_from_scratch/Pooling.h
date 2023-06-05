#pragma once

#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/ModelLayer.h"
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

namespace my_cnn{

enum PoolingType{
    MAX,
    MIN,
    AVG
};
    
class Pooling : public ModelLayer{
public:
    Pooling(dim2 dim, dim2 stride, PoolingType type = MAX):
    dim_(dim), stride_(stride), type_(type)
    {}

    bool checkSize(const SimpleMatrix<double>& input) override {
        return input.dim(0) >= dim_.x && input.dim(1) >= dim_.y;
    }

    // Get the appropriate output size given the input size
    dim3 outputSize(const dim3& input_dim) const{
        const int x_size = std::ceil((input_dim.x - dim_.x + 1.0f)/stride_.x);
        const int y_size = std::ceil((input_dim.y - dim_.y + 1.0f)/stride_.y);
        const int z_size = input_dim.z;
        return dim3(x_size, y_size, z_size); 
    }

    SimpleMatrix<double> propagateForward(SimpleMatrix<double>&& input) override {
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

    SimpleMatrix<double> propagateBackward(
        const SimpleMatrix<double>& X,    [[maybe_unused]] const SimpleMatrix<double>& Y, 
        const SimpleMatrix<double>& dLdY, [[maybe_unused]] double learning_rate, [[maybe_unused]] double norm_penalty) 
    override
    {        
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

private:
    dim2 dim_;
    dim2 stride_;
    PoolingType type_;
    SimpleMatrix<dim3> affected_indices_;
};

} // namespace my_cnn
