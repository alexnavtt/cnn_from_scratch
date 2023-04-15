#pragma once

#include "cnn_from_scratch/imageUtil.h"
#include "cnn_from_scratch/ModelLayer.h"
#include "cnn_from_scratch/SimpleMatrix.h"

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

    bool checkSize(const SimpleMatrix<float>& input) override {
        return input.dim(0) >= dim_.x && input.dim(1) >= dim_.y;
    }

    SimpleMatrix<float> propagateForward(const SimpleMatrix<float>& input) override {
        // Create the approriately sized output
        SimpleMatrix<float> output({input.dim(0)/dim_.x, input.dim(1)/dim_.y, input.dim(2)});

        // Reset the affected indices vector
        affected_indices_.clear();
        affected_indices_.reserve(output.dim(0) * output.dim(1) * output.dim(2));

        const dim3 pool_size{dim_.x, dim_.y, 1};
        for (unsigned z = 0; z < input.dim(2); z++){
            dim3 in_idx{0, 0, z};
            dim3 out_idx{0, 0, z};

            for (in_idx.y = 0, out_idx.y = 0; in_idx.y < input.dim(1) - dim_.y + 1; in_idx.y += stride_.y, out_idx.y++){
                for (in_idx.x = 0, out_idx.x = 0; in_idx.x < input.dim(0) - dim_.x + 1; in_idx.x += stride_.x, out_idx.x++){
                    const auto AoI = input.subMatCopy(in_idx, pool_size);

                    switch (type_){
                        case MIN:
                        {
                            const dim3 min_index = AoI.minIndex();
                            const dim3 min_index_global = in_idx + min_index;
                            const size_t affected_index = input.getIndex(min_index_global);
                            affected_indices_.push_back(affected_index);
                            output(out_idx) = AoI[affected_index];
                            break;
                        }

                        case MAX: 
                            const dim3 max_index = AoI.maxIndex();
                            const dim3 max_index_global = in_idx + max_index;
                            const size_t affected_index = input.getIndex(max_index_global);
                            affected_indices_.push_back(affected_index);
                            output(out_idx) = AoI[affected_index];
                            break;

                        case AVG:
                            output(out_idx) = AoI.sum() / AoI.size();
                            break;
                    }
                }
            }
        }
        return output;
    }

    SimpleMatrix<float> propagateBackward(const SimpleMatrix<float>& input, const SimpleMatrix<float>& output_grad, [[maybe_unused]] float learning_rate) override{
        SimpleMatrix<float> dLdz(input.dims(), 0);
        size_t out_idx = 0;
        for (size_t in_idx : affected_indices_){
            dLdz[in_idx] = output_grad[out_idx++];
        }
        return dLdz;
    }

private:
    dim2 dim_;
    dim2 stride_;
    PoolingType type_;
    std::vector<size_t> affected_indices_;
};

} // namespace my_cnn
