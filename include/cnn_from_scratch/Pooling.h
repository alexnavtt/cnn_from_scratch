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

    bool checkSize(const SimpleMatrix<float>& input) override {
        return input.dim(0) >= dim_.x && input.dim(1) >= dim_.y;
    }

    SimpleMatrix<float> propagateForward(SimpleMatrix<float>&& input) override {
        // Create the approriately sized output
        SimpleMatrix<float> output({input.dim(0)/dim_.x, input.dim(1)/dim_.y, input.dim(2)});

        // Reset the affected indices vector
        affected_indices_.clear();
        affected_indices_.reserve(output.size());

        const dim3 pool_size{dim_.x, dim_.y, 1};
        for (auto out_it = output.begin(); out_it != output.end(); ++out_it){
            dim3 in_idx(out_it.idx().x*stride_.x, out_it.idx().y*stride_.y, out_it.idx().z);
            const auto AoI = input.subMatView(in_idx, pool_size);

            switch (type_){
                case MIN:
                {
                    const dim3 min_index = minIndex(AoI);
                    const dim3 min_index_global = in_idx + min_index;
                    const size_t affected_index = input.getIndex(min_index_global);
                    affected_indices_.push_back(affected_index);
                    *out_it = AoI(min_index);
                    break;
                }

                case MAX: 
                {
                    const dim3 max_index = maxIndex(AoI);
                    const dim3 max_index_global = in_idx + max_index;
                    const size_t affected_index = input.getIndex(max_index_global);
                    affected_indices_.push_back(affected_index);
                    *out_it = AoI(max_index);
                    break;
                }

                case AVG:
                    *out_it = mean(AoI);
                    break;
            }
        }
        return output;
    }

    SimpleMatrix<float> propagateBackward(const SimpleMatrix<float>& X, const SimpleMatrix<float>& Y, const SimpleMatrix<float>& dLdY, [[maybe_unused]] float learning_rate) override{        
        SimpleMatrix<float> dLdx(X.dim());
        switch(type_){
            case MIN:
            case MAX:
            {
                size_t out_idx = 0;
                for (size_t in_idx : affected_indices_){
                    dLdx[in_idx] += dLdY[out_idx++];
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
    std::vector<size_t> affected_indices_;
};

} // namespace my_cnn
