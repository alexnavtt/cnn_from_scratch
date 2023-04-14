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
    unsigned dim0 = 1;
    unsigned dim1 = 1;
    unsigned stride0 = 1;
    unsigned stride1 = 1;
    PoolingType type = PoolingType::MAX;

    bool checkSize(const SimpleMatrix<float>& input){
        return input.dim(0) >= dim0 && input.dim(1) >= dim1;
    }

    SimpleMatrix<float> propagateForward(const SimpleMatrix<float>& input) override {
        SimpleMatrix<float> output({input.dim(0)/dim0, input.dim(1)/dim1, input.dim(2)});
        const dim3 pool_size{dim0, dim1, 1};
        for (unsigned z = 0; z < input.dim(2); z++){
            dim3 in_idx{0, 0, z};
            dim3 out_idx{0, 0, z};
            for (in_idx.y = 0, out_idx.y = 0; in_idx.y < input.dim(1) - dim1 + 1; in_idx.y += stride1, out_idx.y++){
                for (in_idx.x = 0, out_idx.x = 0; in_idx.x < input.dim(0) - dim0 + 1; in_idx.x += stride0, out_idx.x++){
                    const auto AoI = input.subMatCopy(in_idx, pool_size);
                    switch (type){
                        case MIN:
                            output(out_idx) = AoI.min();
                            break;

                        case MAX: 
                            output(out_idx) = AoI.max();
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
        /** TODO: */
        return input;
    }
};

} // namespace my_cnn
