#pragma once

#include <fstream>
#include <string>
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

namespace my_cnn{

struct Image{
    SimpleMatrix<float> data;
    uint8_t label;
};
    
class MNISTReader{
public:
    MNISTReader(std::string image_filename, std::string label_filename);
    ~MNISTReader();

    Image nextImage();

    Image getImage(size_t idx);

private:
    std::fstream image_stream_;
    std::fstream label_stream_;
    
    int32_t num_images_ = 0;
    int32_t image_width_ = 0;
    int32_t image_height_ = 0;
    size_t current_index_ = 0;

    template<typename T>
    T read(std::fstream& stream);

    static bool isBigendian(){
        int i = 0x01020304;
        return ((char*)(&i))[0] == 1;
    }    
};

} // namespace my_cnn
