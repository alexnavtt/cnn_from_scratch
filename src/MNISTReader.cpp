#include <assert.h>
#include <filesystem>
#include "cnn_from_scratch/MNISTReader.h"

#define DATA_NUM_HEADER_LINES 4
#define LABEL_NUM_HEADER_LINES 2

namespace my_cnn{
    
MNISTReader::MNISTReader(std::string image_filename, std::string label_filename)
{
    assert(std::filesystem::exists(image_filename));
    assert(std::filesystem::exists(label_filename));

    image_stream_.open(image_filename, std::ios::in | std::ios::binary);
    label_stream_.open(label_filename, std::ios::in | std::ios::binary);

    // Check if input streams are open and handle any errors
    if (!image_stream_.is_open() || !label_stream_.is_open()) {
        std::cerr << "Error opening input files.\n";
    }

    // Read the magic number from each stream
    int32_t magic_number = read<int32_t>(image_stream_);
    assert(magic_number == 2051);
    magic_number = read<int32_t>(label_stream_);
    assert(magic_number == 2049);

    // Make sure both streams have the same number of images
    int32_t num_images[2];
    num_images[0] = read<int32_t>(image_stream_);
    num_images[1] = read<int32_t>(label_stream_);
    assert(num_images[0] == num_images[1]);
    num_images_ = num_images[0];

    // Read the sizes of the images
    image_height_ = read<int32_t>(image_stream_);
    image_width_  = read<int32_t>(image_stream_);

    std::cout << "Opened files \"" << image_filename << "\" and \"" << label_filename << "\"\n";
    std::cout << "Files contain " << num_images_ << " images and labels\n";
    std::cout << "Each image is of size " << image_width_ << "x" << image_height_ << "\n";
}

MNISTReader::~MNISTReader(){
    image_stream_.close();
    label_stream_.close();
}

Image MNISTReader::getImage(size_t idx){
    current_index_ = idx;

    // Set the index for the image stream
    size_t img_char_start_idx = DATA_NUM_HEADER_LINES*sizeof(int32_t) + idx * image_height_ * image_width_;
    image_stream_.seekg(img_char_start_idx);

    // Set the index for the label stream
    size_t label_char_start_idx = LABEL_NUM_HEADER_LINES*sizeof(int32_t) + idx;
    label_stream_.seekg(label_char_start_idx);

    return nextImage();
}

Image MNISTReader::nextImage(){
    if (current_index_ >= num_images_){
        throw std::runtime_error("Cannot extract image " + std::to_string(current_index_) + " out of " + std::to_string(num_images_) + " images");
    }

    Image img;

    SimpleMatrix<unsigned char> out({image_height_, image_width_, 1});
    for (int row = 0; row < image_height_; row++){
        for (int col = 0; col < image_width_; col++){
            out(row, col, 0) = read<unsigned char>(image_stream_);
        }
    }

    img.label = read<unsigned char>(label_stream_);
    img.data = (out - 128)/255.0f;
    current_index_++;
    return img;
}

template<typename T>
T MNISTReader::read(std::fstream& stream){
    T val;
    char* buf = reinterpret_cast<char*>(&val);
    if (isBigendian()){
        stream.read(&buf[0], sizeof(T));
    }else{
        for (int i = sizeof(T)-1; i >= 0; i--){
            stream.read(&buf[i], 1);
        }
    }
    return val;
}

} // namespace my_cnn
