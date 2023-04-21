#include <assert.h>
#include <filesystem>
#include "cnn_from_scratch/MNISTReader.h"

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

Image MNISTReader::nextImage(){
    Image img;

    SimpleMatrix<unsigned char> out({image_height_, image_width_, 1});
    for (int row = 0; row < image_height_; row++){
        for (int col = 0; col < image_width_; col++){
            out(row, col, 0) = read<unsigned char>(image_stream_);
        }
    }

    img.label = read<unsigned char>(label_stream_);
    img.data = std::move(out);
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
