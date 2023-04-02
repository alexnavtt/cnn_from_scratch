#include <limits>
#include <gtest/gtest.h>
#include "cnn_from_scratch/SimpleMatrix.h"

TEST(Constructor, defaultConstructor){
    my_cnn::SimpleMatrix<float> M1;
    my_cnn::SimpleMatrix<double> M2;
    my_cnn::SimpleMatrix<size_t> M3;

    EXPECT_EQ(M1.size(), 0);
    EXPECT_EQ(M2.size(), 0);
    EXPECT_EQ(M3.size(), 0);
}

TEST(Constructor, dimConstructor){
    my_cnn::SimpleMatrix<char> M1({2, 5, 3});
    EXPECT_EQ(M1.size(), 30);
    EXPECT_EQ(M1.dim(0), 2);
    EXPECT_EQ(M1.dim(1), 5);
    EXPECT_EQ(M1.dim(2), 3);
    
    my_cnn::SimpleMatrix<int>  M2({0, 8, 1});
    EXPECT_EQ(M2.size(), 0);
    EXPECT_EQ(M2.dim(0), 0);
    EXPECT_EQ(M2.dim(1), 8);
    EXPECT_EQ(M2.dim(2), 1);

    my_cnn::SimpleMatrix<int>  M3({0, 0, 0});
    EXPECT_EQ(M3.size(), 0);
    EXPECT_EQ(M3.dim(0), 0);
    EXPECT_EQ(M3.dim(1), 0);
    EXPECT_EQ(M3.dim(2), 0);

    my_cnn::dim3 size;
    size.x = 2;
    size.y = 4;
    size.z = 13;
    my_cnn::SimpleMatrix<double> M4(size);
    EXPECT_EQ(M4.size(), 104);
    EXPECT_EQ(M4.dim(0), 2);
    EXPECT_EQ(M4.dim(1), 4);
    EXPECT_EQ(M4.dim(2), 13);
}

TEST(Constructor, defaultVal){
    my_cnn::SimpleMatrix<char> M1({2, 5, 3});
    for (auto& v : M1){
        EXPECT_EQ(v, 0);
    }
}

TEST(Indexing, writeSingle){
    my_cnn::SimpleMatrix<float> M({5, 5, 2});
    M(3, 2, 1) = 10;

    for (size_t i = 0; i < M.size(); i++){
        if (i == 38) EXPECT_EQ(M[i], 10);
        else EXPECT_EQ(M[i], 0);
    }
}

TEST(Indexing, readSingle){
    my_cnn::SimpleMatrix<float> M({5, 5, 2});
    M[10] = 10;
    EXPECT_EQ(M(0, 2, 0), 10);
}

TEST(Indexing, writeRange){
    my_cnn::SimpleMatrix<float> M({5, 5, 2});
    M[M.subMatIdx({0, 0, 0}, {2, 3, 2})] = 10;

    for (size_t i = 0; i < M.dim(0); i++){
        for (size_t j = 0; j < M.dim(1); j++){
            for (size_t k = 0; k < M.dim(2); k++){
                if (i < 2 && j < 3 && k < 2)
                    EXPECT_EQ(M(i, j, k), 10);
                else
                    EXPECT_EQ(M(i, j, k), 0);
            }
        }
    }
}

TEST(Indexing, readRange){
    my_cnn::SimpleMatrix<int> M({5, 5, 2});
    for (size_t i = 0; i < M.dim(0); i++){
        for (size_t j = 0; j < M.dim(1); j++){
            for (size_t k = 0; k < M.dim(2); k++){
                M(i, j, k) = i + 10*j + 100*k;
            }
        }
    }

    auto MSubRange = M.subMatCopy({1, 2, 0}, {3, 2, 1});
    EXPECT_EQ(MSubRange(0, 0, 0), 21);
    EXPECT_EQ(MSubRange(1, 0, 0), 22);
    EXPECT_EQ(MSubRange(2, 0, 0), 23);
    EXPECT_EQ(MSubRange(0, 1, 0), 31);
    EXPECT_EQ(MSubRange(1, 1, 0), 32);
    EXPECT_EQ(MSubRange(2, 1, 0), 33);
}

TEST(Arithmetic, matrixAdd){
    my_cnn::SimpleMatrix<float> M1({3, 3, 1});
    my_cnn::SimpleMatrix<float> M2({3, 3, 1});

    M1.setEntries({1, 2, 3,
                   4, 5, 6,
                   7, 8, 9});

    M2.setEntries({9, 8, 7,
                   6, 5, 4,
                   3, 2, 1});

    my_cnn::SimpleMatrix<float> M3 = M1 + M2;
    for (auto& v : M3){
        EXPECT_EQ(v,10.0f);
    }
}

int main(int argc, char* argv[]){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}