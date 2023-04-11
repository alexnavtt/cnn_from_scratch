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

    my_cnn::SimpleMatrix<int> M2({5, 1, 1}, 1);
    for (auto& v : M2){
        EXPECT_EQ(v, 1);
    }
}

TEST(Constructor, valueConstructor){
    my_cnn::SimpleMatrix<float> M1({3, 3, 2}, 
        {1, 2, 3,
         4, 5, 6,
         7, 8, 9, 
         
         10, 11, 12,
         13, 14, 15,
         16, 17, 18});

    for (int row = 0; row < 3; row++){
        for (int col = 0; col < 3; col++){
            for (int channel = 0; channel < 2; channel++){
                EXPECT_EQ(M1(row, col, channel), 9*channel + 3*row + col + 1);
            }
        }
    }

    EXPECT_THROW(my_cnn::SimpleMatrix<double> M2({3, 3, 2}, {0, 2, 4, 6, 8}), my_cnn::MatrixSizeException);
}

TEST(Constructor, typeConversion){
    my_cnn::SimpleMatrix<int> M({4, 2, 3}, 3);
    my_cnn::SimpleMatrix<unsigned> M2 = M;

    for (size_t i = 0; i < M.size(); i++){
        EXPECT_EQ(M[i], M2[i]);
    }

    M(0, 0, 0) = -5;
    my_cnn::SimpleMatrix<unsigned> M3(M);
    EXPECT_NE(M[0], M2[0]);
    for (size_t i = 1; i < M.size(); i++){
        EXPECT_EQ(M[i], M2[i]);
    }
}

TEST(Constructor, subMat){
    my_cnn::SimpleMatrix<int> M({3, 3, 2}, 
        {1, 2, 3,
         4, 5, 6,
         7, 8, 9, 
         
         10, 11, 12,
         13, 14, 15,
         16, 17, 18});
    my_cnn::SimpleMatrix<int> M2({2, 2, 2}, M[M.subMatIdx({1, 1, 0}, {2, 2, 2})]);

    my_cnn::SimpleMatrix<int> M_expected({2, 2, 2},
        {5, 6,
         8, 9,
         
         14, 15,
         17, 18});

    EXPECT_EQ(M_expected, M2);
}

TEST(Constructor, subMatCopy){
    my_cnn::SimpleMatrix<int> M({3, 3, 2}, 
        {1, 2, 3,
         4, 5, 6,
         7, 8, 9, 
         
         10, 11, 12,
         13, 14, 15,
         16, 17, 18});
    my_cnn::SimpleMatrix<int> M2 = M.subMatCopy({1, 1, 0}, {2, 2, 2});

    my_cnn::SimpleMatrix<int> M_expected({2, 2, 2},
        {5, 6,
         8, 9,
         
         14, 15,
         17, 18});

    EXPECT_EQ(M_expected, M2);
}

TEST(Assignment, default){
    my_cnn::SimpleMatrix<int> M1({3, 2, 1}, 
        {1, 2,
         3, 4,
         5, 6});
    
    my_cnn::SimpleMatrix<int> M2;
    M2 = M1;

    for (size_t i = 0; i < M1.size(); i++){
        EXPECT_EQ(M1[i], M2[i]);
    }
}

TEST(Assignment, value){
    my_cnn::SimpleMatrix<int> M1({3, 2, 1}, 
        {1, 2,
         3, 4,
         5, 6});
    
    // Directly set the data, column major order
    my_cnn::SimpleMatrix<int> M2({3, 2, 1});
    M2 = {1, 3, 5, 2, 4, 6};

    // Size mismatch
    EXPECT_THROW((M2 = {1,2,3}), my_cnn::MatrixSizeException);

    for (size_t i = 0; i < M1.size(); i++){
        EXPECT_EQ(M1[i], M2[i]);
    }
}

TEST(Assignment, valueMatrix){
    my_cnn::SimpleMatrix<int> M1({3, 2, 1}, 
        {1, 2,
         3, 4,
         5, 6});
    
    // setEntries takes care of reordering
    my_cnn::SimpleMatrix<int> M2({3, 2, 1});
    M2.setEntries({1, 2, 3, 4, 5, 6});

    for (size_t i = 0; i < M1.size(); i++){
        EXPECT_EQ(M1[i], M2[i]);
    }
}

TEST(Assignment, typeConversion){
    my_cnn::SimpleMatrix<float> M1({3, 2, 1}, 
        { 1.0, 2.2,
         -3.0, 4.9,
         -5.3, 6.255});
    
    // setEntries takes care of reordering
    my_cnn::SimpleMatrix<int> M2({3, 2, 1});
    M2 = M1;

    for (size_t i = 0; i < M1.size(); i++){
        EXPECT_EQ((int)M1[i], M2[i]);
    }
}

TEST(SizeCheck, matrix){
    my_cnn::SimpleMatrix<float> M1({3, 4, 5});
    my_cnn::SimpleMatrix<float> M2({3, 4, 5});
    EXPECT_TRUE(M1.sizeCheck(M2));

    my_cnn::SimpleMatrix<float> M3({3, 4, 5});
    my_cnn::SimpleMatrix<float> M4({3, 3, 5});
    EXPECT_FALSE(M3.sizeCheck(M4));

    my_cnn::SimpleMatrix<float> M5({6, 5, 4});
    my_cnn::SimpleMatrix<char>  M6({6, 5, 4});
    EXPECT_TRUE(M5.sizeCheck(M6));
}

TEST(SizeCheck, valarray){
    my_cnn::SimpleMatrix<float> M1({1, 2, 3});
    EXPECT_TRUE(M1.sizeCheck(std::valarray<float>{0, 5, 3, 6, 3, 5}));
    EXPECT_FALSE(M1.sizeCheck(std::valarray<float>{1, 2, 3}));
    EXPECT_TRUE(M1.sizeCheck(std::valarray<char>{1, 2, 3, 4, 5, 6}));
}

TEST(SizeCheck, numeric){
    my_cnn::SimpleMatrix<float> M1({1, 2, 3});
    EXPECT_TRUE(M1.sizeCheck(1));
    EXPECT_TRUE(M1.sizeCheck(2.5));
    EXPECT_TRUE(M1.sizeCheck(8ULL));
    EXPECT_TRUE(M1.sizeCheck(true));

    // Error: std::vector<int> is not an arithmetic type
    // M1.sizeCheck(std::vector<int>{1, 2, 3});
}

TEST(Indexing, getIndex){
    // Uses column major storage, but regular (row, col, depth) matrix indexing
    my_cnn::SimpleMatrix<size_t> M({2, 3, 4});
    EXPECT_EQ(M.getIndex(1, 2, 3), 3*3*2 + 1*3 + 2);
    EXPECT_EQ(M.getIndex(my_cnn::dim3{1, 2, 3}), 3*3*2 + 1*3 + 2);
}

TEST(Indexing, writeSingle){
    my_cnn::SimpleMatrix<float> M({5, 5, 2});
    M(3, 2, 1) = 10;
    size_t flat_idx = M.getIndex(3, 2, 1);

    for (size_t i = 0; i < M.size(); i++){
        if (i == flat_idx) EXPECT_EQ(M[i], 10);
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

    my_cnn::SimpleMatrix<unsigned> M2({3, 3, 1});
    M2[M2.subMatIdx({1, 1, 0}, {2, 2, 1})] = 1;
    EXPECT_EQ(M2(0, 0, 0), 0);
    EXPECT_EQ(M2(0, 1, 0), 0);
    EXPECT_EQ(M2(0, 2, 0), 0);
    EXPECT_EQ(M2(1, 0, 0), 0);
    EXPECT_EQ(M2(1, 1, 0), 1);
    EXPECT_EQ(M2(1, 2, 0), 1);
    EXPECT_EQ(M2(2, 1, 0), 1);
    EXPECT_EQ(M2(2, 2, 0), 1);
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

TEST(Indexing, writeMask){
    my_cnn::SimpleMatrix<int> M({5, 5, 2});
    for (size_t i = 0; i < M.dim(0); i++){
        for (size_t j = 0; j < M.dim(1); j++){
            for (size_t k = 0; k < M.dim(2); k++){
                M(i, j, k) = i + 10*j + 100*k;
            }
        }
    }
    M[M < 100] = -1;

    for (size_t i = 0; i < M.dim(0); i++){
        for (size_t j = 0; j < M.dim(1); j++){
            for (size_t k = 0; k < M.dim(2); k++){
                auto& val = M(i, j, k);
                if (k == 0) EXPECT_EQ(val, -1);
                else EXPECT_EQ(val, i + 10*j + 100*k);
            }
        }
    }
}

TEST(Indexing, readMask){
    my_cnn::SimpleMatrix<int> M({5, 5, 2});
    for (size_t i = 0; i < M.dim(0); i++){
        for (size_t j = 0; j < M.dim(1); j++){
            for (size_t k = 0; k < M.dim(2); k++){
                M(i, j, k) = i + 10*j + 100*k;
            }
        }
    }
    M[M < 100] = -1;
    
    my_cnn::SimpleMatrix<bool> M2(M.dims(), M < 0);

    for (size_t i = 0; i < M.dim(0); i++){
        for (size_t j = 0; j < M.dim(1); j++){
            for (size_t k = 0; k < M.dim(2); k++){
                auto& val = M2(i, j, k);
                if (k == 0) EXPECT_TRUE(val);
                else        EXPECT_FALSE(val);
            }
        }
    }
}

TEST(Arithmetic, scalarAdd){
    my_cnn::SimpleMatrix<float> M1({3, 3, 1});

    M1.setEntries({1, 2, 3,
                   4, 5, 6,
                   7, 8, 9});

    my_cnn::SimpleMatrix<float> M2 = M1 + 1;
    int val = 2;
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            EXPECT_EQ(M2(i,j,0),val++);
        }
    }
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

TEST(Arithmetic, scalarModify){
    my_cnn::SimpleMatrix<float> M1({3, 3, 1});

    M1.setEntries({1, 4, 7,
                   2, 5, 8,
                   3, 6, 9});

    M1 += 1;
    int val = 2;
    for (auto& v : M1){
        EXPECT_EQ(v, val++);
    }
}

TEST(Arithmetic, rangeMatrixModify){
    my_cnn::SimpleMatrix<float> M1({3, 3, 1});

    M1.setEntries({1, 4, 7,
                   2, 5, 8,
                   3, 6, 9});

    // Add a matrix to a range
    my_cnn::dim3 sub_dim{3, 2, 1};
    M1[M1.subMatIdx({0, 1, 0}, sub_dim)] += 
        my_cnn::SimpleMatrix<float>(sub_dim, {6, 3,
                                              5, 2,
                                              4, 1});

    // Add a valarray to a range
    my_cnn::dim3 sub_dim2{3, 1, 1};
    M1[M1.subMatIdx({0, 0, 0}, sub_dim2)] -= {1, 2, 3};

    for (int i : {0, 1, 2}){
        EXPECT_EQ(M1[i], 0);
    }
    for (int i : {3, 4, 5, 6, 7, 8}){
        EXPECT_EQ(M1[i], 10);
    }
}

TEST(Arithmetic, rangeScalarModify){
    my_cnn::SimpleMatrix<float> M1({3, 3, 1});

    M1.setEntries({1, 4, 7,
                   2, 5, 8,
                   3, 6, 9});

    // std::gslice_array doesn't support scalar addition, so you have to create a matrix to add
    my_cnn::dim3 sub_dim{3, 2, 1};
    M1[M1.subMatIdx({0, 1, 0}, sub_dim)] -= my_cnn::SimpleMatrix<float>(sub_dim, 2);

    for (int i : {0, 1, 2}){
        EXPECT_EQ(M1[i], i+1);
    }
    for (int i : {3, 4, 5, 6, 7, 8}){
        EXPECT_EQ(M1[i], i-1);
    }
}

TEST(Arithmetic, channelSum){
    my_cnn::SimpleMatrix<int> M({2, 2, 3});
    M[M.slice(0)] = {1, 2, 3, 4};
    M[M.slice(1)] = {2, 2, 2, 2};
    M[M.slice(2)] = {-1, 0, 1, 0};

    std::valarray<int> sum = M.channelSum();
    EXPECT_EQ(sum[0], 10);
    EXPECT_EQ(sum[1],  8);
    EXPECT_EQ(sum[2],  0);
}

int main(int argc, char* argv[]){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}