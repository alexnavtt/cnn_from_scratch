#include <limits>
#include <gtest/gtest.h>
#include "cnn_from_scratch/Matrix/SimpleMatrix.h"

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

    my_cnn::Dim3 size;
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

TEST(Constructor, copyConstructor){
    my_cnn::SimpleMatrix<float> M1({3, 3, 2}, 
        {1, 2, 3,
         4, 5, 6,
         7, 8, 9, 
         
         10, 11, 12,
         13, 14, 15,
         16, 17, 18});

    my_cnn::SimpleMatrix<float> M2 = M1;
    for (auto it = M1.begin(); it != M1.end(); it++){
        EXPECT_EQ(*it, M2(it.idx()));
    }
}

TEST(Constructor, moveConstructor){
    my_cnn::SimpleMatrix<float> M1({3, 3, 2}, 
        {1, 2, 3,
         4, 5, 6,
         7, 8, 9, 
         
         10, 11, 12,
         13, 14, 15,
         16, 17, 18});

    my_cnn::SimpleMatrix<float> M2 = M1;
    my_cnn::SimpleMatrix<float> M3 = std::move(M1);
    for (auto it = M2.begin(); it != M2.end(); it++){
        EXPECT_EQ(*it, M3(it.idx()));
    }
    EXPECT_NE(M3.size(), M1.size());
}

TEST(Constructor, moveConstructorDiffType){
    my_cnn::SimpleMatrix<float> M1({3, 3, 2}, 
        {1, 2, 3,
         4, 5, 6,
         7, 8, 9, 
         
         10, 11, 12,
         13, 14, 15,
         16, 17, 18});

    my_cnn::SimpleMatrix<int> M2 = std::move(M1);
    for (auto it = M1.begin(); it != M1.end(); it++){
        EXPECT_EQ(*it, M2(it.idx()));
    }
    EXPECT_EQ(M1.size(), M2.size());
}

TEST(Constructor, typeConversion){
    my_cnn::SimpleMatrix<int> M({4, 2, 3}, 3);
    my_cnn::SimpleMatrix<unsigned> M2 = M;

    for (auto it = M.begin(); it != M.end(); it++){
        EXPECT_EQ(*it, M2(it.idx()));
    }

    M(0, 0, 0) = -5;
    my_cnn::SimpleMatrix<unsigned> M3(M);
    EXPECT_NE(M({0, 0, 0}), M2({0, 0, 0}));
    for (auto it = ++M.begin(); it != M.end(); it++){
        EXPECT_EQ(*it, M2(it.idx()));
    }
}

TEST(Constructor, subMatView){
    my_cnn::SimpleMatrix<int> M({3, 3, 2}, 
        {1, 2, 3,
         4, 5, 6,
         7, 8, 9, 
         
         10, 11, 12,
         13, 14, 15,
         16, 17, 18});
    my_cnn::SimpleMatrix<int> M2 = M.subMatView({1, 1, 0}, {2, 2, 2});

    my_cnn::SimpleMatrix<int> M_expected({2, 2, 2},
        {5, 6,
         8, 9,
         
         14, 15,
         17, 18});

    EXPECT_TRUE((my_cnn::matrixEqual(M2, M_expected)));
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

    EXPECT_TRUE((my_cnn::matrixEqual(M2, M_expected)));
}

TEST(Assignment, default){
    my_cnn::SimpleMatrix<int> M1({3, 2, 1}, 
        {1, 2,
         3, 4,
         5, 6});
    
    my_cnn::SimpleMatrix<int> M2;
    M2 = M1;

    for (auto it = M1.begin(); it != M1.end(); it++){
        EXPECT_EQ(*it, M2(it.idx()));
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

    for (auto it = M1.begin(); it != M1.end(); it++){
        EXPECT_EQ(*it, M2(it.idx()));
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

    for (auto it = M1.begin(); it != M1.end(); it++){
        EXPECT_EQ(*it, M2(it.idx()));
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

    for (auto it = M1.begin(); it != M1.end(); it++){
        EXPECT_EQ((int)*it, M2(it.idx()));
    }
}

TEST(Assignment, transpose){
    my_cnn::SimpleMatrix<float> M1({3, 2, 1}, 
        { 1.0, 2.2,
         -3.0, 4.9,
         -5.3, 6.255});
    
    // setEntries takes care of reordering
    my_cnn::SimpleMatrix<float> M2 = transpose(M1);

    for (size_t i = 0; i < M1.dim(0); i++){
        for (size_t j = 0; j < M1.dim(1); j++){
            EXPECT_EQ(M1(i, j), M2(j, i));
        }
    }
}

TEST(Assignment, transposeFat){
    my_cnn::SimpleMatrix<int> M1({3, 3, 2}, 
        {1, 2, 3,
         4, 5, 6,
         7, 8, 9, 
         
         10, 11, 12,
         13, 14, 15,
         16, 17, 18});
    
    // setEntries takes care of reordering
    my_cnn::SimpleMatrix<float> M2 = transpose(M1);

    for (size_t k = 0; k < M1.dim().z; k++){
        for (size_t i = 0; i < M1.dim().x; i++){
            for (size_t j = 0; j < M1.dim().y; j++){
                EXPECT_EQ((M1(i, j, k)), (M2(j, i, k)));
            }
        }
    }
}

TEST(Indexing, getIndex){
    // Uses column major storage, but regular (row, col, depth) matrix indexing
    my_cnn::SimpleMatrix<size_t> M({2, 3, 4});
    EXPECT_EQ(M.getIndex(1, 2, 3), 3*3*2 + 1*3 + 2);
    EXPECT_EQ(M.getIndex(my_cnn::Dim3{1, 2, 3}), 3*3*2 + 1*3 + 2);
}

TEST(Indexing, writeSingle){
    my_cnn::SimpleMatrix<float> M({5, 5, 2});
    M(3, 2, 1) = 10;
    size_t flat_idx = M.getIndex(3, 2, 1);

    for (auto it = M.begin(); it != M.end(); it++){
        if (it.idx() == my_cnn::Dim3(3, 2, 1))
            EXPECT_EQ(*it, 10);
        else
            EXPECT_EQ(*it, 0);
    }
}

TEST(Indexing, readSingle){
    my_cnn::SimpleMatrix<float> M({5, 5, 2});
    M(0, 2, 0) = 10;
    EXPECT_EQ(M(0, 2, 0), 10);
}

TEST(Indexing, writeRange){
    my_cnn::SimpleMatrix<float> M({5, 5, 2});
    M.subMatView({0, 0, 0}, {2, 3, 2}) = 10;

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
    M2.subMatView({1, 1, 0}, {2, 2, 1}) = 1;
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

TEST(Arithmetic, matrixSub){
    my_cnn::SimpleMatrix<float> M1({3, 3, 1});
    my_cnn::SimpleMatrix<float> M2({3, 3, 1});

    M1.setEntries({1, 2, 3,
                   4, 5, 6,
                   7, 8, 9});

    M2.setEntries({2, 3, 4,
                   5, 6, 7,
                   8, 9,10});

    my_cnn::SimpleMatrix<float> M3 = M2 - M1;
    for (auto& v : M3){
        EXPECT_EQ(v,1.0f);
    }
}

TEST(Arithmetic, matrixElementwiseMul){
    my_cnn::SimpleMatrix<float> M1({3, 3, 1});
    my_cnn::SimpleMatrix<float> M2({3, 3, 1});

    M1.setEntries({1, 2, 3,
                   4, 5, 6,
                   7, 8, 9});

    M2.setEntries({2520, 1260, 840,
                    630,  504, 420,
                    360,  315, 280});

    my_cnn::SimpleMatrix<float> M3 = M2 * M1;
    for (auto& v : M3){
        EXPECT_EQ(v,2520.0f);
    }
}

TEST(Arithmetic, matrixElementwiseDiv){
    my_cnn::SimpleMatrix<float> M1({3, 3, 1});
    my_cnn::SimpleMatrix<float> M2({3, 3, 1});

    M1.setEntries({1, 2, 3,
                   4, 5, 6,
                   7, 8, 9});

    M2.setEntries({1, 2, 3,
                   4, 5, 6,
                   7, 8, 9});

    my_cnn::SimpleMatrix<float> M3 = M2 / M1;
    for (auto& v : M3){
        EXPECT_EQ(v, 1.0f);
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
    my_cnn::Dim3 sub_dim{3, 2, 1};
    M1.subMatView({0, 1, 0}, sub_dim) += 
        my_cnn::SimpleMatrix<float>(sub_dim, {6, 3,
                                              5, 2,
                                              4, 1});

    my_cnn::SimpleMatrix<float> expected({3, 3, 1});
    expected.setEntries({1, 10, 10,
                         2, 10, 10,
                         3, 10, 10});
    EXPECT_TRUE(my_cnn::matrixEqual(M1, expected));
}

TEST(Arithmetic, rangeScalarModify){
    my_cnn::SimpleMatrix<float> M1({3, 3, 1});

    M1.setEntries({1, 4, 7,
                   2, 5, 8,
                   3, 6, 9});

    // std::gslice_array doesn't support scalar addition, so you have to create a matrix to add
    my_cnn::Dim3 sub_dim{3, 2, 1};
    M1.subMatView({0, 1, 0}, sub_dim) -= 2;

    my_cnn::SimpleMatrix<float> expected({3, 3, 1});
    expected.setEntries({1, 2, 5,
                         2, 3, 6,
                         3, 4, 7});
    EXPECT_TRUE(my_cnn::matrixEqual(expected, M1));
}

TEST(Arithmetic, matrixViewAdd){
    my_cnn::SimpleMatrix<float> M1({2, 2, 1});
    my_cnn::SimpleMatrix<float> M2({3, 3, 1});

    M1.setEntries({1, 2,
                   4, 5});

    M2.setEntries({9, 8, 7,
                   6, 5, 4,
                   3, 2, 1});

    auto M3 = M1 + M2.subMatView({0, 0, 0}, {2, 2, 1});
    for (auto v : M3){
        EXPECT_EQ(v,10.0f);
    }
}

TEST(Arithmetic, matrixCompoundedAdd){
    my_cnn::SimpleMatrix<float> M1({2, 2, 1});
    my_cnn::SimpleMatrix<float> M2({3, 3, 1});

    M1.setEntries({1, 2,
                   4, 5});

    M2.setEntries({9, 8, 7,
                   6, 5, 5,
                   3, 5, 5});

    my_cnn::SimpleMatrix<float> M3 = M1 + M2.subMatView({0, 0, 0}, {2, 2, 1}) + M2.subMatView({1, 1, 0}, {2, 2, 1});
    for (auto& v : M3){
        EXPECT_EQ(v,15.0f);
    }
}

TEST(Arithmetic, matrixCompoundedOperations){
    const my_cnn::SimpleMatrix<float> M1({2, 2, 1},
       {1, 2,
        4, 5});
    const my_cnn::SimpleMatrix<float> M2({3, 3, 1},
                  {9, 8, 7,
                   6, 5, 5,
                   3, 5, 5});
    const my_cnn::SimpleMatrix<float> M3 = M2.subMatCopy({1, 1, 0}, {2, 2, 1});
    my_cnn::SimpleMatrix<float> M4 = (M1 + M2.subMatView({0, 0, 0}, {2, 2, 1})) * transpose(M3);
    
    for (auto& v : M4){
        EXPECT_EQ(v, 50.0f);
    }
}

TEST(Arithmetic, squareMatMul){
    my_cnn::SimpleMatrix<int> M1({3, 3, 1}), M2({3, 3, 1});

    M1.setEntries({1, 4, 7,
                   2, 5, 8,
                   3, 6, 9});

    M2.setEntries({1, 2, 3,
                   1, 2, 3,
                   1, 2, 3});

    my_cnn::SimpleMatrix<int> M3 = my_cnn::matrixMultiply(M1, M2);

    my_cnn::SimpleMatrix<int> M3_expected({3, 3, 1});
    M3_expected.setEntries({12, 24, 36,
                            15, 30, 45,
                            18, 36, 54});
    
    EXPECT_TRUE(my_cnn::matrixEqual(M3, M3_expected));
}

TEST(Arithmetic, longMatMul){
    my_cnn::SimpleMatrix<int> M1({3, 2, 1}), M2({2, 3, 1});

    M1.setEntries({1, 4,
                   2, 5,
                   3, 6});

    M2.setEntries({1, 2, 3,
                   1, 2, 3});

    my_cnn::SimpleMatrix<int> M3 = my_cnn::matrixMultiply(M1, M2);

    my_cnn::SimpleMatrix<int> M3_expected({3, 3, 1});
    M3_expected.setEntries({5, 10, 15,
                            7, 14, 21,
                            9, 18, 27});
    
    EXPECT_TRUE(my_cnn::matrixEqual(M3, M3_expected));
}

TEST(Arithmetic, badMatMul){
    my_cnn::SimpleMatrix<int> M1({2, 2, 1}), M2({2, 3, 1});

    M1.setEntries({1, 4,
                   3, 6});

    M2.setEntries({1, 2, 3,
                   1, 2, 3});

    EXPECT_THROW(
        (my_cnn::matrixMultiply(M2, M1)),
        my_cnn::MatrixSizeException
    );
}

TEST(Arithmetic, mixedTypeMatMul){
    my_cnn::SimpleMatrix<int>   M1({3, 2, 1});
    my_cnn::SimpleMatrix<float> M2({2, 3, 1});

    M1.setEntries({1, 4,
                   2, 5,
                   3, 6});

    M2.setEntries({1.0f, 2.0f, 3.0f,
                   1.0f, 2.0f, 3.0f});

    my_cnn::SimpleMatrix<float> M3 = matrixMultiply(M1, M2);

    my_cnn::SimpleMatrix<float> M3_expected({3, 3, 1});
    M3_expected.setEntries({5.0f, 10.0f, 15.0f,
                            7.0f, 14.0f, 21.0f,
                            9.0f, 18.0f, 27.0f});
    
    EXPECT_TRUE(my_cnn::matrixEqual(M3, M3_expected));
}

int main(int argc, char* argv[]){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}