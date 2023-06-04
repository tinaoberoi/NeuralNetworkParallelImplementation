#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.h"

int main(){
    matrix* a = (matrix*)malloc(sizeof(struct matrix));
    matrix* b = (matrix*)malloc(sizeof(struct matrix));
    matrix* x = (matrix*)malloc(sizeof(struct matrix));
    matrix* a_ = (matrix*)malloc(sizeof(struct matrix));
    matrix* res1 = (matrix*)malloc(sizeof(struct matrix));
    matrix* res2 = (matrix*)malloc(sizeof(struct matrix));
    matrix* res = (matrix*)malloc(sizeof(struct matrix));
    initRandMatrix(a, 2, 3);
    initMatrix(b, 2, 3, 2);
    initMatrix(x, 3, 2, 3);
    initMatrix(a_, 3, 2, 4);
    printf("Printing A:: \n");
    printMatrix(a);
    printf("-----------");
    printf("Printing B:: \n");
    printMatrix(b);
    printf("-----------");
    transposeMatmul(a, b, res1);
    printf("Printing C:: \n");
    printMatrix(res1);
    // printf("-----------");
    // matmul(a_, b, res2);
    // printf("Printing D:: \n");
    // printMatrix(res2);
    // printf("-----------");
    // printf("Printing E:: \n");
    // sigmoid(a, res);
    // printf("Sigmoid ::\n");
    // printf("-----------");
    transposeMatmulRight(b, a, res);
    printMatrix(res);

    return 0;
}