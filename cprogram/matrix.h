#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>

struct matrix
{
    int rows;
    int cols;
    double* arr;
};
typedef struct matrix matrix;

void printMatrix(const matrix* mt);


void matmul(matrix* m1, matrix* m2, matrix* res)
{
    assert(m1->cols == m2->rows);
    int n = m2->cols;
    int o = m1->cols;
    int m = m1->rows;
    
    if(res == NULL){
        res = (matrix*)malloc(sizeof(matrix));
    }
    
    res->arr = (double*)malloc(sizeof(double) * m * n);
    res->rows = m;
    res->cols = n;
    
    for(int i = 0; i < m; i++){
        for(int j =0; j<n; j++){
            res->arr[i*n + j] = 0.0;
            for(int k = 0; k<o; k++){
                res->arr[i*n +j] += m1->arr[i*o + k] * (m2->arr[k*n + j]);
            }
        }
    }
}

void scalarmul(matrix** m1, double mult, int L)
{
   
    for(int l = 0; l<L; l++){
        int m = m1[l]->rows;
        int n = m1[l]->cols;
       
        for(int i = 0; i<m; i++){
            for(int j = 0; j<n; j++){
                m1[l]->arr[i*n + j] = mult * (m1[l]->arr[i*n + j]);
            }
        }
    }
}

void matadd(const matrix* m1, const matrix* m2)
{
    assert(m1->rows == m2->rows && m1->cols == m2->cols);
    int m = m2->rows;
    int n = m2->cols;
    
   
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            m1->arr[i*n +j] += m2->arr[i*n + j];
        }
    }
}

void matdiff(const matrix* m1, const matrix* m2)
{
    assert(m1->rows == m2->rows && m1->cols == m2->cols);
    int m = m1->rows;
    int n = m1->cols;
    
   
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            m1->arr[i*n +j] -= m2->arr[i*n + j];
        }
    }
}

void _matadd(const matrix* m1, const matrix* m2, matrix* res)
{
    assert(m1->rows == m2->rows && m1->cols == m2->cols);
    int m = m1->rows;
    int n = m1->cols;
    
    if(res == NULL){
        res = (matrix*)malloc(sizeof(matrix));
    }
    
    res->arr = (double*)malloc(sizeof(double) * m * n);
    res->rows = m;
    res->cols = n;
   
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            res->arr[i*n +j] = m1->arr[i*n + j] + m2->arr[i*n + j];
        }
    }
}

void _matdiff(const matrix* m1, const matrix* m2, matrix* res)
{
    assert(m1->rows == m2->rows && m1->cols == m2->cols);
    int m = m1->rows;
    int n = m1->cols;
    
    if(res == NULL){
        res = (matrix*)malloc(sizeof(matrix));
    }
    
    res->arr = (double*)malloc(sizeof(double) * m * n);
    res->rows = m;
    res->cols = n;
    
   
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            res->arr[i*n +j] = m1->arr[i*n + j] - m2->arr[i*n + j];
        }
    }
}


void transposeMatmul(const matrix* m1, const matrix* m2, matrix* res)
{
    assert(m1->rows == m2->rows);
    int m = m1->cols;
    int n = m2->cols;
    int o = m1->rows;

    if(res == NULL){
        res = (matrix*)malloc(sizeof(matrix));
    }
    res->arr = (double*)malloc(sizeof(double) * m * n);
    res->rows = m;
    res->cols = n;

   
    for(int i = 0; i < m; i++){
        for(int j =0; j<n; j++){
            res->arr[i*n + j] = 0.0;
            for(int k = 0; k<o; k++){
                res->arr[i*n +j] += m1->arr[k*m + i] * (m2->arr[k*n + j]);
            }
        }
    }
}
void transposeMatmulRight(const matrix* m1, const matrix* m2, matrix* res)
{
    assert(m1->cols == m2->cols);
    int m = m1->rows;
    int n = m2->rows;
    int o = m1->cols;

    if(res == NULL){
        res = (matrix*)malloc(sizeof(matrix));
    }
    res->arr = (double*)malloc(sizeof(double) * m * n);
    res->rows = m;
    res->cols = n;

   
    for(int i = 0; i < m; i++){
        for(int j =0; j<n; j++){
            res->arr[i*n + j] = 0.0;
            //
            for(int k = 0; k<o; k++){
                res->arr[i*n +j] += m1->arr[i*o + k] * (m2->arr[j*o + k]);
            }
        }
    }
}

void matCopy(matrix* cmt, matrix* mt){
    int m = mt->rows;
    int n = mt->cols;
   

    for(int i = 0; i<m; i++){
        for(int j = 0; j<n; j++){
            cmt[i*n + j] = mt[i*n + j];
        }
    }
}

void dotProduct(const matrix* m1, const matrix* m2, matrix* res)
{
    assert(m1->rows == m2->rows && m1->cols == m2->cols);
    int m = m1->rows;
    int n = m1->cols;
    
    if(res == NULL){
        res = (matrix*)malloc(sizeof(matrix));
    }
    res->arr = (double*)malloc(sizeof(double) * m * n);
    res->rows = m;
    res->cols = n;
    
   
    for(int i = 0; i < m; i++){
        for(int j =0; j<n; j++){
            res->arr[i*n +j] = m1->arr[i*n + j] * (m2->arr[i*n + j]);
        }
    }
}

void sigmoid(const matrix* mt, matrix* res){
    int m = mt->rows;
    int n = mt->cols;
    
    if(res == NULL){
        res = (matrix*)malloc(sizeof(matrix));
    }

    res->arr = (double*)malloc(sizeof(double) * m * n);
    res->rows = m;
    res->cols = n;
    
   
    for(int i = 0; i < m; i++){
        for(int j =0; j<n; j++){
            res->arr[i*n +j] = 1/(1 + exp(-1.0 * (mt->arr[i*n + j])));
            // printf("z :: %f sgmd of z :: %f \n", mt->arr[i*n + j], res->arr[i*n + j]);
        }
    }
}

void sigmoidDerivative(const matrix* mt, matrix* res){
    int m = mt->rows;
    int n = mt->cols;
    
    if(res == NULL){
        res = (matrix*)malloc(sizeof(matrix));
    }

    res->arr = (double*)malloc(sizeof(double) * m * n);
    res->rows = m;
    res->cols = n;
    
   
    for(int i = 0; i < m; i++){
        for(int j =0; j<n; j++){
            res->arr[i*n +j] = exp(-1*(mt->arr[i*n + j])) / pow((1+ exp(-1 * (mt->arr[i*n + j]))),2);
        }
    }
}

void printMatrix(const matrix* mt){
    int m = mt->rows;
    int n = mt->cols;

    for(int i = 0; i<m; i++){
        for(int j = 0; j<n; j++){
            printf("%0.10f ", mt->arr[i*n + j]);
        }
        printf("\n");
    }
}

void initMatrix(matrix* mt, int m, int n, int alpha){
    mt->rows = m;
    mt->cols = n;
    mt->arr = (double*)malloc(sizeof(double) * m * n);
    
   
    for(int i = 0; i<m; i++){
        for(int j = 0; j<n; j++){
            mt->arr[i*n + j] = alpha;
        }
    }
}

// void toBinaryMatrix(double* mt)
// {
//     int m = mt->rows;
//     int n = mt->cols;

//     for(int i =0; i<m; i++){
//         for(int j = 0; j<n; j++){
//             mt->arr[i*n + j] = decimalToBinary((int)mt->arr[i*n +j], int res);
//         }
//     }
// }

void initWeightMatrix(matrix* mt, int m, int n){
    mt->rows = m;
    mt->cols = n;
    mt->arr = (double*)malloc(sizeof(double) * m * n);
    for(int i = 0; i<m; i++){
        for(int j = 0; j<n; j++){
            mt->arr[i*n + j] = 0.31;
        }
    }
}

void initRandMatrix(matrix* mt, int m, int n){
    mt->rows = m;
    mt->cols = n;
    mt->arr = (double*)malloc(sizeof(double) * m * n);
    
   
    for(int i = 0; i<m; i++){
        for(int j = 0; j<n; j++){
            // srand()
            mt->arr[i*n + j] = ((double)rand()/(double)(RAND_MAX));
        }
    }
}

void cost_function(matrix* mt, double* res){
    int m = mt->rows;
    int n = mt->cols;
    // printf("Inside Cost Func \n");
    // printMatrix(mt);
    *res = 0.0;
    //
    for(int i = 0; i<m; i++){
        for(int j = 0; j<n; j++){
            *res += pow(mt->arr[i*n + j], 2);
        }
    }
    // printf("res = %f\n", *res);
}

void freeMatrix(matrix* mt){
    free(mt->arr);
}