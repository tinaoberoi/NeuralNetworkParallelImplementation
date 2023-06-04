#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "helper.h"
#include "mnist.h"
#include <omp.h>

struct neural_net
{
    int i_n;
    int o_n;
    int hidden_layer;
    int num_neu;
    int layers;
};

typedef struct neural_net neural_net;

void initialise_weights(neural_net *ntwrk, matrix **layer_weights, matrix** C_derivative_wrt_w)
{
    int i_n = ntwrk->i_n;
    int o_n = ntwrk->o_n;
    int n = ntwrk->num_neu;
    int n_l = ntwrk->hidden_layer;
    int total_layers = ntwrk->layers;
    // initialize input weights
    matrix *input_weight = malloc(sizeof(matrix));
    matrix *output_weight = malloc(sizeof(matrix));
    initWeightMatrix(input_weight, n, i_n);
    initWeightMatrix(output_weight, o_n, n);
    layer_weights[0] = input_weight;
    layer_weights[1] = input_weight;
    for (int i = 2; i < total_layers - 1; i++)
    {
        matrix *arr = malloc(sizeof(matrix));
        initWeightMatrix(arr, n, n);
        layer_weights[i] = arr;
    }

    layer_weights[total_layers - 1] = output_weight;

    matrix *input_derivative = malloc(sizeof(matrix));
    matrix *output_derivative = malloc(sizeof(matrix));
    initMatrix(input_derivative, n, i_n, 0);
    initMatrix(output_derivative, o_n, n, 0);
    C_derivative_wrt_w[0] = input_derivative;
    C_derivative_wrt_w[1] = input_derivative;
    for (int i = 2; i < total_layers - 1; i++)
    {
        matrix *arr = malloc(sizeof(matrix));
        initMatrix(arr, n, n, 0);
        C_derivative_wrt_w[i] = arr;
    }

    C_derivative_wrt_w[total_layers - 1] = output_derivative;

}

void setup_input(neural_net *ntwrk, matrix *input_nodes, double** train, int* idx, int m)
{
    // for(int i =0; i<4; i++){
    //     printf("%d :: m :: %d \n", idx[i], m);
    // }
    // printf("\n");
    double arr[4][3] = {{0.40985976,	0.17207617,	0.10262124},
                {0.794434,	0.5128737,	0.17492455},
                {0.81463028,	0.42688077,	0.81860707},
                {0.87657878,	0.88418136,	0.9380702}};
    int i_n = ntwrk->i_n;
    // input_nodes->rows = i_n;
    // input_nodes->cols = 1;
    // input_nodes->arr = malloc(sizeof(double)*i_n*1);
    for(int k = 0; k<i_n; k++){
        for(int k1 = 0; k1<1; k1++){
            int x = idx[m];
            input_nodes->arr[k*1 + k1] = train[x][k*1 + k1];
            // input_nodes->arr[k*1 + k1] = arr[x][k*1 + k1];
        }
    }
    // printf("Inside input set func \n");
    // printMatrix(input_nodes);
}

void setup_output(neural_net *ntwrk, matrix *output_nodes, matrix* y, double* train, int* idx, int m)
{
    double arr[4][2] = {0.9989244,	0.3182569, 1.48223225, 1.48223225, 2.06011812, 2.06011812, 2.69883035, 2.69883035};
    int o_n = ntwrk->o_n;
    initMatrix(output_nodes, o_n, 1, 0);

    // y->rows = o_n;
    // y->cols = 1;
    // y->arr = malloc(sizeof(double)*o_n*1);
    double temp = train[idx[m]];
    for(int k = 0; k<o_n; k++){
        for(int k1 = 0; k1<1; k1++){
            if(k*1 + k1 == temp){
               y->arr[k*1 + k1] = 1.0; 
            } else {
                y->arr[k*1 + k1] = 0.0;
            }
            // y->arr[k*1 + k1] = toBinaryMatrix(train[temp]);
            // y->arr[k*1 + k1] = arr[temp][k*1 + k1];

        }
    }
}

int argMax(matrix* mt, int o_n){
    // printMatrix(mt);
    int idx = 0;
    double max_val = mt->arr[0];
    //  for(int i = 0; i< 10; i++)
    //     printf("%f ", mt->arr[i]);
    // printf("\n");
    // printMatrix(mt);

    for(int i = 1; i< 10; i++){
        if(max_val < mt->arr[i]){
            // printf("max %f %f\n", max_val, mt->arr[i]);
            max_val = mt->arr[i];
            idx = i;
        }
    }
    // printf("Index in arg :: %d \n", idx);
    return idx;
}

void setup_layers(neural_net *ntwrk, matrix **layer_bias, matrix **layer_z, matrix **layer_a, matrix **delta)
{
    int i_n = ntwrk->i_n;
    int o_n = ntwrk->o_n;
    int n_l = ntwrk->hidden_layer;
    int n = ntwrk->num_neu;
    int total_layers = ntwrk->layers;

    ///////// BIAS //////////////////////////
    // set bias for layer 0 (i/p) layer as 0
    matrix *input_bias = malloc(sizeof(matrix));
    initMatrix(input_bias, i_n, 1, 0.0);
    layer_bias[0] = input_bias;

    // setup bias for all layers 1(1st hidden layer) to L-1(last hidden layer)
    for (int i = 1; i < total_layers - 1; i++)
    {
        matrix *arr = malloc(sizeof(matrix));
        // initRandMatrix(arr, n, 1);
        initMatrix(arr, n, 1, 0.0);
        layer_bias[i] = arr;
    }
    // setup bias for o/p layer
    matrix *output_bias = malloc(sizeof(matrix));
    // initRandMatrix(output_bias, o_n, 1);
    initMatrix(output_bias, o_n, 1, 0.0);
    layer_bias[total_layers - 1] = output_bias;

    ///////// LAYER Z //////////////////////////
    // set bias for layer 0 (i/p) layer as 0
    matrix *input_z = malloc(sizeof(matrix));
    initMatrix(input_z, i_n, 1, 0.0);
    layer_z[0] = input_z;

    // setup bias for all layers 1(1st hidden layer) to L-1(last hidden layer)
    for (int i = 1; i < total_layers - 1; i++)
    {
        matrix *arr = malloc(sizeof(matrix));
        initMatrix(arr, n, 1, 0);
        layer_z[i] = arr;
    }
    // setup bias for o/p layer
    matrix *output_z = malloc(sizeof(matrix));
    initMatrix(output_z, o_n, 1, 0.0);
    layer_z[total_layers - 1] = output_z;

    ///////// LAYER A //////////////////////////
    // set activation for layer 0 (i/p) layer as 0
    matrix *input_a = malloc(sizeof(matrix));
    initMatrix(input_a, i_n, 1, 0.0);
    layer_a[0] = input_a;

    // setup bias for all layers 1(1st hidden layer) to L-1(last hidden layer)
    for (int i = 1; i < total_layers - 1; i++)
    {
        matrix *arr = malloc(sizeof(matrix));
        initMatrix(arr, n, 1, 0.0);
        layer_a[i] = arr;
    }
    // setup bias for o/p layer
    matrix *output_a = malloc(sizeof(matrix));
    initMatrix(output_a, o_n, 1, 0.0);
    layer_a[total_layers - 1] = output_a;

    ///////// C DERIVATIVE WRT B //////////////////////////
    // set delta for layer 0 (i/p) layer as 0
    matrix *input_delta = malloc(sizeof(matrix));
    initMatrix(input_delta, i_n, 1, 0.0);
    delta[0] = input_delta;

    // Set up delta
    for (int i = 1; i < total_layers - 1; i++)
    {
        matrix *arr = malloc(sizeof(matrix));
        initMatrix(arr, n, 1, 0);
        delta[i] = arr;
    }

    // setup delta for o/p layer
    matrix *output_delta = malloc(sizeof(matrix));
    initMatrix(output_delta, o_n, 1, 0);
    delta[total_layers - 1] = output_delta;
}

void setup(neural_net *ntwrk, matrix** derivative_sum, matrix** derivative_wrt_sum)
{
    int i_n = ntwrk->i_n;
    int o_n = ntwrk->o_n;
    int n = ntwrk->num_neu;
    int total_layers = ntwrk->layers; 

    matrix *input_delta = malloc(sizeof(matrix));
    initMatrix(input_delta, i_n, 1, 0.0);
    derivative_sum[0] = input_delta;
    // freeMatrix(input_delta);
    
    // setup bias for all layers 1(1st hidden layer) to L-1(last hidden layer)
    for (int i = 1; i < total_layers - 1; i++)
    {
        matrix *arr = malloc(sizeof(matrix));
        // initRandMatrix(arr, n, 1);
        initMatrix(arr, n, 1, 0.0);
        derivative_sum[i] = arr;
    }
    // setup bias for o/p layer
    matrix *output_delta = malloc(sizeof(matrix));
    // initRandMatrix(output_bias, o_n, 1);
    initMatrix(output_delta, o_n, 1, 0.0);
    derivative_sum[total_layers - 1] = output_delta;

    matrix *input_derivative = malloc(sizeof(matrix));
    matrix *output_derivative = malloc(sizeof(matrix));
    initMatrix(input_derivative, n, i_n, 0);
    initMatrix(output_derivative, o_n, n, 0);
    derivative_wrt_sum[0] = input_derivative;
    derivative_wrt_sum[1] = input_derivative;
    for (int i = 2; i < total_layers - 1; i++)
    {
        matrix *arr = malloc(sizeof(matrix));
        initMatrix(arr, n, n, 0);
        derivative_wrt_sum[i] = arr;
    }

    derivative_wrt_sum[total_layers - 1] = output_derivative;
    // freeMatrix(input_delta);
    // freeMatrix(output_delta);
    // freeMatrix(input_derivative);
    // freeMatrix(output_derivative);
}

void forward_propagate(neural_net *ntwrk, matrix **w, matrix **b, matrix *input_nodes, matrix **z, matrix **a, matrix *multMatrix, int isTest)
{
    int i_n = ntwrk->i_n;
    int o_n = ntwrk->o_n;
    int total_layers = ntwrk->layers;

    a[0] = input_nodes;
    
    // if(isTest == 1)
    // {
    //     printf("Input Node:: \n");
    //     for(int i = 0; i<30; i++){
    //         printf("%f :: ", input_nodes->arr[784-i]);
    //     }
    //     printf("\n");
    // }
    for (int l = 1; l < total_layers; l++)
    {
        // printf("Line 224 :: \n");
        // printf("Weights :: rows = %d cols = %d \n", w[l]->rows, w[l]->cols);
        // printf("Activation :: rows = %d cols = %d \n", a[l-1]->rows, a[l-1]->cols);
        // printf(w[l])
        // printf("l = %d \n", l);
        // printf("w rows :: %d cols :: %d and a :: rows :: %d cols :: %d\n", w[l]->rows, w[l]->cols, a[l-1]->rows, a[l-1]->cols);
        if (l == total_layers-1 && isTest == 1){
            // printMatrix(w[l]);
            // printf("-----------------\n");
            // printMatrix(a[l-1]);
            // printf("-----------------\n");
        }
        matmul(w[l], a[l - 1], multMatrix);
        // if (isTest == 1){
        //     printf("After multiplication \n");
        //     printMatrix(multMatrix);
        //     printf("------------------------------------------------------------\n");
        // }
        // printf("Inside feed forward \n");
        // printMatrix(multMatrix);
        // printf("\n");
        
        _matadd(multMatrix, b[l], z[l]);
        
        sigmoid(z[l], a[l]);
    }
    // printf("Inside forward :: \n");
    // printMatrix(z[total_layers-1]);
}

void backpropagate(neural_net *ntwrk, matrix **w, matrix **delta, matrix **z, matrix **a, matrix **derivative_wrt_w, matrix* multMatrix, matrix* sgmdDer)
{
    int L = ntwrk->layers;
    // printf("Delta matrix inside backpropagate \n");
    // for(int l = 0; l <L; l++){
    //     printMatrix(delta[l]);
    // }
    #pragma omp parallel for num_threads(8)
    for(int l = 0; l < L; l++){
        int m = derivative_wrt_w[l]->rows;
        int n = derivative_wrt_w[l]->cols;
        initMatrix(derivative_wrt_w[l], m, n, 0.0);
    }
    for (int l = L - 2; l > 0; l--)
    {
        // matrix *multMatrix = malloc(sizeof(matrix));

        transposeMatmul(w[l + 1], delta[l + 1], multMatrix);
        sigmoidDerivative(z[l], sgmdDer);
        dotProduct(multMatrix, sgmdDer, delta[l]);
        
        matrix *derMultMatrix = malloc(sizeof(matrix));
        transposeMatmulRight(delta[l], a[l - 1], derivative_wrt_w[l]);
    }
    // printf("Before multiplication :: \n");
    // printMatrix(derivative_wrt_w[L-1]);
    matrix *derMultMatrix2 = malloc(sizeof(matrix));
    transposeMatmulRight(delta[L-1], a[L - 2], derMultMatrix2);
    
    matadd(derivative_wrt_w[L-1], derMultMatrix2);
}

void gradient_descent(neural_net *ntwrk, matrix **w, matrix **b, matrix **delta_sum, matrix **derivative_wrt_w_sum)
{
    int total_layers = ntwrk->layers;
    #pragma omp parallel for num_threads(8)
    for (int l = total_layers - 1; l > 2; l++)
    {
        // printf("LINE 289 :: \n");
        // printf("Before weights :: \n");
        // printMatrix(w[l]);
        matdiff(w[l], derivative_wrt_w_sum[l]);
        // printf("After weights :: \n");
        // printMatrix(w[l]);
        // printf("LINE 291 :: \n");
        matdiff(b[l], delta_sum[l]);
    }
}

void initialise_neural_net(int num_input_nodes, int num_output_nodes, int num_hidden_layers, int num_neurons, int ne, int m)
{
    neural_net *network = malloc(sizeof(neural_net));
    network->i_n = num_input_nodes;
    network->o_n = num_output_nodes;
    network->num_neu = num_neurons;
    network->hidden_layer = num_hidden_layers;
    network->layers = num_hidden_layers + 2;
    int total_layers = num_hidden_layers + 2;
    
    // matrix* input_set = malloc(4*sizeof(matrix));

    matrix *input_nodes = malloc(sizeof(matrix));
    input_nodes->rows = num_input_nodes;
    input_nodes->cols = 1;
    input_nodes->arr = malloc(sizeof(double)*num_input_nodes*1);

    matrix *y = malloc(sizeof(matrix));
    y->rows = num_output_nodes;
    y->cols = 1;
    y->arr = malloc(sizeof(double)*num_output_nodes*1);

    matrix *output_nodes = malloc(sizeof(matrix));
    matrix **C_derivative_wrt_w = malloc((total_layers) * sizeof(matrix));
    matrix **layer_weights = malloc((total_layers) * sizeof(matrix));
    matrix **layer_bias = malloc((total_layers) * sizeof(matrix));
    matrix **layer_z = malloc((total_layers) * sizeof(matrix));
    matrix **layer_a = malloc((total_layers) * sizeof(matrix));
    matrix **delta = malloc((total_layers) * sizeof(matrix));
    matrix *multMatrix = malloc(sizeof(matrix));
    matrix *multAdd = malloc(sizeof(matrix));
    // matrix *multMatrix = malloc(sizeof(matrix));
    matrix* sgmdDer = malloc(sizeof(matrix));
    matrix **derivative_wrt_w_sum = malloc((num_hidden_layers + 2) * sizeof(matrix));
    matrix **derivative_sum = malloc((num_hidden_layers + 2) * sizeof(matrix));
    double error_term = 0.0;

    initialise_weights(network, layer_weights, C_derivative_wrt_w);
    setup_layers(network, layer_bias, layer_z, layer_a, delta);

    int n = 6000;
    double alpha = 0.1;
    double multTerm = (alpha / m)* 1.0;
    int* idx = malloc(n*sizeof(int));
    generate_idx(idx, n);
    double start = omp_get_wtime();
    for (int i = 0; i < ne; i++)
    {

        if(i>0){
            shuffle_idx(idx, n);
        }
        int j = 0;

        while( j < n)
        {
            setup(network, derivative_sum, derivative_wrt_w_sum);
            for (int k = 0; k < m; k++)
            {
                setup_input(network, input_nodes, train_image, idx, k);
                setup_output(network, output_nodes, y, train_label, idx, k);
                forward_propagate(network, layer_weights, layer_bias, input_nodes, layer_z, layer_a, multMatrix, 0);
                ///////////OUTPUT ERROR//////////////
                matrix *diff = malloc(sizeof(matrix));
                matrix *derivative = malloc(sizeof(matrix));

                _matdiff(layer_a[total_layers - 1], y, diff);
                cost_function(diff, &error_term);
                sigmoidDerivative(layer_z[total_layers - 1], derivative);
                dotProduct(diff, derivative, delta[network->layers - 1]);
                // // //////// BACKPROPAGATE //////////
                backpropagate(network, layer_weights, delta, layer_z, layer_a, C_derivative_wrt_w, multMatrix, sgmdDer);
                
                
                for(int l = 0; l<total_layers; l++){

                    matadd(derivative_wrt_w_sum[l], C_derivative_wrt_w[l]);

                    matadd(derivative_sum[l], delta[l]);
                }
            }
            j += m;
            scalarmul(derivative_sum, multTerm, total_layers);
            for(int l = 0; l<total_layers; l++){
                matdiff(layer_bias[l], derivative_sum[l]);
            }

            scalarmul(derivative_wrt_w_sum, multTerm, total_layers);

            for(int l = 0; l<total_layers; l++){
                matdiff(layer_weights[l], derivative_wrt_w_sum[l]);
            }
        }
        error_term = error_term/(n*2);
        // printf("At epoch %d :: error is :: %0.10f\n", i, error_term);
    }
    
    double end = omp_get_wtime(); 
    printf("Time taken without OMP : %lf \n", end-start);
    int predicted = 0;
    matrix* test_matrix = malloc(sizeof(matrix));
    for(int i = 0; i<1000; i++){
        test_matrix->rows = num_input_nodes;
        test_matrix->cols = 1;
        test_matrix->arr = test_image[i];
        forward_propagate(network, layer_weights, layer_bias, test_matrix, layer_z, layer_a, multMatrix, 1);
        int idx = argMax(layer_a[total_layers-1], network->o_n);
        // printf("idx :: %d %d \n", idx, (int)test_label[i]);
        if( idx == (int)test_label[i]){
            predicted+=1;
        }
    }
    double res = predicted*10/1000.0;

    printf("CORRECTNESS :: %f\n", res);
    freeMatrix(multMatrix);
    freeMatrix(multAdd);
    freeMatrix(sgmdDer);
    // freeMatrix(derMultMatrix);
    freeMatrix(output_nodes);
    freeMatrix(input_nodes);
    freeMatrix(y);
    // for(int i = 0; i<total_layers; i++){
    //     freeMatrix(derivative_sum[i]);
    //     freeMatrix(derivative_wrt_w_sum[i]);
    // }
    // free(derivative_sum);
    free(derivative_wrt_w_sum);
}

int main(int argc, char *argv[])
{
    load_mnist();
    int layers = atoi(argv[1]);;
    int num_neurons = atoi(argv[2]);;
    int num_epochs = atoi(argv[3]);
    int samples_per_batch = atoi(argv[4]);
    int i_n = 784;
    int o_n = 10;
    int i_n = 784;
    int o_n = 10;
    initialise_neural_net(i_n, o_n, layers-2, num_neurons, num_epochs, samples_per_batch);
    return 0;
}
