#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void shuffle_idx(int *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
            srand(time(NULL));
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

void generate_idx(int* idx_arr, int n)
{
    for(int i = 0; i<n; i++){
        idx_arr[i] = i;
    }
}
