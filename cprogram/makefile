neural_omp: neural_net_omp.c matrix_parallel.h helper.h mnist.h
	gcc -o omp neural_net_omp.c -lm -fopenmp

neural_serial: neural_net.c matrix.h helper.h mnist.h
	gcc -o serial neural_net.c -lm