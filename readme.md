# Report

- Number of hidden layers = 1, number of neurons in hidden layer = 10.
- Tested for number of epochs = 100, batch size = 10, training sample size = 6000
- Here I limited to 100 epochs since the around 200 the process was killed (probably due to memory shortage).
- alpha used is :: 0.1
- Optimum Batch size came out to be 10 
- The mininum epoch achieved was :: `0.0008333333`

## Serial Code
- The serial code is `neural_net.c`
- Time Taken :: 120.14 sec
- Accuracy :: 75 percent
- Grind rate :: 4994.17

## Parallel Code
- The parallel code is `neural_net_omp.c`
- Time Taken :: 30.8 sec
- Accuracy :: 66 percent
- Grind rate :: 19480.106

- Since the major calculation portion was the matrix multiplications and other operations I parallelised all the matrix operations.
- Apart from that I parallelised on places where I am looping through all the layers and not sharing resources for every input.
- It was not possible according to my architecture to parallelize for `m for loop ` since for that every input will have to have separate layer_a(activation layer array), layer_z(Z of layers) etc.
- The speed increased but the accuracy dropped I beleive its due to some race conditions while paralleslising which I could not debug.
