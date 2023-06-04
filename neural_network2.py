import random
import numpy as np
from mnist import MNIST

class network:
    def __init__(self, i_n, h_l, h_l_n, o_n):
        self.i_n = i_n
        self.h_l = h_l
        self.h_l_n = h_l_n
        self.o_n = o_n
        
        self.input_nodes = self.initialize_input_nodes()
        self.hidden_nodes = np.random.rand(h_l, h_l_n)
        # Initialize output_nodes
        # self.output_nodes = np.zeros(self.o_n).reshape(1, self.o_n)
        self.output_nodes = np.array([1]).reshape(1, 1)
        
        self.input_weights = np.random.rand(i_n, h_l_n)
        self.hidden_weights = np.random.rand(self.h_l_n, self.h_l_n)
        
        self.hidden_layer_bias = np.random.rand(h_l, h_l_n)
        self.output_bias = np.random.rand(1, self.o_n)
        
        self.hidden_z = np.zeros(h_l * h_l_n).reshape(h_l, h_l_n)
        self.hidden_a  = np.zeros(h_l * h_l_n).reshape(h_l, h_l_n)
        
        self.output_z = np.zeros(o_n).reshape(1, o_n)
        self.output_a = np.zeros(o_n).reshape(1, o_n)
        
        self.delta_hidden_layers = np.zeros(h_l * h_l_n).reshape(h_l, h_l_n) 
        self.delta_output = np.zeros(o_n).reshape(1, o_n)
        
    def initialize_input_nodes(self):
        # input_layer = np.random.rand(1, n)
        input_layer = np.array([0.2]).reshape(1, 1)
        return input_layer
        
    def initialize_hidden_weights(self):
        w = []
        # initialisze hidden layer weight matrix
        for _ in range(self.h_l - 1):
            w_h = np.random.rand(self.h_l_n, self.h_l_n)
            w.append(w_h)
        return w
    
    def frontpropagation(self):
        # Feedforward
        # first layer
        layer_0 = np.matmul(self.input_nodes, self.input_weights) + self.hidden_layer_bias[0]
        self.hidden_z[0] = layer_0
        self.hidden_a[0] = self.sigmoid(layer_0)
        error =  np.zeros(self.o_n).reshape(1, self.o_n)

        # from layer 1 to layer last and output layer
        for l in range(1, self.h_l + 1):
            if l == self.h_l:
                layer_l = np.matmul(self.hidden_weights[-1], self.hidden_a[-1]) + self.output_bias
                self.output_z = layer_l
                self.output_a = self.sigmoid(layer_l)
            else:
                layer_l = np.matmul(self.hidden_weights[l-1], self.hidden_a[l-1]) + self.hidden_layer_bias[l-1]
                self.hidden_z[l] = layer_l
                self.hidden_a[l] = self.sigmoid(layer_l)
        
        error = self.calc_output() - self.output_nodes
        return error, self.output_z
    
    def outputerror(self):
       for x in range(self.m):
           self.delta_output[x] = (self.output_a[x] - self.output_nodes) * self.sigmoid_derivative(self.output_z[x])
       
    def calc_output(self):
        max_idx = 0
        arr = self.output_a.flatten()
        x = arr.argmax(axis = 0)
        y = np.zeros(10).reshape(10)
        y[x] = 1
        return y
     
    def backpropagation(self):
        error_activation = self.delay_output * self.hidden_nodes[-1]
        #for output and last layer
        # Why transpose?
        if (self.h_l > 1):
            x = self.delta_hidden_layers[-1].reshape(self.h_l_n, 1)
            x = np.matmul(self.hidden_weights[-1].T.reshape(self.h_l_n, self.o_n), self.delta_output) * self.sigmoid_derivative(self.hidden_z[-1].T.reshape(self.h_l_n, 1))
            error_activation += np.matmul(x, self.hidden_a[self.h_l-2])
            
            # for middle layers
            for l in range(self.h_l-1, 0, -1):
                self.delta_hidden_layers[l-1] = np.matmul(self.hidden_weights[l-1].T.reshape(self.h_l_n, 1), self.delta_hidden_layers[l]) * self.sigmoid_derivative(self.hidden_z[l-1])
                error_activation += np.matmul(self.delta_hidden_layers[l-1], self.hidden_a[l-2].T)  
                
            self.delta_hidden_layers[0] = np.matmul(self.input_weights.T.reshape(self.h_l_n, 1), self.delta_hidden_layers[1]) * self.sigmoid_derivative(self.hidden_z[0])
            error_activation += np.matmul(self.delta_hidden_layers[0], self.input_nodes.T)
        else:
            x = self.delta_hidden_layers[0].reshape(self.h_l_n, 1)
            print(x.shape)
            print(self.delta_output.shape)
            print(self.input_weights.shape)
            print(self.hidden_weights.shape)
            print(self.hidden_weights[0].shape)
            self.delta_hidden_layers[0] = np.matmul(self.delta_output, self.input_weights.T)
            y = self.sigmoid_derivative(self.hidden_z[0]).reshape(self.h_l_n, 1) 
            x = x * y
            error_activation += np.matmul(self.delta_hidden_layers[0].reshape(self.h_l_n, 1), self.input_nodes.T).T
        
        return error_activation, self.delta_output * (self.hidden_weights[-1])

    def is_converged(self, i_w, h_w, o_b, h_b):
        e = 0.0001
        flag = ((np.abs(i_w) < e).all() and (np.abs(o_b) < e).all()) and (np.abs(h_w) < e).all() and (np.abs(h_b) < e).all()
        # for i in range(self.h_l):
        #     flag &= (np.ans(h_w[i]) < e).all()
        #     flag &= (np.abs(h_b[i]) < e).all()
        return flag
        
    def gradient_descent(self, sum_e_o, sum_e_h, sum_e_a_o, sum_e_a_h, m):
        alpha = 0.1
        self.hidden_layer_bias -= (alpha/m) * sum_e_h
        self.output_bias -= (alpha/m) * sum_e_o
        self.hidden_weights -= (alpha/m) * sum_e_a_h
        self.input_weights -= (alpha/m) * sum_e_a_o  
        if (self.is_converged(sum_e_o, sum_e_h, sum_e_a_h, sum_e_a_o)):
            return -1
    
    def sigmoid_derivative(self, z):
        return np.exp(-z) / pow((1+ np.exp(-z)),2)
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def cost_func(self, n):
        return pow(np.linalg.det(self.output_nodes - self.output_a), 2) * 1/ (2*n)
    
def convert_binary(num):
    res = []
    str_num = '{0:010b}'.format(num)
    for s in str_num:
        res.append(int(s))
    return np.array(res).reshape(1, 10)

def main():
    data = MNIST('data')
    X_train, y_train = data.load_training()
    
    h_n = 2 #number of hidden layers
    h_l_n = 10
    
    neural_net = network(784, h_n, h_l_n, 10)
    
    n = len(X_train)
    m = 1000
    c = 0.0
    # for training
    while(True):
        for _ in range(n, 0, -m):
            sum_e_o = neural_net.delta_output
            sum_e_h = neural_net.delta_hidden_layers
            sum_e_a_h = np.zeros(neural_net.h_l * neural_net.h_l_n).reshape(neural_net.h_l, neural_net.h_l_n)
            sum_e_a_o = np.zeros(neural_net.o_n * neural_net.h_l_n).reshape(neural_net.o_n, neural_net.h_l_n)
            # m = random.rand
            for i in range(m):
                neural_net.input_nodes = X_train[i]
                neural_net.output_nodes = convert_binary(y_train[i])
                error, output_z_val = neural_net.frontpropagation()
                neural_net.delay_output = error * neural_net.sigmoid_derivative(output_z_val)
                x, y = neural_net.backpropagation()
                sum_e_o += neural_net.delta_output
                sum_e_h += neural_net.delta_hidden_layers
                sum_e_a_h += x
                sum_e_a_o += y
                c += neural_net.cost_func(n)
            if(neural_net.gradient_descent(sum_e_o, sum_e_h, sum_e_a_o, sum_e_a_h, m) == -1):
                break
            print(c)
        
    
if __name__ == "__main__":
    main()