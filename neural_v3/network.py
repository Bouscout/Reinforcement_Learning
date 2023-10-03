from typing import Any
import numpy as np
from math import floor
import random
from neural_v3.activations import sigmoid, linear, softmax, tanh, Relu
from neural_v3.layer import layer_layout
from neural_v3.optimizers import SGD, RMSprop, Adam
from neural_v3.loss_functions import *


# this file will assemble all the components in order to make network easy to use

class network():
    def __init__(self, neurons:list, activation:str ='tanh', loss_func:str = 'MSE', learning_rate:float =0.01, optimizer:str="SGD") -> None:
        
        self.learning_rate = learning_rate
        self.first_feed = False
        self.neurons = neurons # list of neurons at each layer

        # choices for the activations layers
        activations_choice = {
            'sigmoid' : sigmoid,
            'linear' : linear,
            'tanh' : tanh,
            'relu' : Relu,
        }
        self.activ_function = activations_choice[activation] # defining global activation layer
 
        self.learning_rate = learning_rate

        # declaring the loss function
        loss_functions_choices = {
            'MSE' : [MSE, derivative_MSE] ,
            'cross-entropy': [cross_entropy_non_one_hot, derivate_cross_entropy_non_one_hot],
        }
        self.loss_function = loss_functions_choices[loss_func]

        self.error = None # the overall error of the network

        #optimizers choises
        optimizer_choices = {
            'SGD' : SGD,
            'RMSPROP' : RMSprop,
            "ADAM" : Adam,
        }

        self.optimizer_chosen = optimizer_choices[optimizer.upper()]

        # preparing the layers
        self.layers = []
        self.activ_layers = []
        for num in range(len(neurons) - 1) :
            self.layers.append(layer_layout(input_size=neurons[num], output_size=neurons[num+1], optimizer=self.optimizer_chosen, l_r=learning_rate))

            # initializing the weight differently if relu
            if activation == 'relu' :
                layer = self.layers[num]
                layer.initialize_relu_weights()  
           
            # activation layer
            self.activ_layers.append(self.activ_function())
        

    def feed(self, input: np.ndarray) -> np.ndarray:
        # checking if first layer created
        if self.first_feed == False:
            output_size = self.layers[0].output_size
            self.layers[0] = layer_layout(input_size=input.shape[-1], 
                                        output_size=output_size,
                                        optimizer=self.optimizer_chosen,
                                        l_r=self.learning_rate,
                                        )
            self.first_feed = True

        output = input
        for lay, activ in zip(self.layers, self.activ_layers) :
        # for lay in self.layers  :
            non_active_output = lay.forward_propagation(output)
            output = activ.forward_propagation(non_active_output)
            
        return output
    
    # function to propagate the error backwards and adjust weight and bias
    def adjust(self, output_gradient:np.ndarray, average=True) -> None:
        gradient = output_gradient
        for activ, lay in zip(reversed(self.activ_layers) ,reversed(self.layers)):
            gradient_activ = activ.backward_propagation(gradient, self.learning_rate)
            gradient = lay.backward_propagation(gradient_activ, self.learning_rate, average)


    # function to train the network
    def train(self, X:np.ndarray, Y:np.ndarray, epochs:int = 5, batch_size: int = 1, shuffle:bool=False) -> None :

        loss_function, loss_function_derivative = self.loss_function

        X, Y = self.divide_batch(X, Y, batch_size, shuffle=shuffle) # data will be in batch of size batch_size

        BATCH_SIZE = batch_size
        for _ in range(epochs):

            # for x, y in zip(X, Y) :
            for index in range(len(X)) :
                # array of length batch_size
                x_batch = X[index] 
                y_batch = Y[index]

                output = x_batch

                output = self.feed(input=output)

                if np.any(np.isnan(output)) or np.any(np.isinf(output)) :
                    print()

                #error and backprog
                error = np.mean(loss_function(output, y_batch))
                gradients = loss_function_derivative(output, y_batch)

                if np.any(np.isnan(gradients)) or np.any(np.isinf(gradients)) :
                    print()

                self.adjust(gradients)


       
                
              

        self.error = error

        return output

    def predict(self, input: np.ndarray) -> np.ndarray:
        output = self.feed(input=input)
        return output
    
    # overiding the call function
    def __call__(self, input:np.ndarray) -> np.ndarray:
        return self.predict(input=input)

        
    def divide_batch(self, x_data:np.ndarray, y_data:np.ndarray, size:int, shuffle:bool = False) :
        # checking if divisble
        if len(x_data) % size != 0 :
            raise ValueError('the number of datapoint is not perfectly divisble by the size of batches')

        # an array shuffler :
        def shuffler(arr1, arr2):
            current_index = len(arr1)
            random_index = len(arr1)
            # while there are still elements to shuffle
            while current_index != 0 :
                random_index = floor((random.random() * current_index)) 
                current_index -= 1

                # swapping them
                arr1[current_index], arr1[random_index] = arr1[random_index], arr1[current_index]

                # swapping the y arrays with the same index
                arr2[current_index], arr2[random_index] = arr2[random_index], arr2[current_index]

            return (arr1, arr2)
        
        if shuffle :
            x_data, y_data = shuffler(x_data, y_data)

        # declaring the container arrays
        x_shape = x_data.shape
        x_shape = (int((len(x_data)/size)), size, *x_shape[1:])
        x_batch = np.zeros(shape=x_shape)

        # for y
        y_shape = y_data.shape
        y_batch = np.zeros(shape=(int(len(y_data)/size), size, *y_shape[1:]))


        batch_index = 0
        for index in range(int(len(x_data) / size)) :
            # assigin the x batch
            subset_x = x_data[batch_index:batch_index + size]
            x_batch[index] = subset_x

            # for y
            subset_y = y_data[batch_index:batch_index + size]
            y_batch[index] = subset_y

            #incrementing
            batch_index += size

        return (x_batch, y_batch)
    
    # get models weights and bias
    def get_params(self):
        parameters = [0 for _ in range(len(self.layers))]
        for index, layer in enumerate(self.layers) :
            parameters[index] = (np.copy(layer.weights), np.copy(layer.bias))

        return parameters
    
    def reset_optimizers(self):
        for layer in self.layers :
            layer.weights_optimizer.reset_optimizer()
            layer.bias_optimizer.reset_optimizer()

    # set the weights and bias from given numpy arrays of correct shape
    # should be in a format with list of tuples containing weights and bias
    def set_params(self, params):
        if len(params) != len(self.layers) :
            raise ValueError('mismatch num of layers')
        
        for index, variables in enumerate(params):
            weights, bias = variables
            self.layers[index].weights = weights
            self.layers[index].bias = bias
            
        self.first_feed = True   

    # create a model with the received list of tuple of weights and bias as layers
    def set_model_with_params(self, params):
        layer = []
        activ_layer = []

        for i in range(len(params)) :
            input_size, output_size = params[i][0].shape 

            layer_init = layer_layout(input_size, output_size, optimizer=self.optimizer_chosen, l_r=self.learning_rate)
            layer_init.weights = params[i][0]
            layer_init.bias = params[i][1]

            layer.append(layer_init)
            activ_layer.append(self.activ_function())

        self.layers = layer
        self.activ_layers = activ_layer
