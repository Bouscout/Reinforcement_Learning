import numpy as np


# this function will contain the logic for defining a layer of the network
# we won't need to represent the neuron individually, they will just be considered by their indexes in the 
# different arrays

# we assume the weight data would come from the previous layer

class layer_layout:
    def __init__(self, input_size: int, output_size: int, optimizer=None, l_r:float = 0.001):
        # we will represent the neurons in a matrix with the num of rows representing the num of neurons
        # on the next layer and the columns representing the weights for each neurons
        self.input_size = input_size # num of neurons in layer
        self.output_size = output_size # num of weights per neuron for next layer
        self.alpha = l_r # learning rate

        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)

        self.input = None # input received through propagation

        self.level = 0

        # initializing optimizers for each parameter
        self.weights_optimizer = optimizer(layer_shape=[self.input_size, self.output_size])
        self.bias_optimizer = optimizer(layer_shape=[1, self.output_size])

    def forward_propagation(self, data:np.array ) :
        # we assume input will arrive in the correct form from prv layer
        # we perform a matrix multiplication with the weights on this layer
        
        self.input = data
        # z = np.dot(self.weights, data)  #old version
        z = np.matmul(data, self.weights)
        # output = z 
        output = z + self.bias
        
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
           raise ValueError('one of the value is overflowing')
       
        return output

    def backward_propagation(self, gradient, learning_rate, average):
        # we will receive a matrix of gradient corresponding to each weight
        # in each neuron, so we just perform a gradient descent to adjust each weight 
        
        # we need to find the gradient of the input which will correspond to the ouput of the previous layer
        
        # derv_input_x1 = sum(w*derv_loss for every w and derv_loss tied to x1)
        input_gradient_set = np.matmul(gradient, self.weights.T) # new version

        # weights update
        # the gradient of the weights from the derivation is found through the operation between the input and the loss gradient
        inputs = self.input[:, :, None]
        gradient_column = gradient[:, None, :]

        weight_gradients = np.matmul(inputs, gradient_column)
        if average :
            weight_gradients = np.mean(weight_gradients, axis=0)
        else :
            weight_gradients = np.sum(weight_gradients, axis=0)

        if self.weights_optimizer :
            step_sizes = self.weights_optimizer.find_step_size(weight_gradients, self.alpha)
            self.weights -= step_sizes
        else :
            # SGD
            self.weights -= (learning_rate * weight_gradients) 
        

        # the bias gradient is only the loss gradient so we sum the gradient options on the row axis
        if average :
            bias_gradient = np.mean(gradient, axis=0)
        else :
            bias_gradient = np.sum(gradient, axis=0)
            
        step_size = self.bias_optimizer.find_step_size(bias_gradient, self.alpha)
        # self.bias -= (learning_rate * bias_gradient)
        self.bias -= step_size

        return input_gradient_set
            
    def initialize_relu_weights(self, stddev:float=None) -> np.ndarray:
        # Calculate the standard deviation for the weights
        if not stddev :
            stddev = np.sqrt(2.0 / self.input_size)

        # Initialize the weights with random values from a Gaussian distribution
        self.weights = np.random.normal(0, stddev, (self.input_size, self.output_size))
        self.bias = np.random.normal(0, stddev, (1, self.output_size))
    


