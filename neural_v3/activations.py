import numpy as np

# this file will contain all the activation function possible that we 
# will place between the layers

class activation:
    def __init__(self) :
        self.input = None
        # self.output = None

# all values between -1 and 1
class tanh(activation):
    def forward_propagation(self, data):
        self.input = data
        output = np.tanh(self.input)
        return output
    
    def backward_propagation(self, output_gradient, learning_gradient):
        drv_output = 1 - np.tanh(self.input) ** 2

        output = np.multiply(output_gradient, drv_output)
        return output
    
# all values between 0 and 1
class sigmoid(activation):
    def forward_propagation(self, data):
            self.input = data
            output = 1 / (1 + np.exp(-data))
            return output
    
    def backward_propagation(self, output_gradient, learning_gradient):
        s = 1 / (1 + np.exp(-self.input))
        drv_output =  s * (1 - s)    

        output = np.multiply(output_gradient, drv_output)
        return output



    
# return zero if value < 0 otherwise return the actual value
class Relu(activation) :
    def forward_propagation(self, data) :
        self.input = data
        return np.maximum(0, data)

    def backward_propagation(self, output_gradient, learning_rate) :
        drv_output = np.where(self.input > 0, 1, 0) # filter values
        # drv_output = drv_output.T
        # output = np.multiply(output_gradient, drv_output)
        output = drv_output * output_gradient
        return output
    
#TO DO
class leaking_Relu(activation):
    pass
    
# return actual value
class linear(activation):
    def forward_propagation(self, data):
        return data
    
    def backward_propagation(self, output_gradient, learning_rate):
        # just return the gradient times one
        return output_gradient

# divide the output between a probability between the different options
# useful for classification problems output layer
class softmax_v1(activation):
    def __init__(self, is_binary:bool = False):
        super().__init__()
        self.is_binary = is_binary

    def forward_propagation(self, data):
        # in order to make sure we don't get overflow warning for too large number
        # clipped_data = np.clip(data, -100, 100)

        self.input = data
        # we suppose it's coming in batches so
        all_outputs = np.zeros_like(data)

        for index in range(len(all_outputs)) :
            m = np.exp(self.input[index])
            if np.any(np.isnan(m)) or np.any(np.isinf(m)):
                clipped_data = np.clip(data, -100, 100)
                self.input = clipped_data
                m = np.exp(self.input[index])

            output = m / np.sum(m)

            all_outputs[index] = output

        # checking if it is not a binary classification problem
        self.output = all_outputs
    
                
        return all_outputs
    
    def backward_propagation(self, output_gradient, learning_rate):
        # size = np.size(self.output)
        # output = np.dot((np.identity(size) - self.output.T) * self.output, output_gradient)

        # return output # old

        #softmax can be the jacobian matrix of of the softmax with respect to the input
        the_shape = self.output.shape[-1]
        identity = np.identity(the_shape)
        
        # formula soft_gradient = softmax(x) * (identity - softmax(x))
        # * being an element wise multiplication


        batch_gradient = np.dot((self.output[0] * (identity - self.output[0])), output_gradient[0]) # first gradient

        # correctly shaping the gradient container array for all batches
        all_gradients = np.zeros(shape=(len(output_gradient), *batch_gradient.shape)) 
        all_gradients[0] = batch_gradient

        for index in range(1, len(output_gradient)) :
            soft_gradient = self.output[index] * (identity - self.output[index])
            batch_gradient = np.dot(soft_gradient, output_gradient[index].T)

            all_gradients[index] = batch_gradient


        # gradient = np.dot(soft_gradients[0], output_gradient[0].T)
        
        return all_gradients
      


class softmax(activation):
    def __init__(self, is_binary: bool = False):
        super().__init__()
        self.is_binary = is_binary

    def forward_propagation(self, logits:np.ndarray):
        self.input = logits

        # Compute softmax values for each batch
        exp_data = np.exp(logits)
        softmax_output = exp_data / np.sum(exp_data, axis=1, keepdims=True)

        self.output = softmax_output
        return softmax_output

    def backward_propagation(self, output_gradient, learning_rate):
        # Compute the Jacobian matrix for softmax
        identity = np.eye(self.output.shape[-1])
        output_w_extra_dim = self.output[: ,:, None]
        jacobian = self.output[:, :, None] * (identity - self.output[:, None, :])

        # Compute the gradient for each batch
        batch_gradients = np.matmul(jacobian, output_gradient[:, :, None])
        all_gradients = batch_gradients[:, :, 0]

        return all_gradients