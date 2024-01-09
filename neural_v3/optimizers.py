import numpy as np

# contain all the different optimizer for gradient descent

class optimizer_layout:
    def __init__(self, layer_shape) -> None:
        self.layer_shape = layer_shape
        
        # for each parameters
        self.speed = None  
        self.momentum = None

    def reset_optimizer(self) :
        self.momentum = np.zeros(self.layer_shape)


# stochaistic gradient descent
class SGD(optimizer_layout):
    def __init__(self, layer_shape) -> None:
        super().__init__(layer_shape)

    def find_step_size(self, gradient, learning_rate):
        step_size = learning_rate * gradient
        return step_size
    
# root mean squared propagation
class RMSprop(optimizer_layout):
    def __init__(self, layer_shape) -> None:
        super().__init__(layer_shape)

        self.beta = 0.9

        self.momentum = np.zeros(self.layer_shape)

    def find_step_size(self, gradient, learning_rate):
        epsilon = 1e-8   #for numerical stability

        # from the rmsprop formula
        momentum_old = self.momentum
        self.momentum = (self.beta * momentum_old) + ((1 - self.beta) * np.square(gradient))

        # step size
        step_size = (learning_rate / (np.sqrt(self.momentum) + epsilon)) * gradient

        return step_size
    
class Adam(optimizer_layout):
    def __init__(self, layer_shape) -> None:
        super().__init__(layer_shape)

        self.beta = 0.9
        self.beta_2 = 0.999

        self.momentum = np.zeros(self.layer_shape)
        self.speed = np.zeros(self.layer_shape)

        self.iteration = 1

    def find_step_size(self, gradient, learning_rate):
        epsilon = 1e-8

        momemtum_old = self.momentum
        speed_old = self.speed

        # formula
        self.momentum = (self.beta * momemtum_old) + ((1 - self.beta) * gradient)
        self.speed = (self.beta_2 * speed_old) + ((1 - self.beta_2) * np.power(gradient, 2))

        # bias correction
        momentum_estim = self.momentum / (1 - np.power(self.beta, self.iteration))
        speed_estim = self.speed / (1 - np.power(self.beta_2, self.iteration))
        
        step_size = (learning_rate * momentum_estim) / (np.sqrt(speed_estim) + epsilon)

        self.iteration += 1
        return step_size
    
    def reset_optimizer(self):
        self.speed = np.zeros(self.layer_shape)
        return super().reset_optimizer()

