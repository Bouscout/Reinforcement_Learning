import numpy as np
# this file would contain all the options for the different loss functions

def MSE(y_predict:np.ndarray , y_true:np.ndarray ) -> np.ndarray:
    # mean squared error
    error = np.mean(np.power(y_true - y_predict, 2))
    return error

def derivative_MSE(y_predict:np.ndarray, y_true:np.ndarray):
    error_drv = 2 * (y_predict - y_true)
    num_outputs = y_predict.shape[-1]

    gradient = error_drv / num_outputs

    return gradient

# for multi classification problems
def cross_entropy(y_predict: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    epsilon = 1e-10  # small value to avoid division by zero
    y_predict = np.clip(y_predict, epsilon, 1.0 - epsilon)  # clip values to avoid logarithm of zero
    # loss = -np.sum(y_true * np.log(y_predict)) / np.size(y_true)
    loss2 = -np.mean(y_true * np.log(y_predict))
    return loss2

def derivative_cross_entropy(y_predict: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    epsilon = 1e-10  # small value to avoid division by zero
    y_predict_clipped = np.clip(y_predict, epsilon, 1.0 - epsilon)  # clip values to avoid logarithm of zero
    
    derivative = -(y_true / y_predict_clipped) / np.size(y_true)
    return derivative

def cross_entropy_non_one_hot(y_predict: np.ndarray, y_true: int) -> float :
    epsilon = 1e-10 # to avoid division by zero
    Y_preddict_clipped = np.clip(y_predict, epsilon, 1.0 - epsilon)

    # one hot encoding the label
    all_one_hot = np.zeros_like(y_predict)
    all_one_hot[np.arange(len(y_predict)), y_true] = 1
   
    # from the loss formula
    loss = -np.mean(all_one_hot * np.log(Y_preddict_clipped))
    return loss

def derivate_cross_entropy_non_one_hot(y_predict: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_true = y_true.astype(np.int32) # making sure all values are integer for one hot encoding

    epsilon = 1e-10
    y_predict_clipped = np.clip(y_predict, epsilon, 1.0 - epsilon)
    
    # one hot encoding the label
    all_one_hot = np.zeros_like(y_predict)
    the_shape = np.arange(len(y_predict))
    all_one_hot[np.arange(len(y_predict)), y_true] = 1

    # finding the derivative
    derivative = -(all_one_hot / y_predict_clipped) 
    return derivative


def derivative_softmax_cross_entropy_sparse(y_predict:np.ndarray, y_true:np.ndarray) -> np.ndarray :
    # find probability distribution of output
    exp_data = np.exp(y_predict)
    if np.any(np.isnan(exp_data)) or np.any(np.isinf(exp_data)):
        raise ValueError('one of the value is overflowing')
    softmax_output = exp_data / np.sum(exp_data, axis=1, keepdims=True)

    # find the derivative or negative log with respect to the label
    all_one_hot = np.zeros_like(softmax_output)
    all_one_hot[np.arange(len(softmax_output)), y_true] = 1

    # negative log
    derivative = softmax_output - all_one_hot

    return derivative

def derivative_softmax_cross_entropy(y_predict:np.ndarray, y_true=np.ndarray) -> np.ndarray :
    exp_data = np.exp(y_predict)

    softmax_output = exp_data / np.sum(exp_data, axis=1, keepdims=True)

    one_hot_label = np.zeros_like(softmax_output)
    one_hot_label[np.arange(len(y_predict)), y_true] = 1

    loss = np.log(softmax_output) * one_hot_label

    loss = -np.mean(loss)
   

    return loss 