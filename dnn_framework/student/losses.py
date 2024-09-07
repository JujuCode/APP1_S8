import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        yhat = softmax(x)
        test = target @ np.log(yhat)
        loss = -np.mean(target @ np.log(yhat))
        input_grad = -target @ (1/yhat)
        return loss, input_grad

def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)



class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        loss = np.mean((x - target) ** 2)
        input_grad = 2 * (x - target) / np.size(x)
        return loss, input_grad
