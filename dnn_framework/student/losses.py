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
        N, C = x.shape

        # Calcule le softmax avec x en entrée
        yhat = softmax(x)

        # Crée la matrice one-hot
        yhat_target = yhat[np.arange(yhat.shape[0]), target]
        one_hot_class = np.zeros_like(yhat)
        one_hot_class[np.arange(N), target] = 1

        # Calcul de l'entropie croisée
        loss = -np.sum(np.log(yhat_target)) / yhat.shape[0]

        # Calcul du gradient en entrée
        input_grad = (yhat - one_hot_class)/N

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
