import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        # Input and output count
        self.input_count = input_count
        self.output_count = output_count
        # Parameters w and
        self.parameters = {
            'w': np.zeros(shape=(self.output_count, self.input_count)),
            'b': np.zeros(shape=(1, self.output_count))
        }

    def get_parameters(self):
        return self.parameters

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        w = self.get_parameters().get('w')
        b = self.get_parameters().get('b')
        return ((x @ np.transpose(w)) + b), x

    def backward(self, output_grad, cache):
        w = self.get_parameters().get('w')
        input_grad = output_grad @ w

        param_grad = {
            'w': output_grad.T @ cache,
            'b': np.sum(output_grad, axis=0)
        }

        return input_grad, param_grad


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):

        self.input_count = input_count
        self.alpha = alpha
        self.epsilon = 1e-8
        self.learning_rate = 0.001
        # Parameters to be learned
        #self.gamma = np.ones(shape=(1, input_count))
        #self.beta = np.zeros(shape=(1, input_count))
        self.parameters = {
            'gamma': np.ones(shape=(1, self.input_count)),
            'beta': np.zeros(shape=(1, self.input_count))
        }

        # Buffers
        self.buffers = {
            'running_mean': np.zeros(input_count),
            'running_var': np.ones(input_count),
            'global_mean': np.zeros(input_count),
            'global_variance': np.zeros(input_count)
        }

        # # Flag for training / evaluation mode
        self._is_training = True

    def get_parameters(self):
        return self.parameters

    def get_buffers(self):
        return self.buffers

    def forward(self, x):
        if self.is_training():
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)

    def _forward_training(self, x):

        # Calcul de la moyenne et de la variance
        mean = np.mean(x, axis=0)
        variance = np.var(x, axis=0)

        # Normalisation de l'entrée
        x_hat = (x - mean) / np.sqrt(variance + self.epsilon)
        y = self.parameters['gamma'] * x_hat + self.parameters['beta']

        # # Mise à jour des statistiques de l'ensemble du lot
        self.buffers['global_mean'] = ((1 - self.alpha) * self.buffers['global_mean']) + self.alpha * mean
        self.buffers['global_variance'] = ((1 - self.alpha) * self.buffers['global_variance']) + self.alpha * variance

        # Place dans le cache x, x_hat, la moyenne et la variance
        cache = (x, x_hat, mean, variance)

        # Retourne la sortie
        return y, cache

    def _forward_evaluation(self, x):
        # Calcul de la moyenne et de la variance
        mean = self.buffers['global_mean']
        variance = self.buffers['global_variance']

        # Normalisation de l'entrée
        x_hat = (x - mean) / np.sqrt(variance + self.epsilon)
        y = self.parameters['gamma'] * x_hat + self.parameters['beta']

        # Place dans le cache x, x_hat, la moyenne et la variance
        cache = (x, x_hat, mean, variance)
        print(type(cache))
        # Retourne la sortie
        return y, cache

    def backward(self, output_grad, cache):
        dLdxhat = output_grad * self.parameters['gamma']
        dLdvar = np.sum(dLdxhat * (cache[0] - cache[2] * (-1/2*(cache[3] + self.epsilon) ** -3/2)))
        dLdmean = - np.sum(dLdxhat/ np.sqrt(cache[3] + self.epsilon))
        input_grad = dLdxhat / np.sqrt(cache[3] + self.epsilon) + (2/np.size(cache[0])) * dLdvar * (cache[0] - cache[2]) + (1/np.size(cache[0])) * dLdmean
        dLdGamma = output_grad * cache[1]
        dldBeta = output_grad

        # Mise à jour des valeurs de gamma et beta par la descente de gradient
        # self.parameters['gamma'] = self.parameters['gamma'] - self.learning_rate * dLdGamma
        # self.parameters['beta'] = self.parameters['beta'] - self.learning_rate * dldBeta

        parameters_grad = {
            'gamma': dLdGamma,
            'beta': dldBeta
        }
        return input_grad, parameters_grad


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        y = 1/(1 + np.exp(-x))
        return y, x

    def backward(self, output_grad, cache):
        sigmoid_forward = 1/(1 + np.exp(-cache))
        input_grad = sigmoid_forward * (1 - sigmoid_forward)
        y = output_grad * input_grad
        return y, None


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        return np.maximum(x, 0), x

    def backward(self, output_grad, cache):
        return output_grad * (cache > 0), None
