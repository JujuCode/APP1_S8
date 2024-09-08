import numpy as np

from dnn_framework.layer import Layer

epsilon = 1e-08

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
            'w': np.random.randn(output_count, input_count),
            'b': np.random.randn(output_count)
        }

    def get_parameters(self):
        return self.parameters

    def get_buffers(self):
        buffers = {}
        return buffers

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

        self.learning_rate = 0.01

        # Parameters to be learned
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
        x_hat = (x - mean) / np.sqrt(variance + epsilon)
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
        x_hat = (x - mean) / np.sqrt(variance + epsilon)
        y = self.parameters['gamma'] * x_hat + self.parameters['beta']

        # Place dans le cache x, x_hat, la moyenne et la variance
        cache = (x, x_hat, mean, variance)
        # print(type(cache))
        # Retourne la sortie
        return y, cache

    def backward(self, output_grad, cache):

        # Attribution des valeurs du cache (pour simplifier les calculs de gradient)
        x, x_hat, mean, variance = cache
        M = x.shape[0]

        # Calcul du gradient de x_hat
        dLdxhat = output_grad * self.parameters['gamma']

        # Calcul des gradients de la variance et de la moyenne
        dLdvar = np.sum(dLdxhat * (x - mean) * -0.5 * (variance + epsilon) ** -1.5, axis=0)
        dLdmean = -np.sum(dLdxhat / np.sqrt(variance + epsilon), axis=0)

        # Calcul du gradient de l'entrée
        input_grad = dLdxhat / np.sqrt(variance + epsilon) + (2/M) * dLdvar * (x - mean) + (1/M) * dLdmean

        # Calcul des gradients des paramètres gamma et beta
        dLdGamma = np.sum(output_grad * x_hat, axis=0)
        dldBeta = np.sum(output_grad, axis=0)

        # Mise à jour des valeurs de gamma et beta par la descente de gradient
        #self.parameters['gamma'] -= (self.learning_rate * dLdGamma)
        #self.parameters['beta'] -= (self.learning_rate * dldBeta)

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
        parameters = {}
        return parameters

    def get_buffers(self):
        buffers = {}
        return buffers

    def forward(self, x):
        y = 1/(1 + np.exp(-x))
        return y, x

    def backward(self, output_grad, cache):
        sigmoid_forward = 1/(1 + np.exp(-cache))
        input_grad = sigmoid_forward * (1 - sigmoid_forward)
        y = output_grad * input_grad
        return y, input_grad


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        parameters = {}
        return parameters

    def get_buffers(self):
        buffers = {}
        return buffers

    def forward(self, x):
        dictx = {'x': x}
        return np.maximum(x, 0), dictx

    def backward(self, output_grad, cache):
        return output_grad * (cache['x'] > 0), cache
