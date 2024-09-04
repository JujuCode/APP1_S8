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
        # raise NotImplementedError()

    def get_parameters(self):
        return self.parameters

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        w = self.get_parameters().get('w')
        b = self.get_parameters().get('b')
        return ((x @ np.transpose(w)) + b), None

    def backward(self, output_grad, cache):
        raise NotImplementedError()


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):

        self.input_count = input_count
        self.alpha = alpha
        self.epsilon = 1e-10

        # Parameters to be learned
        #self.gamma = np.ones(shape=(1, input_count))
        #self.beta = np.zeros(shape=(1, input_count))
        self.parameters = {
            'gamma': np.zeros(shape=(1, self.input_count)),
            'beta': np.zeros(shape=(1, self.input_count))
        }

        # Buffers
        self.running_mean = np.zeros(input_count)
        self.running_var = np.ones(input_count)

        # Flag for training / evaluation mode
        self.training = True

    def get_parameters(self):
        return self.parameters

    def get_buffers(self):
        return self.running_mean, self.running_var

    def forward(self, x):
        if self.training:
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)

    def _forward_training(self, x):

        # Calcul de la moyenne et de la variance
        self.mean = np.mean(x, axis=0)
        self.variance = np.var(x, axis=0)

        # Normalisation de l'entrée
        self.x_hat = (x - self.mean) / np.sqrt(self.variance + self.epsilon)
        self.y = self.get_parameters().get('gamma') * self.x_hat + self.get_parameters().get('beta')

        # Mise à jour des statistiques de l'ensemble du lot
        self.running_mean = ((1 - self.alpha) * self.running_mean) + self.alpha * self.mean
        self.running_var = ((1 - self.alpha) * self.running_var) + self.alpha * self.variance

        # Place dans le cache x, x_hat, la moyenne et la variance
        self.cache = (x, self.x_hat, self.mean, self.variance)

        # Retourne la sortie
        return self.y, None

    def _forward_evaluation(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()


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
