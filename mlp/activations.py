#Todas as funções de ativação e suas derivadas.
import numpy as np
from scipy.special import erf

class Activation:

    def forward(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError
    
class Sigmoid(Activation):
    
    def __repr__(self):
        return "Sigmoid"

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):

        s = self.forward(x)

        return s * (1 - s)
    
class Tanh(Activation):

    def __repr__(self):
        return "Tanh"

    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
class ReLU(Activation):

    def __repr__(self):
        return "ReLU"

    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)
    
class LeakyReLU(Activation):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x):
        return np.where(x > 0, 1, self.alpha)

class ELU(Activation):

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x):
        return np.where(
            x > 0,
            x,
            self.alpha * (np.exp(x) - 1)
        )

    def derivative(self, x):
        return np.where(
            x > 0,
            1,
            self.alpha * np.exp(x)
        )

class SELU(Activation):

    alpha = 1.67326324
    scale = 1.05070098

    def forward(self, x):
        return self.scale * np.where(
            x > 0,
            x,
            self.alpha * (np.exp(x) - 1)
        )

    def derivative(self, x):
        return self.scale * np.where(
            x > 0,
            1,
            self.alpha * np.exp(x)
        )

class Softplus(Activation):

    def forward(self, x):
        return np.log1p(np.exp(x))

    def derivative(self, x):
        return 1 / (1 + np.exp(-x))
    
class Swish(Activation):

    def forward(self, x):

        return x * (
            1 / (1 + np.exp(-x))
        )

    def derivative(self, x):

        sig = 1 / (1 + np.exp(-x))

        return sig + x * sig * (1 - sig)

class Mish(Activation):

    def forward(self, x):

        return x * np.tanh(
            np.log1p(np.exp(x))
        )

    def derivative(self, x):

        sp = np.log1p(np.exp(x))
        tsp = np.tanh(sp)

        sig = 1 / (1 + np.exp(-x))

        return tsp + x * sig * (1 - tsp**2)

class GELU(Activation):

    def forward(self, x):

        return 0.5 * x * (
            1 + erf(x / np.sqrt(2))
        )

    def derivative(self, x):

        pdf = np.exp(
            -x**2 / 2
        ) / np.sqrt(2*np.pi)

        return (
            0.5 * (
                1 + erf(x/np.sqrt(2))
            )
            + x * pdf
        )

class Softmax(Activation):

    def forward(self, x):

        exp = np.exp(
            x - np.max(x, axis=1, keepdims=True)
        )

        return exp / np.sum(
            exp,
            axis=1,
            keepdims=True
        )

    def derivative(self, x):
        pass