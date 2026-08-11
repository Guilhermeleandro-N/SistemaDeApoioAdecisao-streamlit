import numpy as np


class Optimizer:
    """
    Classe base para todos os otimizadores.
    """

    def update(self, layer):
        raise NotImplementedError


# =====================================================
# SGD
# =====================================================

class SGD(Optimizer):

    def __repr__(self):
        return "SGD"

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, layer):

        if not hasattr(layer, "weights"):
            return

        layer.weights -= (
            self.lr *
            layer.grad_weights
        )

        layer.bias -= (
            self.lr *
            layer.grad_bias
        )


# =====================================================
# MOMENTUM
# =====================================================

class Momentum(Optimizer):
    """
    (Polyak, 1964) - Some methods of speeding up the convergence of iteration methods
    """

    def __repr__(self):
        return "Momentum"

    def __init__(
        self,
        lr=0.01,
        momentum=0.9
    ):

        self.lr = lr
        self.momentum = momentum

        self.vw = {}
        self.vb = {}

    def update(self, layer):

        if not hasattr(layer, "weights"):
            return

        lid = id(layer)

        if lid not in self.vw:

            self.vw[lid] = np.zeros_like(
                layer.weights
            )

            self.vb[lid] = np.zeros_like(
                layer.bias
            )

        self.vw[lid] = (
            self.momentum *
            self.vw[lid]
            -
            self.lr *
            layer.grad_weights
        )

        self.vb[lid] = (
            self.momentum *
            self.vb[lid]
            -
            self.lr *
            layer.grad_bias
        )

        layer.weights += self.vw[lid]
        layer.bias += self.vb[lid]


# =====================================================
# RMSPROP
# =====================================================

class RMSProp(Optimizer):

    def __repr__(self):
        return "RMSProp"

    def __init__(
        self,
        lr=0.001,
        beta=0.9,
        epsilon=1e-8
    ):

        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon

        self.sw = {}
        self.sb = {}

    def update(self, layer):

        if not hasattr(layer, "weights"):
            return

        lid = id(layer)

        if lid not in self.sw:

            self.sw[lid] = np.zeros_like(
                layer.weights
            )

            self.sb[lid] = np.zeros_like(
                layer.bias
            )

        self.sw[lid] = (
            self.beta *
            self.sw[lid]
            +
            (1 - self.beta)
            *
            (layer.grad_weights ** 2)
        )

        self.sb[lid] = (
            self.beta *
            self.sb[lid]
            +
            (1 - self.beta)
            *
            (layer.grad_bias ** 2)
        )

        layer.weights -= (
            self.lr
            *
            layer.grad_weights
            /
            (
                np.sqrt(self.sw[lid])
                +
                self.epsilon
            )
        )

        layer.bias -= (
            self.lr
            *
            layer.grad_bias
            /
            (
                np.sqrt(self.sb[lid])
                +
                self.epsilon
            )
        )


# =====================================================
# ADAM
# =====================================================

class Adam(Optimizer):

    def __repr__(self):
        return "Adam"

    def __init__(
        self,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        self.lr = lr

        self.beta1 = beta1
        self.beta2 = beta2

        self.epsilon = epsilon

        self.t = 0

        self.mw = {}
        self.vw = {}

        self.mb = {}
        self.vb = {}

    def update(self, layer):

        if not hasattr(layer, "weights"):
            return

        lid = id(layer)

        if lid not in self.mw:

            self.mw[lid] = np.zeros_like(
                layer.weights
            )

            self.vw[lid] = np.zeros_like(
                layer.weights
            )

            self.mb[lid] = np.zeros_like(
                layer.bias
            )

            self.vb[lid] = np.zeros_like(
                layer.bias
            )

        self.t += 1

        # Primeiro momento

        self.mw[lid] = (
            self.beta1 *
            self.mw[lid]
            +
            (1 - self.beta1)
            *
            layer.grad_weights
        )

        self.mb[lid] = (
            self.beta1 *
            self.mb[lid]
            +
            (1 - self.beta1)
            *
            layer.grad_bias
        )

        # Segundo momento

        self.vw[lid] = (
            self.beta2 *
            self.vw[lid]
            +
            (1 - self.beta2)
            *
            (layer.grad_weights ** 2)
        )

        self.vb[lid] = (
            self.beta2 *
            self.vb[lid]
            +
            (1 - self.beta2)
            *
            (layer.grad_bias ** 2)
        )

        # Bias correction

        mw_hat = (
            self.mw[lid]
            /
            (1 - self.beta1 ** self.t)
        )

        mb_hat = (
            self.mb[lid]
            /
            (1 - self.beta1 ** self.t)
        )

        vw_hat = (
            self.vw[lid]
            /
            (1 - self.beta2 ** self.t)
        )

        vb_hat = (
            self.vb[lid]
            /
            (1 - self.beta2 ** self.t)
        )

        layer.weights -= (
            self.lr
            *
            mw_hat
            /
            (
                np.sqrt(vw_hat)
                +
                self.epsilon
            )
        )

        layer.bias -= (
            self.lr
            *
            mb_hat
            /
            (
                np.sqrt(vb_hat)
                +
                self.epsilon
            )
        )


# =====================================================
# ADAMW
# =====================================================

class AdamW(Adam):
    """"
    (Loshchilov e Hutter, 2019) - Decoupled Weight Decay Regularization
    """
    def __repr__(self):
        return "AdamW"
    
    def __init__(
        self,
        lr=0.001,
        weight_decay=0.01,
        **kwargs
    ):

        super().__init__(
            lr=lr,
            **kwargs
        )

        self.weight_decay = weight_decay

    def update(self, layer):

        super().update(layer)

        if not hasattr(layer, "weights"):
            return

        layer.weights -= (
            self.lr
            *
            self.weight_decay
            *
            layer.weights
        )


# =====================================================
# NADAM
# =====================================================

class Nadam(Optimizer):
    """"
    (Dozat, 2016) - INCORPORATING NESTEROV MOMENTUM INTO ADAM
    """

    def __repr__(self):
        return "Nadam"
    
    def __init__(
        self,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        self.lr = lr

        self.beta1 = beta1
        self.beta2 = beta2

        self.epsilon = epsilon

        self.t = 0

        self.mw = {}
        self.vw = {}

        self.mb = {}
        self.vb = {}

    def update(self, layer):

        if not hasattr(layer, "weights"):
            return

        lid = id(layer)

        if lid not in self.mw:

            self.mw[lid] = np.zeros_like(
                layer.weights
            )

            self.vw[lid] = np.zeros_like(
                layer.weights
            )

            self.mb[lid] = np.zeros_like(
                layer.bias
            )

            self.vb[lid] = np.zeros_like(
                layer.bias
            )

        self.t += 1

        self.mw[lid] = (
            self.beta1 *
            self.mw[lid]
            +
            (1 - self.beta1)
            *
            layer.grad_weights
        )

        self.mb[lid] = (
            self.beta1 *
            self.mb[lid]
            +
            (1 - self.beta1)
            *
            layer.grad_bias
        )

        self.vw[lid] = (
            self.beta2 *
            self.vw[lid]
            +
            (1 - self.beta2)
            *
            (layer.grad_weights ** 2)
        )

        self.vb[lid] = (
            self.beta2 *
            self.vb[lid]
            +
            (1 - self.beta2)
            *
            (layer.grad_bias ** 2)
        )

        mw_hat = (
            self.mw[lid]
            /
            (1 - self.beta1 ** self.t)
        )

        mb_hat = (
            self.mb[lid]
            /
            (1 - self.beta1 ** self.t)
        )

        vw_hat = (
            self.vw[lid]
            /
            (1 - self.beta2 ** self.t)
        )

        vb_hat = (
            self.vb[lid]
            /
            (1 - self.beta2 ** self.t)
        )

        nesterov_w = (
            self.beta1 * mw_hat
            +
            (
                (1 - self.beta1)
                *
                layer.grad_weights
                /
                (
                    1 -
                    self.beta1 ** self.t
                )
            )
        )

        nesterov_b = (
            self.beta1 * mb_hat
            +
            (
                (1 - self.beta1)
                *
                layer.grad_bias
                /
                (
                    1 -
                    self.beta1 ** self.t
                )
            )
        )

        layer.weights -= (
            self.lr
            *
            nesterov_w
            /
            (
                np.sqrt(vw_hat)
                +
                self.epsilon
            )
        )

        layer.bias -= (
            self.lr
            *
            nesterov_b
            /
            (
                np.sqrt(vb_hat)
                +
                self.epsilon
            )
        )