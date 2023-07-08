import jax
import jax.numpy as jnp
import numpy as np
import abc

class Optimizer(abc.ABC):
    """
        It is a base class for all optimizers. Custom optimizer may be created by inheriting from this class.
    """
    def __init__(self, loss_fn):
        """
            Initializes the optimizer with loss function.
        """
        self.loss_fn = jax.jit(loss_fn)
        self.grad_fn = jax.jit(jax.grad(loss_fn, argnums=2))

    @abc.abstractmethod
    def update(self, weights: tuple, X: jnp.ndarray, t: jnp.ndarray):
        """
            Performs update of the weights
        """
        pass

    def loss(self, weights: tuple, X: jnp.ndarray, t: jnp.ndarray):
        """
            Returns the loss value.
        """
        return self.loss_fn(weights, X, t)


class SGD(Optimizer):
    """
        Stochastic Gradient Descent optimizer.
    """
    def __init__(
            self, 
            loss_fn,
            learning_rate: float
        ):
        """
            Initializes the optimizer with loss function and learning rate.
        """
        super().__init__(loss_fn)
        self.learning_rate = learning_rate

    def update(self, weights: tuple, X: jnp.ndarray, t: jnp.ndarray):
        gradients = self.grad_fn(weights, X, t)
        for i in range(len(weights)):
            weights[i] -= self.learning_rate * gradients[i]

