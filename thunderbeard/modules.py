import abc
import jax.numpy as jnp
import numpy as np
import jax

from thunderbeard.optimizers import Optimizer

class Module(abc.ABC):
    """
        It is a base class for all modules. Custom module may be created by inheriting from this class.
    """
    def __init__(self, input_shape: tuple, output_shape: tuple):
        """
            Initializes the module with input and output shapes.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
            Performs forward pass of the module.
        """
        return self.pure()(x, *self.get_weights())

    @abc.abstractmethod
    def get_weights(self) -> tuple:
        """
            Returns the weights of the module.
        """
        pass

    @abc.abstractmethod
    def pure(self):
        """
            Returns the function that implements this module. It must take input and weights as parameters and return output.
        """
        pass
    
    def __or__(self, other):
        """
            Combines two modules in series, returns a new Module.
        """
        if not isinstance(other, Module):
            raise TypeError("Cannot chain Module with non-Module")
        if self.output_shape != other.input_shape:
            raise ValueError("Cannot chain Modules with incompatible shapes")
        class ChainedModule(Module):
            def __init__(self, first, second):
                super().__init__(first.input_shape, second.output_shape)
                self.weights = tuple(list(first.get_weights()) + list(second.get_weights()))
                self.first = first
                self.second = second

            def get_weights(self):
                return self.weights
            
            def forward(self, X):
                return self.pure()(X, *self.weights)
            
            def pure(self):
                def pure_forward(X, *weights):
                    return self.second.pure()(self.first.pure()(X, *weights[:len(self.first.get_weights())]), *weights[len(self.first.get_weights()):])
                return pure_forward

        return ChainedModule(self, other)
    
    def __add__(self, other):
        """
            Combines two modules in parallel and merges the results by adding them, returns a new Module.
        """
        return self.combine_in_parallel(other, lambda x, y: x + y)
    
    def combine_in_parallel(self, other, merge_operation):
        """
            Combines two modules in parallel and merges the results by using the given merge operation, returns a new Module.
        """
        if not isinstance(other, Module):
            raise TypeError("Cannot combine Module with non-Module")
        class CombinedModule(Module):
            def __init__(self, first, second):
                super().__init__(first.input_shape, first.output_shape)
                self.weights = tuple(list(first.get_weights()) + list(second.get_weights()))
                self.first = first
                self.second = second

            def get_weights(self):
                return self.weights
            
            def forward(self, X):
                return self.pure()(X, *self.weights)
            
            def pure(self):
                def pure_forward(X, *weights):
                    return merge_operation(self.first.pure()(X, *weights[:len(self.first.get_weights())]), self.second.pure()(X, *weights[len(self.first.get_weights()):]))
                return pure_forward

        return CombinedModule(self, other)
    
    def num_params(self):
        """
            Returns the total number of parameters of the module.
        """
        return jnp.sum(jnp.array([weight.size for weight in self.get_weights()]))
    
    def fit(self, X: jnp.ndarray, t: jnp.ndarray, optimizer: Optimizer, epochs: int, batch_size: int):
        """
            Fits the module to the given data using the given optimizer.
        """
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                optimizer.update(self.get_weights(), X[i:i+batch_size], t[i:i+batch_size])
            print(f"Epoch {epoch+1} loss: {optimizer.loss(self.get_weights(), X, t)}")

class FullyConnectedLayer(Module):
    """
        A fully connected layer with given number of inputs, outputs and activation function.
    """
    def __init__(self, inputs, outputs, activation):
        """
            Initializes the fully connected layer with given number of inputs, outputs and activation function.
        """
        super().__init__((inputs,), (outputs,))
        self.W = jnp.array(np.random.randn(outputs, inputs))
        self.b = jnp.zeros((outputs,))
        self.activation = activation
    
    def get_weights(self):
        return self.W, self.b
    
    def pure(self):
        def pure_forward(X, W, b):
            return self.activation(jnp.dot(X, W.T) + b)
        return pure_forward
    
class FullyConnectedLayerWith2DInput(Module):
    """
        Processes all the rows of the input matrix into the same FF layer and returns a matrix of the same shape as the input
    """
    def __init__(self, input_shape, outputs, activation):
        """
            Initializes the fully connected layer with given input_shape, number of outputs and activation function.
        """
        if (len(input_shape) != 2):
            raise ValueError("input_shape must be 2D")
        super().__init__(input_shape, (input_shape[0], outputs))
        self.W = jnp.array(np.random.randn(outputs, input_shape[1]))
        self.b = jnp.zeros((outputs,))
        self.activation = activation
    
    def get_weights(self):
        return self.W, self.b
    
    def pure(self):
        def pure_forward(X, W, b):
            if len(X.shape) == 3:
                return jnp.array([pure_forward(x, W, b) for x in X])
            elif len(X.shape) == 2:
                return self.activation(jnp.dot(X, W.T) + b)
            else:
                raise ValueError("input must be 2D or 3D (batched)")
        return pure_forward

class AttentionLayer(Module):
    """
        A single attention layer with given input vectors size and context window size.
    """
    def __init__(self, d_model, ctx_win_size):
        """
            Initializes the attention layer with given input vectors size (d_model) and context window size.
        """
        super().__init__((ctx_win_size, d_model), (ctx_win_size, d_model))
        self.Wq, self.Wk, self.Wv = (jnp.array(np.random.randn(d_model, d_model)) for _ in range(3))
    
    def get_weights(self):
        return self.Wq, self.Wk, self.Wv
    
    def pure(self):
        def pure_forward(X, Wq, Wk, Wv):
            Q = jnp.tensordot(X, Wq, axes=((-1), (0)))
            K = jnp.tensordot(X, Wk, axes=((-1), (0)))
            V = jnp.tensordot(X, Wv, axes=((-1), (0)))
            if len(Q.shape) == 2:
                QKT = jnp.dot(Q, K.T)
            else:
                QKT = jnp.array([jnp.dot(Q[i], K[i].T) for i in range(Q.shape[0])])
            QKT_normalized = jax.nn.softmax(QKT / jnp.sqrt(K.shape[-1]))
            if len(Q.shape) == 2:
                return jnp.dot(QKT_normalized, V)
            else:
                return jnp.array([jnp.dot(QKT_normalized[i], V[i]) for i in range(QKT_normalized.shape[0])])
        return pure_forward

class MultiHeadedAttentionLayer(Module):
    """
        A multi-headed attention layer with given number of heads, input vectors size and context window size.
    """
    def __init__(self, n_heads, d_model, ctx_win_size):
        """
            Initializes the multi-headed attention layer with given number of heads, input vectors size (d_model) and context window size.
        """
        super().__init__((ctx_win_size, d_model), (ctx_win_size, d_model))
        self.attention_layers = [AttentionLayer(d_model, ctx_win_size) for _ in range(n_heads)]
        self.Wo = jnp.array(np.random.randn(n_heads * d_model, d_model))
    
    def get_weights(self):
        return tuple([weight for layer in self.attention_layers for weight in layer.get_weights()] + [self.Wo])
    
    def pure(self):
        def pure_forward(X, *weights):
            attention_outputs = [layer.pure()(X, *weights[i:i+3]) for i, layer in enumerate(self.attention_layers)]
            if len(attention_outputs[0].shape) == 2:
                attention_outputs_concatenated = jnp.concatenate(attention_outputs, axis=1)
                return jnp.dot(attention_outputs_concatenated, weights[-1])
            else:
                attention_outputs_concatenated = jnp.array(
                    [
                        jnp.concatenate([attention_outputs[j][i, :, :] for j in range(len(attention_outputs))], axis=1)
                    for i in range(attention_outputs[0].shape[0])]
                )
                return jnp.array(
                    [
                        jnp.dot(attention_outputs_concatenated[i], weights[-1])
                    for i in range(attention_outputs_concatenated.shape[0])]
                )
        return pure_forward
    
class IdentityLayer(Module):
    """
        An identity layer that does nothing.
    """
    def __init__(self, input_shape, output_shape):
        """
            Initializes the identity layer with given input and output shapes.
        """
        super().__init__(input_shape, output_shape)
    
    def get_weights(self):
        return ()
    
    def pure(self):
        def pure_forward(X):
            return X
        return pure_forward
    
class NormalizationLayer(Module):
    """
        A normalization layer that normalizes the input.
    """
    def __init__(self, input_shape, output_shape):
        """
            Initializes the normalization layer with given input and output shapes.
        """
        super().__init__(input_shape, output_shape)
        self.gamma = jnp.array([1.0])
        self.beta = jnp.array([0.0])
    
    def get_weights(self):
        return self.gamma, self.beta
    
    def pure(self):
        def pure_forward(X, gamma, beta):
            mean = jnp.mean(X, axis=-1)
            variance = jnp.var(X, axis=-1)
            return gamma * (X - mean[..., jnp.newaxis]) / jnp.sqrt(variance + 1e-8)[..., jnp.newaxis] + beta
        return pure_forward
    
