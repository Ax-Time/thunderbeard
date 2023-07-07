# thunderbeard
thunderbeard is a small personal project of a python library that implements basic deep learning modules to be combined together to create more complex models. It is heavily dependent on [jax](https://github.com/google/jax).

## Example usage
~~~python
# Create a feedforward network with layers [10, 20, 2]
nn = tbm.FullyConnectedLayer(10, 20, jax.nn.relu) \
    | tbm.FullyConnectedLayer(20, 1, jax.nn.relu) \
    | tbm.FullyConnectedLayer(1, 2, jax.nn.softmax)

# Create some random data
x = jnp.array(np.random.randn(10))
y = nn.forward(x)
~~~

## Implement a basic transformer encoder
~~~python
# Create a transformer encoder
d_model = 128
n_heads = 4
ff_hidden_layer_neurons = 512
ctx_win_size = 256

attention = tbm.MultiHeadedAttentionLayer(n_heads, d_model, ctx_win_size) + tbm.IdentityLayer((ctx_win_size, d_model), (ctx_win_size, d_model))
normalization = tbm.NormalizationLayer((ctx_win_size, d_model), (ctx_win_size, d_model))
ff_nn = (
    tbm.FullyConnectedLayerWith2DInput((ctx_win_size, d_model), ff_hidden_layer_neurons, jax.nn.relu) \
    | tbm.FullyConnectedLayerWith2DInput((ctx_win_size, ff_hidden_layer_neurons), d_model, jax.nn.relu)
) + tbm.IdentityLayer((ctx_win_size, d_model), (ctx_win_size, d_model))

encoder = attention \
        | normalization \
        | ff_nn

# Create some random data
batch_size = 32
X = jnp.array(np.random.randn(batch_size, ctx_win_size, d_model))

# Forward pass
Y = encoder.forward(X)
~~~

## Calculating gradients
~~~python
# Create a feedforward network with layers [10, 20, 6]
nn = tbm.FullyConnectedLayer(10, 20, jax.nn.relu) \
    | tbm.FullyConnectedLayer(20, 1, jax.nn.relu) \
    | tbm.FullyConnectedLayer(1, 6, jax.nn.softmax)

nn_pure = nn.pure()
nn_weights = nn.get_weights()

def loss(X, t, weights):
    return -jnp.mean(jnp.sum(t * jnp.log(nn_pure(X, *weights)), axis=1))

grad = jax.jit(jax.grad(loss, argnums=2))

# Create some random data
batch_size = 32
X = jnp.array(np.random.randn(batch_size, 10))
t = jnp.array(np.random.randn(batch_size, 6))

calculated_loss = loss(X, t, nn_weights)
calculated_gradients = grad(X, t, nn_weights)
~~~
