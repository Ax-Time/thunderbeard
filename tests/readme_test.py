from thunderbeard import modules as tbm

import jax
import jax.numpy as jnp
import numpy as np

def test_ff():
    # Create a feedforward network with layers [10, 20, 2]
    nn = tbm.FullyConnectedLayer(10, 20, jax.nn.relu) \
        | tbm.FullyConnectedLayer(20, 1, jax.nn.relu) \
        | tbm.FullyConnectedLayer(1, 2, jax.nn.softmax)
    
    # Create some random data
    x = jnp.array(np.random.randn(10))
    y = nn.forward(x)
    assert y.shape == (2,)

def test_transformer_encoder():
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
    Y = encoder.forward(X)
    assert Y.shape == (batch_size, ctx_win_size, d_model)

    