from thunderbeard.modules import FullyConnectedLayerWith2DInput

import jax
import jax.numpy as jnp
import numpy as np

def test_forward():
    L = 100
    d = 15
    out = 20
    x = jnp.array(np.random.randn(L, d))
    layer = FullyConnectedLayerWith2DInput((L, d), out, jax.nn.relu)
    y = layer.forward(x)
    assert y.shape == (L, out)

    batch_size = 32
    X = jnp.array(np.random.randn(batch_size, L, d))
    Y = layer.forward(X)
    assert Y.shape == (batch_size, L, out)