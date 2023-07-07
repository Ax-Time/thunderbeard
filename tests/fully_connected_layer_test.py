from thunderbeard.modules import FullyConnectedLayer

import jax
import jax.numpy as jnp
import numpy as np

def test_forward():
    L = 100
    d = 15
    out = 20
    x = jnp.array(np.random.randn(d))
    layer = FullyConnectedLayer(d, out, jax.nn.relu)
    y = layer.forward(x)
    assert y.shape == (out,)

    x = jnp.array(np.random.randn(L, d))
    y = layer.forward(x)
    assert y.shape == (L, out)