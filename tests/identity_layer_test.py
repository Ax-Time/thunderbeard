from thunderbeard.modules import IdentityLayer

import jax.numpy as jnp
import numpy as np

def test_forward():
    L = 100
    d = 15
    x = jnp.array(np.random.randn(d))
    layer = IdentityLayer(d, d)
    y = layer.forward(x)
    assert y.shape == (d,)

    x = jnp.array(np.random.randn(L, d))
    y = layer.forward(x)
    assert y.shape == (L, d)