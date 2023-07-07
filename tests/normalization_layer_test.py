from thunderbeard.modules import NormalizationLayer

import jax.numpy as jnp
import numpy as np

def test_forward():
    m = 30
    n = 40
    X = jnp.array(np.random.randn(m, n))
    layer = NormalizationLayer((m, n), (m, n))
    Y = layer.forward(X)
    assert Y.shape == (m, n)

    batch_size = 32
    X = jnp.array(np.random.randn(batch_size, m, n))
    Y = layer.forward(X)
    assert Y.shape == (batch_size, m, n)