from thunderbeard.modules import AttentionLayer, MultiHeadedAttentionLayer

import jax
import jax.numpy as jnp
import numpy as np

def test_forward_att():
    ctx_win_size = 64
    d_model = 16
    att = AttentionLayer(d_model, ctx_win_size)
    x = jnp.array(np.random.randn(ctx_win_size, d_model))
    y = att.forward(x)
    assert y.shape == (ctx_win_size, d_model)

    batch_size = 32
    X = jnp.array(np.random.randn(batch_size, ctx_win_size, d_model))
    Y = att.forward(X)
    assert Y.shape == (batch_size, ctx_win_size, d_model)

def test_forward_multi_att():
    ctx_win_size = 64
    d_model = 16
    n_heads = 4
    att = MultiHeadedAttentionLayer(n_heads, d_model, ctx_win_size)
    x = jnp.array(np.random.randn(ctx_win_size, d_model))
    y = att.forward(x)
    assert y.shape == (ctx_win_size, d_model)

    batch_size = 32
    X = jnp.array(np.random.randn(batch_size, ctx_win_size, d_model))
    Y = att.forward(X)
    assert Y.shape == (batch_size, ctx_win_size, d_model)