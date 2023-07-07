from thunderbeard.modules import FullyConnectedLayer

import jax
import jax.numpy as jnp
import numpy as np

def test_or():
    layer1 = FullyConnectedLayer(10, 20, jax.nn.relu)
    layer2 = FullyConnectedLayer(20, 30, jax.nn.relu)
    layer3 = FullyConnectedLayer(30, 4, jax.nn.softmax)
    model = layer1 | layer2 | layer3
    x = jnp.array(np.random.randn(10))
    y = model.forward(x)
    assert y.shape == (4,)

    x = jnp.array(np.random.randn(100, 10))
    y = model.forward(x)
    assert y.shape == (100, 4)

def test_or_gradient():
    layer1 = FullyConnectedLayer(10, 20, jax.nn.relu)
    layer2 = FullyConnectedLayer(20, 30, jax.nn.relu)
    layer3 = FullyConnectedLayer(30, 4, jax.nn.softmax)
    model = layer1 | layer2 | layer3
    x = jnp.array(np.random.randn(100, 10))
    t = jnp.array(np.random.randn(100, 4))
    def loss(x, t, params):
        return -jnp.mean(jnp.sum(t * jnp.log(model.pure()(x, *params)), axis=1))
    grad = jax.jit(jax.grad(loss, argnums=2))
    for param, gradient in zip(model.get_weights(), grad(x, t, model.get_weights())):
        assert param.shape == gradient.shape

def test_num_params():
    layer1 = FullyConnectedLayer(10, 20, jax.nn.relu)
    layer2 = FullyConnectedLayer(20, 30, jax.nn.relu)
    layer3 = FullyConnectedLayer(30, 4, jax.nn.softmax)
    model = layer1 | layer2 | layer3
    assert model.num_params() == 10 * 20 + 20 + 20 * 30 + 30 + 30 * 4 + 4

def test_combine():
    merge_operation = lambda x, y: x + y

    layer1 = FullyConnectedLayer(10, 20, jax.nn.relu)
    layer2 = FullyConnectedLayer(20, 30, jax.nn.relu)
    layer3 = FullyConnectedLayer(20, 30, jax.nn.relu) 
    model = layer1 | (layer2.combine_in_parallel(layer3, merge_operation))

    x = jnp.array(np.random.randn(10))
    y = model.forward(x)
    assert y.shape == (30,)

    x = jnp.array(np.random.randn(100, 10))
    y = model.forward(x)
    assert y.shape == (100, 30)