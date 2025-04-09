# model.py
import jax
import jax.numpy as jnp
from config import config

def init_mlp_params(layer_sizes, key):
    keys = jax.random.split(key, len(layer_sizes)-1)
    params = []
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        w = jax.random.normal(keys[i], (n_in, n_out)) * jnp.sqrt(2/n_in)
        b = jnp.zeros(n_out)
        params.append({'weights': w, 'biases': b})
    return params

def mlp_forward(params, x):
    *hidden_layers, last_layer = params
    for layer in hidden_layers:
        x = jnp.dot(x, layer['weights']) + layer['biases']
        x = jax.nn.relu(x)
    x = jnp.dot(x, last_layer['weights']) + last_layer['biases']
    return jax.nn.sigmoid(x)  # Cambiado a sigmoid para clasificación binaria

def binary_cross_entropy(params, x, y_true):
    y_pred = mlp_forward(params, x)
    return -jnp.mean(y_true * jnp.log(y_pred + 1e-8) + (1-y_true) * jnp.log(1-y_pred + 1e-8))

def cross_entropy_loss(params, x, y_true):
    # Cambiar a binary_cross_entropy para clasificación binaria
    return binary_cross_entropy(params, x, y_true)

def compute_accuracy(params, x, y_true):
    y_pred = mlp_forward(params, x)
    predicted_class = jnp.argmax(y_pred, axis=1)
    true_class = jnp.argmax(y_true, axis=1)
    return jnp.mean(predicted_class == true_class)