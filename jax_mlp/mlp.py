import jax
import jax.numpy as jnp
from jax import random
from functools import partial

class MLP:
    def __init__(self, layer_sizes, rng_key):
        self.layer_sizes = layer_sizes
        self.params = self.init_params(rng_key)
        
    def init_params(self, rng_key):
        keys = random.split(rng_key, len(self.layer_sizes)-1)
        params = []
        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            W_key, b_key = random.split(keys[i])
            W = random.normal(W_key, (n_in, n_out)) * jnp.sqrt(2/n_in)
            b = jnp.zeros(n_out)
            params.append((W, b))
        return params
    
    def forward(self, params, x):
        for W, b in params[:-1]:
            x = jnp.dot(x, W) + b
            x = jnp.tanh(x)
        W_last, b_last = params[-1]
        x = jnp.dot(x, W_last) + b_last
        return x
    
    def loss_fn(self, params, batch):
        inputs, targets = batch
        preds = self.forward(params, inputs)
        return jnp.mean((preds - targets) ** 2)
    
    # Manual gradient computation
    def grad_fn(self, params, batch):
        grad = []
        eps = 1e-4
        for i, (W, b) in enumerate(params):
            W_grad = jnp.zeros_like(W)
            b_grad = jnp.zeros_like(b)
            
            # Compute W gradient
            for j in range(W.shape[0]):
                for k in range(W.shape[1]):
                    W_perturbed = W.at[j,k].add(eps)
                    params_perturbed = params[:i] + [(W_perturbed, b)] + params[i+1:]
                    loss_plus = self.loss_fn(params_perturbed, batch)
                    
                    W_perturbed = W.at[j,k].add(-eps)
                    params_perturbed = params[:i] + [(W_perturbed, b)] + params[i+1:]
                    loss_minus = self.loss_fn(params_perturbed, batch)
                    
                    W_grad = W_grad.at[j,k].set((loss_plus - loss_minus) / (2 * eps))
            
            # Compute b gradient
            for j in range(b.shape[0]):
                b_perturbed = b.at[j].add(eps)
                params_perturbed = params[:i] + [(W, b_perturbed)] + params[i+1:]
                loss_plus = self.loss_fn(params_perturbed, batch)
                
                b_perturbed = b.at[j].add(-eps)
                params_perturbed = params[:i] + [(W, b_perturbed)] + params[i+1:]
                loss_minus = self.loss_fn(params_perturbed, batch)
                
                b_grad = b_grad.at[j].set((loss_plus - loss_minus) / (2 * eps))
            
            grad.append((W_grad, b_grad))
        
        return grad
    
    def update(self, params, grads, lr=0.01):
        return [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(params, grads)]