# trainer.py
import jax
import jax.numpy as jnp
from model import cross_entropy_loss, compute_accuracy

def manual_gradient(params, x, y):
    # Finite difference gradient approximation
    eps = 1e-4
    grad = []
    for layer_idx, layer in enumerate(params):  # Iterar con índices
        w_grad = jnp.zeros_like(layer['weights'])
        b_grad = jnp.zeros_like(layer['biases'])
        
        # Compute gradients for weights
        for i in range(layer['weights'].shape[0]):
            for j in range(layer['weights'].shape[1]):
                # Copiar parámetros para modificar solo la capa actual
                params_plus = [dict(l) for l in params]  # Crear copias profundas
                params_minus = [dict(l) for l in params]
                
                # Modificar los pesos de la capa actual
                params_plus[layer_idx]['weights'] = params_plus[layer_idx]['weights'].at[i, j].add(eps)
                params_minus[layer_idx]['weights'] = params_minus[layer_idx]['weights'].at[i, j].add(-eps)
                
                # Calcular las pérdidas
                loss_plus = cross_entropy_loss(params_plus, x, y)
                loss_minus = cross_entropy_loss(params_minus, x, y)
                
                # Calcular el gradiente
                w_grad = w_grad.at[i, j].set((loss_plus - loss_minus) / (2 * eps))
        
        # Compute gradients for biases
        for j in range(layer['biases'].shape[0]):
            # Copiar parámetros para modificar solo la capa actual
            params_plus = [dict(l) for l in params]  # Crear copias profundas
            params_minus = [dict(l) for l in params]
            
            # Modificar los biases de la capa actual
            params_plus[layer_idx]['biases'] = params_plus[layer_idx]['biases'].at[j].add(eps)
            params_minus[layer_idx]['biases'] = params_minus[layer_idx]['biases'].at[j].add(-eps)
            
            # Calcular las pérdidas
            loss_plus = cross_entropy_loss(params_plus, x, y)
            loss_minus = cross_entropy_loss(params_minus, x, y)
            
            # Calcular el gradiente
            b_grad = b_grad.at[j].set((loss_plus - loss_minus) / (2 * eps))
        
        grad.append({'weights': w_grad, 'biases': b_grad})
    
    return grad

def update_params(params, grads, learning_rate):
    return [{
        'weights': layer['weights'] - learning_rate * grad['weights'],
        'biases': layer['biases'] - learning_rate * grad['biases']
    } for layer, grad in zip(params, grads)]

def train_epoch(params, x_train, y_train, batch_size, learning_rate):
    num_batches = x_train.shape[0] // batch_size
    epoch_loss = 0.0
    
    for i in range(num_batches):
        batch_x = x_train[i*batch_size:(i+1)*batch_size]
        batch_y = y_train[i*batch_size:(i+1)*batch_size]
        
        grads = manual_gradient(params, batch_x, batch_y)
        params = update_params(params, grads, learning_rate)
        batch_loss = cross_entropy_loss(params, batch_x, batch_y)
        epoch_loss += batch_loss
    
    return params, epoch_loss / num_batches

def evaluate(params, x, y):
    # Asegúrate de que las etiquetas sean compatibles con binary_cross_entropy
    loss = binary_cross_entropy(params, x, y)
    accuracy = compute_accuracy(params, x, y)
    return loss, accuracy