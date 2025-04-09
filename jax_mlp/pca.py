import jax.numpy as jnp
from jax import random

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        # Center the data
        self.mean = jnp.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov = jnp.dot(X_centered.T, X_centered) / (X.shape[0] - 1)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = jnp.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = eigenvalues[idx]
        
        # Store components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        
    def transform(self, X):
        X_centered = X - self.mean
        return jnp.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)