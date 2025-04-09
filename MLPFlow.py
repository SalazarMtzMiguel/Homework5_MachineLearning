from metaflow import FlowSpec, step, Parameter
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from jax_mlp.mlp import MLP
from jax_mlp.pca import PCA
import pymysql
import pandas as pd
import os

class MLPFlow(FlowSpec):
    n_components = Parameter('n_components', help='PCA components', default=10)
    hidden_layers = Parameter('hidden_layers', help='Hidden layer sizes', default='64,32')
    learning_rate = Parameter('lr', help='Learning rate', default=0.01)
    epochs = Parameter('epochs', help='Training epochs', default=100)
    k_folds = Parameter('k_folds', help='Number of folds', default=5)
    
    @step
    def start(self):
        # Configuración de conexión a MariaDB
        db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'root',
            'port': 3306,
            'database': 'employees',
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor
        }
        
        try:
            # Establecer conexión
            connection = pymysql.connect(**db_config)
            
            # Consulta SQL para obtener datos de empleados y salarios
            query = """
            SELECT 
                e.emp_no, 
                s.salary, 
                e.gender, 
                TIMESTAMPDIFF(YEAR, e.birth_date, CURDATE()) AS age,
                TIMESTAMPDIFF(YEAR, e.hire_date, CURDATE()) AS tenure,
                d.dept_name
            FROM employees e
            JOIN salaries s ON e.emp_no = s.emp_no
            JOIN dept_emp de ON e.emp_no = de.emp_no
            JOIN departments d ON de.dept_no = d.dept_no
            WHERE s.to_date = '9999-01-01'
            AND de.to_date = '9999-01-01'
            """
            
            # Ejecutar consulta y cargar datos en DataFrame
            with connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                df = pd.DataFrame(result)
            
            # Crear variable objetivo (salario > mediana)
            median_salary = df['salary'].median()
            df['target'] = (df['salary'] > median_salary).astype(int)
            
            # Preprocesamiento de características
            # Convertir género a numérico (M=1, F=0)
            df['gender'] = df['gender'].map({'M': 1, 'F': 0})
            
            # Codificación one-hot para departamentos
            dept_dummies = pd.get_dummies(df['dept_name'], prefix='dept')
            df = pd.concat([df, dept_dummies], axis=1)
            
            # Seleccionar características y objetivo
            self.features = df.drop(['emp_no', 'salary', 'target', 'dept_name'], axis=1).values
            self.targets = df['target'].values
            
            # Normalizar características
            self.features = (self.features - self.features.mean(axis=0)) / self.features.std(axis=0)
            
            print(f"Datos cargados correctamente. Forma de características: {self.features.shape}")
            
        except Exception as e:
            print(f"Error al conectar a MariaDB: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()
        
        self.next(self.pca_transform)
    
    
    @step
    def pca_transform(self):
        # Apply PCA
        self.pca = PCA(n_components=self.n_components)
        self.features_reduced = self.pca.fit_transform(jnp.array(self.features))
        
        # Split into k folds
        self.indices = np.arange(len(self.features_reduced))
        np.random.shuffle(self.indices)
        self.folds = np.array_split(self.indices, self.k_folds)
        
        self.next(self.train_model, foreach='folds')
    
    @step
    def train_model(self):
        # Current fold is validation set
        val_idx = self.folds[self.input]
        train_idx = np.concatenate([f for i, f in enumerate(self.folds) if i != self.input])
        
        # Split into train/val/test (60/20/20)
        test_size = int(0.2 * len(train_idx))
        test_idx = train_idx[:test_size]
        train_idx = train_idx[test_size:]
        
        X_train, y_train = self.features_reduced[train_idx], self.targets[train_idx]
        X_val, y_val = self.features_reduced[val_idx], self.targets[val_idx]
        X_test, y_test = self.features_reduced[test_idx], self.targets[test_idx]
        
        # Initialize MLP
        layer_sizes = [self.n_components] + [int(x) for x in self.hidden_layers.split(',')] + [1]
        rng_key = jax.random.PRNGKey(42)
        self.mlp = MLP(layer_sizes, rng_key)
        
        # Training loop
        for epoch in range(self.epochs):
            grads = self.mlp.grad_fn(self.mlp.params, (X_train, y_train))
            self.mlp.params = self.mlp.update(self.mlp.params, grads, self.learning_rate)
            
            # Calculate metrics
            train_preds = self.mlp.forward(self.mlp.params, X_train)
            val_preds = self.mlp.forward(self.mlp.params, X_val)
            
            # Store metrics if needed
        
        # Store final predictions for ROC/PR curves
        self.test_preds = self.mlp.forward(self.mlp.params, X_test)
        self.test_labels = y_test
        
        self.next(self.aggregate)
    
    @step
    def aggregate(self, inputs):
        # Collect all test predictions and labels
        all_preds = np.concatenate([inp.test_preds for inp in inputs])
        all_labels = np.concatenate([inp.test_labels for inp in inputs])
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        
        # Compute Precision-Recall curve
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        pr_auc = auc(recall, precision)
        
        # Plot curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        
        plt.savefig('results/curves.png')
        plt.close()
        
        self.next(self.end)
    
    @step
    def end(self):
        pass

if __name__ == '__main__':
    MLPFlow()