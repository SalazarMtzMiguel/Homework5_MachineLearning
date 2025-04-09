from metaflow import FlowSpec, step, Parameter
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from jax_mlp.mlp import MLP
from jax_mlp.pca import PCA
import pymysql
import pandas as pd
import os
from sklearn.model_selection import train_test_split

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
            features_df = df.drop(['emp_no', 'salary', 'target', 'dept_name'], axis=1)
            
            # Convertir todas las columnas a numérico
            features_df = features_df.apply(pd.to_numeric, errors='coerce')
            
            # Eliminar filas con NaN (si las hay)
            features_df = features_df.dropna()
            targets = df.loc[features_df.index, 'target'].values
            
            # Normalizar características usando StandardScaler
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features_df.values)
            self.targets = targets
            
            print(f"Datos cargados correctamente. Forma de características: {self.features.shape}")
            print(f"Distribución de clases: {np.bincount(self.targets)}")
            
        except Exception as e:
            print(f"Error durante el procesamiento: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()
        
        self.next(self.pca_transform)
    
    @step
    def pca_transform(self):
        # Aplicar PCA
        self.pca = PCA(n_components=self.n_components)
        self.features_reduced = self.pca.fit_transform(jnp.array(self.features))
        
        # Dividir en k folds
        self.indices = np.arange(len(self.features_reduced))
        np.random.shuffle(self.indices)
        self.folds = np.array_split(self.indices, self.k_folds)
        
        self.next(self.train_model, foreach='folds')
    
    @step
    def train_model(self):
        # En Metaflow, para steps foreach, self.input es el valor actual de la iteración
        # No necesitamos usar self.folds[self.input]
        val_idx = self.input  # Esto ya son los índices del fold actual
        
        # Los otros folds son para entrenamiento
        train_idx = np.concatenate([f for f in self.folds if not np.array_equal(f, val_idx)])
    
        # Split into train/val/test (60/20/20)
        test_size = int(0.2 * len(train_idx))
        test_idx = train_idx[:test_size]
        train_idx = train_idx[test_size:]
        
        X_train = self.features_reduced[train_idx]
        y_train = self.targets[train_idx]
        X_val = self.features_reduced[val_idx]
        y_val = self.targets[val_idx]
        X_test = self.features_reduced[test_idx]
        y_test = self.targets[test_idx]
        
        # Initialize MLP
        layer_sizes = [self.n_components] + [int(x) for x in self.hidden_layers.split(',')] + [1]
        rng_key = jax.random.PRNGKey(42)
        self.mlp = MLP(layer_sizes, rng_key)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Forward pass y cálculo de gradientes
            grads = self.mlp.grad_fn(self.mlp.params, (X_train, y_train))
            self.mlp.params = self.mlp.update(self.mlp.params, grads, self.learning_rate)
            
            # Calcular métricas
            train_loss = self.mlp.loss_fn(self.mlp.params, (X_train, y_train))
            val_loss = self.mlp.loss_fn(self.mlp.params, (X_val, y_val))
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Store final predictions for ROC/PR curves
        self.test_preds = self.mlp.forward(self.mlp.params, X_test)
        self.test_labels = y_test
        self.train_losses = train_losses
        self.val_losses = val_losses
        
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
        plt.figure(figsize=(15, 5))
        
        # Plot training curves
        plt.subplot(1, 3, 1)
        for inp in inputs:
            plt.plot(inp.train_losses, label='Train Loss')
            plt.plot(inp.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # ROC Curve
        plt.subplot(1, 3, 2)
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # Precision-Recall Curve
        plt.subplot(1, 3, 3)
        plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        
        # Create results directory if not exists
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/performance_curves.png')
        plt.close()
        
        print(f"Evaluación final - ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        
        self.next(self.end)
    
    @step
    def end(self):
        print("Flujo completado exitosamente")

if __name__ == '__main__':
    MLPFlow()