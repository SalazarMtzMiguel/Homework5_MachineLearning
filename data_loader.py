import jax
import jax.numpy as jnp
import pymysql
from config import config
from sklearn.preprocessing import StandardScaler

def load_data_from_db():
    # Usa DictCursor para que los resultados sean diccionarios
    connection = pymysql.connect(cursorclass=pymysql.cursors.DictCursor, **config['db_config'])
    
    try:
        with connection.cursor() as cursor:
            # Ejemplo: Predecir el género basado en salario y antigüedad
            sql = """
            SELECT 
                e.gender, 
                s.salary, 
                TIMESTAMPDIFF(YEAR, e.hire_date, CURDATE()) AS years_of_service
            FROM 
                employees e
                JOIN salaries s ON e.emp_no = s.emp_no
            WHERE 
                s.to_date = '9999-01-01'
            LIMIT 10000  # Limitar para no sobrecargar
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            
            # Convertir a arrays
            data = []
            labels = []
            for row in results:
                labels.append(0 if row['gender'] == 'M' else 1)  # Clasificación binaria
                data.append([row['salary'], row['years_of_service']])
            
            # Convertir a JAX arrays
            data = jnp.array(data, dtype=jnp.float32)
            labels = jnp.array(labels, dtype=jnp.int32)
            
            # Normalizar datos
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            
            # Convertir a JAX arrays
            data = jnp.array(data, dtype=jnp.float32)
            
            # One-hot encoding para las etiquetas
            labels = jax.nn.one_hot(labels, num_classes=config['output_size'])
            
            return data, labels
            
    finally:
        connection.close()

def prepare_datasets():
    data, labels = load_data_from_db()
    
    # Split train/val/test (80/10/10)
    n = data.shape[0]
    n_val = n_test = n // 10
    
    indices = jax.random.permutation(jax.random.PRNGKey(config['random_seed']), n)
    
    train_data = data[indices[:n-2*n_val]]
    train_labels = labels[indices[:n-2*n_val]]
    
    val_data = data[indices[n-2*n_val:n-n_val]]
    val_labels = labels[indices[n-2*n_val:n-n_val]]
    
    test_data = data[indices[n-n_val:]]
    test_labels = labels[indices[n-n_val:]]
    
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)