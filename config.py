config = {
    'db_config': {
        'host': 'localhost',
        'user': 'root',
        'password': 'root',
        'port': 3306,
        'database': 'employees',
        'charset': 'utf8mb4'
    },
    'hidden_layers': [64, 32],
    'output_size': 2,  # Para clasificación binaria
    'learning_rate': 0.01,
    'epochs': 10,
    'batch_size': 32,
    'k_folds': 5,
    'random_seed': 42
}