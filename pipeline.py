# pipeline.py
import jax  # Importar jax para usar jax.random.PRNGKey
from data_loader import prepare_datasets
from model import init_mlp_params, mlp_forward, compute_accuracy
from trainer import train_epoch, evaluate
from utils import plot_roc_curve, plot_precision_recall_curve, k_fold_cross_validation
from config import config

def main():
    # Preparar datasets
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = prepare_datasets()
    
    # Inicializar modelo
    input_size = train_data.shape[1]
    layer_sizes = [input_size] + config['hidden_layers'] + [config['output_size']]
    key = jax.random.PRNGKey(config['random_seed'])
    params = init_mlp_params(layer_sizes, key)
    
    # Entrenamiento
    for epoch in range(config['epochs']):
        params, train_loss = train_epoch(params, train_data, train_labels, config['batch_size'], config['learning_rate'])
        val_loss, val_acc = evaluate(params, val_data, val_labels)
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Evaluación final
    test_loss, test_acc = evaluate(params, test_data, test_labels)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
if __name__ == "__main__":
    main()