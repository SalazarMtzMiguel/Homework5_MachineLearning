# database.py
import pymysql
from config import db_config

db_config['cursorclass'] = pymysql.cursors.DictCursor

def get_connection():
    return pymysql.connect(**db_config)

def save_results_to_db(experiment_data):
    """Ejemplo: Guardar resultados del modelo en la base de datos"""
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            sql = """INSERT INTO model_results (accuracy, loss, training_time, experiment_name) VALUES (%s, %s, %s, %s)"""
            cursor.execute(sql, (
                experiment_data['accuracy'],
                experiment_data['loss'],
                experiment_data['training_time'],
                experiment_data['experiment_name']
            ))
        connection.commit()
    finally:
        connection.close()