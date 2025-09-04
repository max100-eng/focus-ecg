# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Configuración de rutas y parámetros ---
# ⚠️ Importante: Asegúrate de que estas rutas sean correctas para tu sistema.
# Si los archivos están en una carpeta 'signal_ecg_archive' dentro de 'focus-ecg',
# la ruta relativa 'signal_ecg_archive/...' funcionará.
RUTA_PTBDB_NORMAL = 'signal_ecg_archive/ptbdb_normal.csv'
RUTA_PTBDB_ABNORMAL = 'signal_ecg_archive/ptbdb_abnormal.csv'
RUTA_MITBIH_TRAIN = 'signal_ecg_archive/mitbih_train.csv'
RUTA_MITBIH_TEST = 'signal_ecg_archive/mitbih_test.csv'

NUM_CLASSES_PTBDB = 2
NUM_CLASSES_MITBIH = 5
EPOCHS = 50
BATCH_SIZE = 32

def load_data(file_path):
    """Carga un archivo CSV y separa los datos de las etiquetas."""
    try:
        df = pd.read_csv(file_path)
        # La última columna es la etiqueta de clase
        labels = df.iloc[:, -1]
        data = df.iloc[:, :-1]
        return data, labels
    except FileNotFoundError:
        print(f"Error: El archivo {file_path} no se encontró.")
        return None, None
    except Exception as e:
        print(f"❌ Error al cargar el archivo {file_path}: {e}")
        return None, None

def build_model(input_shape, num_classes):
    """Construye un modelo básico para la clasificación de series de tiempo."""
    model = Sequential([
        # Capas para extraer características de la señal
        Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Aplanar para la capa densa
        Flatten(),
        
        # Capas densas para la clasificación
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_and_evaluate(X_train, y_train, X_val, y_val, model, model_name):
    """Entrena y evalúa el modelo, guardando el mejor resultado."""
    print(f"\n--- Entrenando el modelo: {model_name} ---")
    
    # Normalizar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Reshape para que coincida con la entrada del modelo Conv1D
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
    
    # Definir el callback para guardar el mejor modelo
    checkpoint = ModelCheckpoint(
        filepath=f'best_model_{model_name}.keras',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    # Entrenar el modelo
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_reshaped, y_val),
        callbacks=[checkpoint]
    )
    
    return history

def main():
    """Función principal para cargar, entrenar y evaluar los modelos."""
    # --- Modelo para PTBDB (Normal/Anormal) ---
    data_ptbdb_normal, labels_ptbdb_normal = load_data(RUTA_PTBDB_NORMAL)
    data_ptbdb_abnormal, labels_ptbdb_abnormal = load_data(RUTA_PTBDB_ABNORMAL)

    if data_ptbdb_normal is not None and data_ptbdb_abnormal is not None:
        X_ptbdb = pd.concat([data_ptbdb_normal, data_ptbdb_abnormal]).values
        y_ptbdb = pd.concat([labels_ptbdb_normal, labels_ptbdb_abnormal]).values
        
        X_train_ptbdb, X_val_ptbdb, y_train_ptbdb, y_val_ptbdb = train_test_split(
            X_ptbdb, y_ptbdb, test_size=0.2, random_state=42
        )
        
        input_shape_ptbdb = (X_ptbdb.shape[1], 1)
        model_ptbdb = build_model(input_shape_ptbdb, NUM_CLASSES_PTBDB)
        history_ptbdb = train_and_evaluate(X_train_ptbdb, y_train_ptbdb, X_val_ptbdb, y_val_ptbdb, model_ptbdb, "ptbdb")

    # --- Modelo para MIT-BIH (Clases Múltiples) ---
    data_mitbih_train, labels_mitbih_train = load_data(RUTA_MITBIH_TRAIN)
    data_mitbih_test, labels_mitbih_test = load_data(RUTA_MITBIH_TEST)

    if data_mitbih_train is not None and data_mitbih_test is not None:
        X_mitbih_train = data_mitbih_train.values
        y_mitbih_train = labels_mitbih_train.values
        X_mitbih_test = data_mitbih_test.values
        y_mitbih_test = labels_mitbih_test.values

        input_shape_mitbih = (X_mitbih_train.shape[1], 1)
        model_mitbih = build_model(input_shape_mitbih, NUM_CLASSES_MITBIH)
        history_mitbih = train_and_evaluate(X_mitbih_train, y_mitbih_train, X_mitbih_test, y_mitbih_test, model_mitbih, "mitbih")

if __name__ == "__main__":
    main()


