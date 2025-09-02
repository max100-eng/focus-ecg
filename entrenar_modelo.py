# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURACIÓN Y PARÁMETROS ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
# Puedes poner un número de épocas alto, el EarlyStopping lo detendrá
EPOCHS = 50 

# --- 1. CARGAR DATOS DIRECTAMENTE DESDE CARPETAS ---
def load_and_preprocess_data_from_folders():
    """
    Carga los datos de entrenamiento y validación desde directorios.
    """
    try:
        # Asegúrate de que esta ruta apunte a tu carpeta principal de dataset.
        data_dir = 'C:/Users/maram/focus-ecg/ECG_DATA'
        
        if not os.path.isdir(data_dir):
            print(f"❌ Error: El directorio '{data_dir}' no existe.")
            return None, None, False

        print("Cargando datos de entrenamiento desde carpetas...")
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=f'{data_dir}/train',
            labels='inferred',
            label_mode='categorical',
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            interpolation='nearest',
            shuffle=True
        )

        print("\nCargando datos de validación desde carpetas...")
        validation_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=f'{data_dir}/validation',
            labels='inferred',
            label_mode='categorical',
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            interpolation='nearest',
            shuffle=False
        )

        normalization_layer = tf.keras.layers.Rescaling(1./255)
        train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
        validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

        return train_dataset, validation_dataset, True

    except Exception as e:
        print(f"❌ Error al cargar el dataset desde carpetas. Error: {e}")
        return None, None, False

# --- 2. CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO CON AJUSTE FINO ---
def build_and_train_model(train_ds, validation_ds):
    """
    Construye y entrena el modelo de aprendizaje por transferencia con Fine-Tuning.
    """
    print("\n--- FASE 1: ENTRENANDO EL CABEZAL DE CLASIFICACIÓN ---")
    
    # Crea el modelo base VGG16 y congela todas sus capas
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    base_model.trainable = False

    # Crea el cabezal de clasificación
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compila y entrena el modelo (solo el cabezal)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    print("\n--- INICIANDO ENTRENAMIENTO (FASE 1) ---")
    model.fit(
        train_ds,
        epochs=10, # Entrenamos 10 épocas para estabilizar el cabezal
        validation_data=validation_ds,
        callbacks=[early_stopping]
    )

    print("\n--- FASE 2: AJUSTE FINO DEL MODELO BASE ---")
    
    # Descongela algunas capas superiores de VGG16
    base_model.trainable = True
    for layer in base_model.layers[:-4]: # Descongela las últimas 4 capas convolucionales
        layer.trainable = False

    # Vuelve a compilar el modelo con un learning rate muy bajo
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5), # <--- Tasa de aprendizaje más baja
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    print("\n--- INICIANDO ENTRENAMIENTO (FASE 2) ---")
    model.fit(
        train_ds,
        epochs=EPOCHS, # Continúa el entrenamiento por más épocas
        validation_data=validation_ds,
        callbacks=[early_stopping]
    )

    # --- Evaluación final del modelo ---
    print("\n--- EVALUANDO EL MODELO FINAL ---")
    loss, accuracy = model.evaluate(validation_ds)
    print(f"✅ Precisión final del modelo: {accuracy:.4f}")
    print(f"✅ Pérdida final del modelo: {loss:.4f}")

    # Guardar el modelo entrenado
    model_path = 'modelo_ecg_2d.h5'
    model.save(model_path)
    print(f"\n✅ Modelo 2D entrenado guardado como '{model_path}'.")

if __name__ == '__main__':
    train_ds, validation_ds, data_loaded = load_and_preprocess_data_from_folders()
    
    if data_loaded:
        build_and_train_model(train_ds, validation_ds)