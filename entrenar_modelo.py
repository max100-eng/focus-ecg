# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import cv2
from sklearn.model_selection import train_test_split
import os

# --- CONFIGURACIÓN Y PARÁMETROS ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 10

# --- 1. CARGAR DATOS DIRECTAMENTE DESDE CARPETAS (OPCIÓN A) ---
def load_and_preprocess_data_from_folders():
    """
    Carga los datos de entrenamiento y validación desde directorios.
    """
    try:
        data_dir = 'C:/Users/maram/focus-ecg/focus-ecg/ECG_DATA'
        
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

# --- 2. CARGAR DATOS DESDE ARCHIVOS .NPY (OPCIÓN B) ---
def create_ecg_image_from_signal(signal_1d, img_size=IMAGE_SIZE):
    """
    Convierte una señal de ECG 1D en una imagen 2D en escala de grises.
    """
    img = np.zeros(img_size, dtype=np.uint8)
    scaled_signal = (signal_1d - np.min(signal_1d)) / (np.max(signal_1d) - np.min(signal_1d))
    scaled_signal = (scaled_signal * (img_size[0] - 1)).astype(int)
    
    for i, y in enumerate(scaled_signal):
        if i < img_size[1]:
            img[y, i] = 255
            
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img_rgb

def load_and_preprocess_data_from_npy():
    """
    Carga tus datos de entrenamiento y validación desde archivos .npy.
    """
    try:
        signals = np.load('ecg_signals.npy')
        labels = np.load('ecg_labels.npy')

        X_train_signals, X_val_signals, y_train, y_val = train_test_split(
            signals, labels, test_size=0.2, random_state=42
        )

        X_train_images = np.array([create_ecg_image_from_signal(s.flatten()) for s in X_train_signals])
        X_val_images = np.array([create_ecg_image_from_signal(s.flatten()) for s in X_val_signals])

        print(f"Datos de entrenamiento: {len(X_train_images)} imágenes")
        print(f"Datos de validación: {len(X_val_images)} imágenes")
        
        return X_train_images, y_train, X_val_images, y_val, True

    except FileNotFoundError:
        print("❌ Error: No se encontraron los archivos .npy. Asegúrate de que las rutas sean correctas.")
        return None, None, None, None, False

# --- 3. CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO ---
def build_and_train_model(train_ds, validation_ds):
    """
    Construye y entrena el modelo de aprendizaje por transferencia.
    """
    print("\n--- CONSTRUYENDO MODELO ---")
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    print("\n--- INICIANDO ENTRENAMIENTO ---")
    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=validation_ds
    )

    # Guardar el modelo entrenado
    model_path = 'modelo_ecg_2d.h5'
    model.save(model_path)
    print(f"\n✅ Modelo 2D entrenado guardado como '{model_path}'.")

if __name__ == '__main__':
    # Elige tu método de carga de datos aquí.
    # OPCIÓN A: Cargar desde carpetas (si ya tienes imágenes)
    train_ds, validation_ds, data_loaded = load_and_preprocess_data_from_folders()

    # OPCIÓN B: Cargar desde archivos .npy (si tienes señales 1D)
    # train_data, y_train, val_data, y_val, data_loaded = load_and_preprocess_data_from_npy()
    # Si eliges esta opción, necesitarás adaptar el código en la función de entrenamiento
    # para usar los arrays de numpy en lugar de los objetos de dataset.

    if data_loaded:
        # Si elegiste la opción A, usa build_and_train_model(train_ds, validation_ds)
        # Si elegiste la opción B, la función de entrenamiento necesita ser ligeramente adaptada
        # para usar arrays de numpy: model.fit(train_data, y_train, validation_data=(val_data, y_val))
        build_and_train_model(train_ds, validation_ds)
