# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import cv2
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN Y PARÁMETROS ---
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 4 # Ritmo Normal, IAM, Arritmia, Bloqueo de Branca
EPOCHS = 10
BATCH_SIZE = 32

# --- 1. FUNCIÓN DE CONVERSIÓN DE SEÑAL A IMAGEN ---
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

# --- 2. CARGA Y PREPARACIÓN DE DATOS ---
def load_and_preprocess_data():
    """
    Carga tus datos de entrenamiento y validación desde archivos .npy.
    Asegúrate de que las rutas y los nombres de los archivos sean correctos.
    """
    try:
        # Reemplaza con la ruta real a tus archivos de datos.
       signals = np.load('ecg_signals.npy')
        labels = np.load('ecg_labels.npy')

        X_train_signals, X_val_signals, y_train, y_val = train_test_split(
            signals, labels, test_size=0.2, random_state=42
        )

        # Convertir señales 1D a imágenes 2D
        X_train_images = np.array([create_ecg_image_from_signal(s.flatten()) for s in X_train_signals])
        X_val_images = np.array([create_ecg_image_from_signal(s.flatten()) for s in X_val_signals])

        print(f"Datos de entrenamiento: {len(X_train_images)} imágenes")
        print(f"Datos de validación: {len(X_val_images)} imágenes")
        
        return X_train_images, y_train, X_val_images, y_val

    except FileNotFoundError:
        print("❌ Error: No se encontraron los archivos de datos. Asegúrate de que las rutas sean correctas.")
        return None, None, None, None

# --- 3. CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO ---
if __name__ == '__main__':
    X_train, y_train, X_val, y_val = load_and_preprocess_data()
    
    if X_train is not None:
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
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val)
        )

        # Guardar el modelo entrenado
        model_path = 'modelo_ecg_2d.h5'
        model.save(model_path)
        print(f"\n✅ Modelo 2D entrenado guardado como '{model_path}'.")
