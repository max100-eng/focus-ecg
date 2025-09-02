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

# --- CONFIGURACIÓN Y PARÁMETROS ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 10

# --- 1. CARGAR DATOS DESDE ARCHIVOS .NPY ---
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

# --- 2. CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO ---
def build_and_train_model():
    """
    Construye y entrena el modelo de aprendizaje por transferencia.
    """
    X_train, y_train, X_val, y_val, data_loaded = load_and_preprocess_data_from_npy()
    
    if data_loaded:
        # CONVERTIR ETIQUETAS A ÍNDICES NUMÉRICOS para calcular los pesos de clase
        y_train_indices = np.argmax(y_train, axis=1)
        
        # CALCULAR PESOS DE CLASE
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_indices),
            y=y_train_indices
        )
        class_weights = dict(enumerate(class_weights_array))
        print("✅ Pesos de clase calculados:", class_weights)
        
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

        print("\n--- INICIANDO ENTRENAMIENTO CON PESOS DE CLASE ---")
        model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            class_weight=class_weights # AQUÍ SE IMPLEMENTA EL PESO DE CLASE
        )

        # Guardar el modelo entrenado
        model_path = 'modelo_ecg_2d.h5'
        model.save(model_path)
        print(f"\n✅ Modelo 2D entrenado guardado como '{model_path}'.")

if __name__ == '__main__':
    build_and_train_model()