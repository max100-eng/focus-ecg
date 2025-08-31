# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import cv2

# --- CONFIGURACIÓN Y PARÁMETROS ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 10

# --- 1. CARGAR DATOS DIRECTAMENTE DESDE CARPETAS ---
def load_and_preprocess_data_from_folders():
    """
    Carga los datos de entrenamiento y validación desde directorios.
    La función se encarga automáticamente de redimensionar y etiquetar.
    """
    try:
        # Asegúrate de que esta ruta apunte a tu carpeta principal de dataset.
        data_dir = 'C:/Users/maram/focus-ecg/focus-ecg/ECG_DATA' 
#focus-ecg
        print("Cargando datos de entrenamiento...")
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=f'{data_dir}/train',
            labels='inferred',
            label_mode='categorical',
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            interpolation='nearest',
            shuffle=True
        )

        print("\nCargando datos de validación...")
        validation_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=f'{data_dir}/validation',
            labels='inferred',
            label_mode='categorical',
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            interpolation='nearest',
            shuffle=False
        )

        # Opcional: Estandarizar los valores de los píxeles a [0, 1]
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
        validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

        return train_dataset, validation_dataset, True

    except Exception as e:
        print(f"❌ Error al cargar el dataset. Asegúrate de que las rutas sean correctas. Error: {e}")
        return None, None, False

# --- 2. CONSTRUIR Y ENTRENAR EL MODELO ---
if __name__ == '__main__':
    train_ds, validation_ds, data_loaded = load_and_preprocess_data_from_folders()
    
    if data_loaded:
        # Construir el modelo de aprendizaje por transferencia
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
