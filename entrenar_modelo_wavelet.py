import pywt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import os
import cv2

# --- CONFIGURACIÓN Y PARÁMETROS ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 10 

# --- 1. FUNCIÓN PARA APLICAR LA TRANSFORMADA DE WAVELET ---
def apply_wavelet_transform(image):
    """
    Aplica la transformada de wavelet 'db1' a una imagen en escala de grises.
    Devuelve los coeficientes de aproximación.
    """
    # Convertir a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Aplicar la transformada de wavelet (de Haar) en 2 niveles
    coeffs2 = pywt.dwt2(gray_image, 'db1')
    ll, (lh, hl, hh) = coeffs2
    
    # La aproximación (LL) contiene las características de baja frecuencia,
    # que son útiles para el reconocimiento de patrones de ondas.
    return ll

# --- 2. CARGAR Y PREPROCESAR DATOS CON WAVELET ---
def load_and_preprocess_data_with_wavelet():
    """
    Carga los datos desde carpetas y aplica la transformada de wavelet.
    """
    try:
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

        # Aplicar la transformada de wavelet a los datasets
        def preprocess_with_wavelet(image, label):
            wavelet_image = tf.numpy_function(
                func=apply_wavelet_transform,
                inp=[image],
                Tout=tf.float32
            )
            # Normalizar los coeficientes y redimensionar.
            wavelet_image = tf.expand_dims(wavelet_image, axis=-1)
            wavelet_image = tf.image.resize(wavelet_image, (112, 112))
            wavelet_image.set_shape([112, 112, 1])
            return wavelet_image / 255.0, label

        train_dataset = train_dataset.map(preprocess_with_wavelet)
        validation_dataset = validation_dataset.map(preprocess_with_wavelet)

        return train_dataset, validation_dataset, True

    except Exception as e:
        print(f"❌ Error al cargar el dataset. Error: {e}")
        return None, None, False

# --- 3. CONSTRUIR Y ENTRENAR EL MODELO DE PREENTRENAMIENTO ---
def build_and_train_wavelet_model(train_ds, validation_ds):
    """
    Construye y entrena un modelo simple para datos de wavelet.
    """
    print("\n--- CONSTRUYENDO MODELO DE PREENTRENAMIENTO CON WAVELET ---")
    model = Sequential([
        # Corregimos la forma de entrada para que coincida con la imagen de wavelet
        Conv2D(32, (3, 3), activation='relu', input_shape=(112, 112, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    print("\n--- INICIANDO ENTRENAMIENTO DEL MODELO WAVELET ---")
    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=validation_ds
    )

    # Guardar el modelo pre-entrenado
    model_path = 'modelo_preentrenado_wavelet.h5'
    model.save(model_path)
    print(f"\n✅ Modelo pre-entrenado guardado como '{model_path}'.")
    return model

if __name__ == '__main__':
    train_ds, validation_ds, data_loaded = load_and_preprocess_data_with_wavelet()
    
    if data_loaded:
        build_and_train_wavelet_model(train_ds, validation_ds)


