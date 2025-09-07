import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

# --- Cargar ambos modelos ---
@st.cache_resource
def load_models():
    # Cargar el modelo de imágenes
    model_2d_path = "modelo_ecg_2d.h5"
    try:
        model_2d = tf.keras.models.load_model(model_2d_path)
        st.success("✅ Modelo de imágenes (2D) cargado exitosamente.")
        return model_2d
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo de imágenes (2D): {e}")
        st.info("La clasificación visual no estará disponible.")
        return None

# Importar el script de análisis de señales
try:
    from entrenar_modelo_wavelet import analyze_ecg_from_image_path
    st.success("✅ Modelo de análisis de señales (Wavelet) cargado.")
except ImportError:
    st.error("❌ Error: No se encontró el script 'entrenar_modelo_wavelet.py'. Asegúrate de que esté en la misma carpeta.")
    analyze_ecg_from_image_path = None

model_2d = load_models()

# --- Funciones de análisis y preprocesamiento ---
def preprocess_image_for_model(image_data):
    """Carga y preprocesa una imagen de ECG para el modelo de imágenes (2D)."""
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array.astype('float32') / 255.0
        return image_array
    except Exception as e:
        st.error(f"❌ Error al preprocesar la imagen: {e}")
        return None

def generate_ecg_graph():
    """Genera un gráfico simulado de un trazado de ECG."""
    t = np.linspace(0, 5, 500)
    p_wave = 0.1 * np.exp(-100 * (t - 0.1)**2)
    qrs_complex = -0.6 * np.exp(-1000 * (t - 0.2)**2) + 1.2 * np.exp(-1000 * (t - 0.25)**2) - 0.2 * np.exp(-1000 * (t - 0.3)**2)
    t_wave = 0.3 * np.exp(-80 * (t - 0.45)**2)
    ecg_wave = p_wave + qrs_complex + t_wave
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, ecg_wave, color='red')
    ax.set_title("Trazado ECG Simulado")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Voltaje (mV)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f7f7f7')
    plt.tight_layout()
    return fig

# --- Interfaz de Streamlit ---
def main():
    st.title("CardioSense")
    st.subheader("Análisis de Electrocardiogramas (ECG) mediante IA")
    
    st.sidebar.title("Configuración")
    st.sidebar.markdown("Sube una imagen de ECG para analizar.")
    
    uploaded_file = st.file_uploader("Sube una imagen de ECG...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.write("---")
        st.subheader("Imagen subida")
        image_data = uploaded_file.getvalue()
        st.image(image_data, caption=uploaded_file.name, use_container_width=True)
        
        st.write("---")
        st.subheader("Resultados del análisis")
        
        # --- 1. Análisis por el modelo de Imágenes (VGG16) ---
        st.markdown("### 👁️ Clasificación Visual (por `modelo_ecg_2d.h5`)")
        if model_2d:
            try:
                # Preprocesar la imagen para el modelo de clasificación
                data_for_prediction = preprocess_image_for_model(image_data)
                
                # Realizar la predicción
                predictions = model_2d.predict(data_for_prediction)
                predicted_class_index = np.argmax(predictions[0])
                
                # Simular la etiqueta de diagnóstico (ejemplo)
                # En tu código real, mapearías el índice a una etiqueta de clase
                class_labels = ["Ritmo Normal", "Arritmia"]
                visual_diagnosis = class_labels[predicted_class_index]
                
                st.info(f"El modelo de imágenes clasifica el ECG como: **{visual_diagnosis}**")
                
                st.write("---")
            except Exception as e:
                st.error(f"❌ Error al ejecutar el modelo de imágenes: {e}")
        else:
            st.info("El modelo de imágenes no está disponible. No se puede realizar la clasificación visual.")
        
        # --- 2. Análisis por el modelo de Señales (Wavelet) ---
        st.markdown("### 📈 Análisis Numérico (por tu modelo de `wavelet`)")
        if analyze_ecg_from_image_path:
            try:
                # Guardar la imagen temporalmente para que tu modelo la lea
                temp_image_path = "temp_ecg_image.jpg"
                with open(temp_image_path, "wb") as f:
                    f.write(image_data)
                
                # Llamar a la función de análisis de tu modelo de wavelet
                analysis_results = analyze_ecg_from_image_path(temp_image_path)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Ritmo Cardíaco (bpm)", value=analysis_results.get("heartRate", "N/A"))
                with col2:
                    st.metric(label="Diagnóstico Automático", value=analysis_results.get("autoDiagnosis", "N/A"))
                    
                st.write("#### Datos Clave del ECG")
                ecg_intervals = analysis_results.get("ecgIntervals")
                if ecg_intervals:
                    df = pd.DataFrame(ecg_intervals)
                    st.dataframe(df.set_index('interval'))
                else:
                    st.info("No se encontraron datos de intervalos en los resultados del modelo.")
                
                # Eliminar la imagen temporal
                os.remove(temp_image_path)

            except Exception as e:
                st.error(f"❌ Error al ejecutar tu modelo de análisis de señales: {e}")
                st.info("No se pudo obtener el análisis numérico. Comprueba tu script `entrenar_modelo_wavelet.py`.")
        else:
            st.info("El modelo de análisis de señales no está disponible. No se puede realizar el análisis numérico.")
        
        # --- Gráfico simulado del trazado ECG ---
        st.write("---")
        st.markdown("### 📊 Trazado ECG Simulado")
        st.write("Este gráfico representa un trazado de ECG simulado para fines de demostración.")
        fig = generate_ecg_graph()
        st.pyplot(fig)
        
        st.success("Análisis completo. Consulta ambos resultados.")
        st.warning("⚠️ **Aviso Importante**: Esta es una herramienta experimental. Consulta siempre a un profesional de la salud para un diagnóstico médico.")

# --- Ejecutar la aplicación ---
if __name__ == "__main__":
    main()