import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import pandas as pd # Importa pandas para la tabla de intervalos
import cv2

# --- Cargar modelos ---
@st.cache_resource
def load_models():
    model_2d_path = "modelo_ecg_2d.h5"
    try:
        model_2d = tf.keras.models.load_model(model_2d_path)
        st.success("✅ Modelo de imágenes (2D) cargado exitosamente.")
        return model_2d
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo de imágenes (2D): [Errno 2] No such file or directory: '{model_2d_path}'. Asegúrate de que el archivo esté en la misma carpeta.")
        return None

model_2d = load_models()

# --- Funciones de análisis y preprocesamiento ---
def simulate_ecg_analysis():
    """
    Simula el análisis de los intervalos y métricas del ECG.
    """
    return {
        "heartRate": 72,
        "autoDiagnosis": "Ritmo Sinusal Normal",
        "ecgIntervals": [
            {"interval": "PR", "duration": 160, "normalRange": "120-200"},
            {"interval": "QRS", "duration": 90, "normalRange": "80-120"},
            {"interval": "QT", "duration": 380, "normalRange": "350-440"},
            {"interval": "QTc", "duration": 420, "normalRange": "340-440"}
        ]
    }

def preprocess_image(image_data):
    """Carga y preprocesa una imagen de ECG para el modelo desde datos binarios."""
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
    # Simula un trazado de onda de ECG (P, QRS, T)
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
        
        # Simular que el modelo existe para mostrar la interfaz
        if model_2d:
            st.write("---")
            st.subheader("Resultados del análisis:")
            
            # Obtener datos simulados de análisis de intervalos
            analysis_results = simulate_ecg_analysis()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Ritmo Cardíaco (bpm)", value=analysis_results["heartRate"])
            with col2:
                st.metric(label="Diagnóstico Automático", value=analysis_results["autoDiagnosis"])
                
            st.warning("⚠️ Recuerda: Estos resultados son una **simulación**. El análisis de ritmo e intervalos requiere algoritmos complejos que no están presentes en este modelo de clasificación de imágenes. Consulta a un profesional de la salud.")
            
            st.write("### Datos Clave del ECG (Simulado)")
            
            df = pd.DataFrame(analysis_results["ecgIntervals"])
            st.dataframe(df.set_index('interval'))
            
            # --- Generar el gráfico del trazado del ECG ---
            st.write("---")
            st.write("### Trazado ECG Simulado")
            st.write("Este gráfico representa un trazado de ECG simulado para fines de demostración.")
            
            fig = generate_ecg_graph()
            st.pyplot(fig)
            
            st.success("Análisis completado!")

# --- Ejecutar la aplicación ---
if __name__ == "__main__":
    main()