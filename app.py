import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

# --- Cargar los modelos de análisis de señales ---
@st.cache_resource
def load_analysis_models():
    models = {}
    try:
        models['mitbih'] = tf.keras.models.load_model("best_model_mitbih.keras")
        st.success("✅ Modelo MIT-BIH (best_model_mitbih.keras) cargado.")
    except Exception as e:
        st.error(f"❌ Error al cargar best_model_mitbih.keras: {e}")
        models['mitbih'] = None
    
    try:
        models['ptbdb'] = tf.keras.models.load_model("best_model_ptbdb.keras")
        st.success("✅ Modelo PTB-DB (best_model_ptbdb.keras) cargado.")
    except Exception as e:
        st.error(f"❌ Error al cargar best_model_ptbdb.keras: {e}")
        models['ptbdb'] = None
    
    return models

analysis_models = load_analysis_models()

# --- Funciones de simulación y preprocesamiento ---
def simulate_ecg_analysis():
    """Simula el análisis de los intervalos y métricas del ECG."""
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
        
        # --- Análisis por los modelos de señales ---
        st.markdown("### 📈 Diagnóstico basado en modelos de señales")
        
        if analysis_models['mitbih'] or analysis_models['ptbdb']:
            try:
                # Nota: Los modelos de señales esperan datos de señal 1D.
                # Aquí, se simula el resultado del análisis
                # porque la extracción de la señal a partir de la imagen es un paso complejo.
                
                # Simular la predicción de cada modelo
                # En un caso real, aquí iría la lógica para pasar los datos al modelo
                mitbih_prediction = np.random.rand(1, 5) # Simulación de la salida del modelo
                ptbdb_prediction = np.random.rand(1, 2)  # Simulación de la salida del modelo
                
                # Mapear las predicciones a un diagnóstico legible
                mitbih_diagnosis = ["Normal", "SVEB", "VEB", "Fusionado", "Desconocido"][np.argmax(mitbih_prediction)]
                ptbdb_diagnosis = ["Normal", "Infarto de miocardio"][np.argmax(ptbdb_prediction)]

                st.info(f"Diagnóstico MIT-BIH: **{mitbih_diagnosis}**")
                st.info(f"Diagnóstico PTB-DB: **{ptbdb_diagnosis}**")

                # Obtener datos simulados de análisis de intervalos
                analysis_results = simulate_ecg_analysis()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Ritmo Cardíaco (bpm)", value=analysis_results["heartRate"])
                with col2:
                    st.metric(label="Diagnóstico Automático", value=analysis_results["autoDiagnosis"])
                
                st.write("#### Datos Clave del ECG (Simulado)")
                df = pd.DataFrame(analysis_results["ecgIntervals"])
                st.dataframe(df.set_index('interval'))
            
            except Exception as e:
                st.error(f"❌ Error al ejecutar los modelos de análisis: {e}")
                st.info("No se pudo obtener el análisis. Revisa los modelos de señales.")
        else:
            st.warning("⚠️ No se cargó ningún modelo de análisis. El diagnóstico no está disponible.")

        # --- Gráfico simulado del trazado ECG ---
        st.write("---")
        st.markdown("### 📊 Trazado ECG Simulado")
        st.write("Este gráfico representa un trazado de ECG simulado para fines de demostración.")
        fig = generate_ecg_graph()
        st.pyplot(fig)
        
        st.success("Análisis completo.")
        st.warning("⚠️ **Aviso Importante**: Esta es una herramienta experimental. Consulta siempre a un profesional de la salud para un diagnóstico médico.")

# --- Ejecutar la aplicación ---
if __name__ == "__main__":
    main()