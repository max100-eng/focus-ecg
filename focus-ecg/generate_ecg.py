import numpy as np
import matplotlib.pyplot as plt

def generate_ecg(bpm, duration, noise_level=0.1):
    """
    Genera una señal de ECG sintética.

    Args:
        bpm (int): Frecuencia cardíaca en latidos por minuto.
        duration (int): Duración de la simulación en segundos.
        noise_level (float): Nivel de ruido aleatorio.
        
    Returns:
        tuple: (tiempo_array, ecg_signal_array)
    """
    # Frecuencia de muestreo
    fs = 200  # Muestras por segundo
    
    # Período del latido cardíaco en segundos
    period = 60 / bpm
    
    # Número total de puntos
    num_points = duration * fs
    
    # Array de tiempo
    t = np.linspace(0, duration, num_points, endpoint=False)
    
    # Inicializar la señal
    ecg_signal = np.zeros_like(t)

    # Coordenadas de los picos (relativas a un ciclo)
    # picos: [P, Q, R, S, T]
    picos_relativos = np.array([0.15, 0.22, 0.25, 0.28, 0.45])
    
    # Amplitudes de los picos
    amplitudes = np.array([0.1, -0.5, 2.0, -0.6, 0.4])
    
    # Anchos de los picos (sigmas de la gaussiana)
    sigmas = np.array([0.02, 0.01, 0.01, 0.01, 0.04])
    
    # Generar latidos en cada ciclo
    num_beats = int(duration / period)
    for i in range(num_beats):
        beat_start_time = i * period
        for j in range(len(picos_relativos)):
            peak_time = beat_start_time + picos_relativos[j] * period
            ecg_signal += amplitudes[j] * np.exp(-((t - peak_time)**2) / (2 * sigmas[j]**2))

    # Añadir ruido aleatorio
    noise = noise_level * np.random.randn(num_points)
    ecg_signal += noise
    
    return t, ecg_signal

def plot_ecg(t, signal, title="Simulación de Señal de ECG"):
    """
    Grafica la señal de ECG.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, label='ECG Sintético')
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Simular un ritmo cardíaco normal
    t_normal, ecg_normal = generate_ecg(bpm=75, duration=10)
    plot_ecg(t_normal, ecg_normal, title="Simulación de ECG (BPM=75)")

    # Simular una taquicardia (ritmo acelerado)
    t_tachy, ecg_tachy = generate_ecg(bpm=120, duration=10)
    plot_ecg(t_tachy, ecg_tachy, title="Simulación de ECG (Taquicardia, BPM=120)")

    # Simular una bradicardia (ritmo lento)
    t_brady, ecg_brady = generate_ecg(bpm=45, duration=10)
    plot_ecg(t_brady, ecg_brady, title="Simulación de ECG (Bradicardia, BPM=45)")