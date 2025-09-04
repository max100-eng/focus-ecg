# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

def procesar_imagen_vision(ruta_imagen):
    """
    Procesa una imagen para detección de bordes y reducción de ruido.
    
    Args:
        ruta_imagen (str): La ruta del archivo de imagen a procesar.
    
    Returns:
        tuple: Una tupla que contiene la imagen original, la imagen en 
               escala de grises, la imagen con ruido reducido y la imagen con bordes detectados.
    """
    # 1. Cargar la imagen
    imagen_original = cv2.imread(ruta_imagen)
    if imagen_original is None:
        print(f"Error: No se pudo cargar la imagen en la ruta {ruta_imagen}")
        return None, None, None, None

    # 2. Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)

    # 3. Aplicar un filtro Gaussiano para reducir el ruido
    imagen_desenfocada = cv2.GaussianBlur(imagen_gris, (5, 5), 0)

    # 4. Detectar los bordes con el algoritmo de Canny
    bordes = cv2.Canny(imagen_desenfocada, 50, 150)

    return imagen_original, imagen_gris, imagen_desenfocada, bordes

def mostrar_imagenes(imagen_original, imagen_gris, imagen_desenfocada, bordes):
    """
    Muestra las imágenes procesadas en una ventana.
    """
    if imagen_original is not None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        
        axs[0, 0].imshow(cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title('1. Imagen Original')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(imagen_gris, cmap='gray')
        axs[0, 1].set_title('2. Escala de Grises')
        axs[0, 1].axis('off')

        axs[1, 0].imshow(imagen_desenfocada, cmap='gray')
        axs[1, 0].set_title('3. Ruido Reducido (Gaussiano)')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(bordes, cmap='gray')
        axs[1, 1].set_title('4. Detección de Bordes (Canny)')
        axs[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

# --- Uso del script ---
if __name__ == "__main__":
    # Asegúrate de que el archivo 'imagen_ecg_mi.jpg' exista en la carpeta 'ECG_DATA'
    # y que el nombre de la carpeta esté escrito correctamente
    ruta_archivo = ruta_archivo ='C:/Users/maram/focus-ecg/ECG_DATA/imagen_ecg_mi.jpg'
    
    # Procesar la imagen
    original, gris, desenfocada, bordes = procesar_imagen_vision(ruta_archivo)
    
    # Mostrar los resultados si el procesamiento fue exitoso
    if original is not None:
        mostrar_imagenes(original, gris, desenfocada, bordes)