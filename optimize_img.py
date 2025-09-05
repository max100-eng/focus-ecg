import os
from PIL import Image

def convertir_a_webp(carpeta_entrada, carpeta_salida, calidad=85):
    """
    Convierte y optimiza imágenes de una carpeta a formato WebP.

    Args:
        carpeta_entrada (str): Ruta de la carpeta con las imágenes originales.
        carpeta_salida (str): Ruta de la carpeta donde se guardarán las imágenes WebP.
        calidad (int): Nivel de calidad de la compresión (0-100).
    """
    # Crea la carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    # Recorre todos los archivos en la carpeta de entrada
    for nombre_archivo in os.listdir(carpeta_entrada):
        # Filtra solo los archivos de imagen
        if nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            ruta_entrada = os.path.join(carpeta_entrada, nombre_archivo)
            
            # Crea un nombre de archivo para la salida con la extensión .webp
            nombre_salida = os.path.splitext(nombre_archivo)[0] + '.webp'
            ruta_salida = os.path.join(carpeta_salida, nombre_salida)

            try:
                # Abre la imagen con Pillow
                with Image.open(ruta_entrada) as img:
                    # Guarda la imagen en formato WebP con optimización
                    img.save(ruta_salida, 'webp', quality=calidad)
                    print(f"✅ Imagen convertida: {nombre_archivo} -> {nombre_salida}")
            except Exception as e:
                print(f"❌ Error al procesar la imagen {nombre_archivo}: {e}")

if __name__ == "__main__":
    # Define las rutas de entrada y salida
    # Asegúrate de que tu carpeta de imágenes de ECG exista en esta ruta
    carpeta_original = 'ECG_DATA/train'
    carpeta_optimizada = 'ECG_DATA/train_webp'

    print(f"Iniciando conversión de imágenes de {carpeta_original}...")
    convertir_a_webp(carpeta_original, carpeta_optimizada)
    print("Conversión de imágenes finalizada.")


