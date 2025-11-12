import os
import cv2
import zipfile
import shutil
import kagglehub
import requests # Agregado: Necesario para descargar el Haar Cascade
from tqdm import tqdm

# --- CONFIGURACIÓN ---
DATASET_REF = "jangedoo/utkface-new"
OUTPUT_DIR = "UTKFace_Processed"
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
TARGET_SIZE = (128, 128) # Tamaño estandarizado para la entrada del modelo ML

def find_image_directory(root_dir):
    """Busca recursivamente el directorio con la mayor cantidad de archivos .jpg."""
    best_dir = root_dir
    max_images = 0
    
    # Recorrer subdirectorios
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Contar archivos .jpg en el directorio actual
        jpg_count = sum(1 for f in filenames if f.endswith(('.jpg', '.jpeg')))
        if jpg_count > max_images:
            max_images = jpg_count
            best_dir = dirpath
            
    if max_images > 0:
        return best_dir
    return None # No se encontró un directorio con imágenes

def download_and_extract_dataset():
    """
    Descarga el dataset de Kagglehub y devuelve la ruta del directorio
    donde se encuentran las imágenes RAW.
    """
    print(f"1. Descargando el dataset {DATASET_REF} a través de KaggleHub...")
    
    try:
        # Descarga la última versión del dataset
        dataset_path = kagglehub.dataset_download(DATASET_REF)
        print(f"Descarga completada. Ruta local: {dataset_path}")
        
        # Buscar recursivamente la carpeta real que contiene las imágenes .jpg
        raw_image_dir = find_image_directory(dataset_path)

        if raw_image_dir:
            print(f"Directorio de imágenes RAW encontrado: {raw_image_dir}")
            return raw_image_dir
        else:
            print("Advertencia: No se encontró un subdirectorio con imágenes .jpg dentro de la descarga.")
            return None
        
    except Exception as e:
        print(f"Error durante la descarga: {e}")
        return None

def prepare_face_detector():
    """
    Descarga el clasificador de rostro Haar Cascade si no existe localmente.
    """
    if not os.path.exists(FACE_CASCADE_PATH):
        print("Descargando clasificador de rostro (haarcascade)...")
        # Usaremos la versión de GitHub de OpenCV (Requiere conexión a internet)
        cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            r = requests.get(cascade_url, allow_redirects=True)
            with open(FACE_CASCADE_PATH, 'wb') as f:
                f.write(r.content)
            print("Haar Cascade descargado.")
        except Exception as e:
            print(f"Error al descargar Haar Cascade: {e}")
            return None
    
    return cv2.CascadeClassifier(FACE_CASCADE_PATH)

def preprocess_images(raw_image_dir):
    """
    2. Recorte y Preprocesamiento: Itera sobre las imágenes, detecta rostros,
    recorta, normaliza (TARGET_SIZE) y guarda en el directorio de salida.
    """
    if not raw_image_dir or not os.path.isdir(raw_image_dir):
        print("El directorio de imágenes RAW no es válido. Abortando preprocesamiento.")
        return
        
    # Crea el directorio de salida para las imágenes procesadas
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Inicializa el detector de rostros
    face_cascade = prepare_face_detector()
    if face_cascade is None:
        return

    # Usamos os.listdir(raw_image_dir) que ahora debe apuntar al directorio correcto
    image_files = [f for f in os.listdir(raw_image_dir) if f.endswith('.jpg')]
    
    print(f"\n2. Iniciando preprocesamiento de {len(image_files)} imágenes...")
    
    processed_count = 0
    
    for filename in tqdm(image_files, desc="Procesando imágenes"):
        filepath = os.path.join(raw_image_dir, filename)
        
        # El nombre del archivo en UTKFace codifica la edad (ej: 10_0_0_20170109232349547.jpg)
        # La edad es el primer componente
        try:
            age = int(filename.split('_')[0])
            gender = int(filename.split('_')[1])
            race = int(filename.split('_')[2])
        except (ValueError, IndexError):
            # Ignorar archivos con nombres mal formados
            continue 

        # Cargar imagen en escala de grises para el detector (más rápido)
        img = cv2.imread(filepath)
        # Asegurarse de que la imagen se cargó correctamente
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            
            # Recortar el rostro
            cropped_face = img[y:y+h, x:x+w]
            
            # Normalizar el tamaño (resizing)
            normalized_face = cv2.resize(cropped_face, TARGET_SIZE)
            
            # Guardar la imagen procesada con el mismo nombre
            output_filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(output_filepath, normalized_face)
            processed_count += 1

    print(f"\nPreprocesamiento finalizado. {processed_count} rostros guardados en la carpeta '{OUTPUT_DIR}'.")
    
    # Opcional: Limpiar archivos temporales de descarga
    # shutil.rmtree(os.path.dirname(raw_image_dir), ignore_errors=True)
    
    return os.path.abspath(OUTPUT_DIR)

def run_data_pipeline():
    """
    Función principal para ejecutar todo el pipeline de datos.
    """
    raw_dir = download_and_extract_dataset()
    if raw_dir:
        processed_dir = preprocess_images(raw_dir)
        print(f"\nPipeline de datos completado. Imágenes listas para entrenamiento en: {processed_dir}")
        return processed_dir
    else:
        print("No se pudo completar la fase de recolección de datos.")
        return None

if __name__ == "__main__":
    # Si ejecutas este archivo directamente, iniciará la descarga y el preprocesamiento
    run_data_pipeline()