import flet as ft
import random
import numpy as np
import cv2
import tensorflow as tf
import os

# --- Constantes Globales (Importadas de la fase de entrenamiento) ---
MODEL_FILENAME = "aging_estimator_model.keras" 
TARGET_SIZE = (128, 128) # Tamaño estandarizado para la entrada del modelo ML

# URLs de placeholders para la UI
PLACEHOLDER_IMAGE_URL = "https://placehold.co/400x400/808080/FFFFFF?text=FOTO+DEL+ROSTRO"
PLACEHOLDER_IMAGE_UPLOADED = "https://picsum.photos/400/400"

# Variable para almacenar el modelo cargado (se inicializa en la primera llamada)
global_model = None

def load_age_model():
    """
    Carga el modelo de estimación de edad desde el archivo .keras.
    """
    global global_model
    if global_model is None:
        try:
            print(f"Cargando modelo: {MODEL_FILENAME}...")
            # IMPORTANTE: Asegúrate de que el archivo 'aging_estimator_model.keras' exista 
            # después de ejecutar 'model_training.py'.
            global_model = tf.keras.models.load_model(MODEL_FILENAME)
            print("Modelo de edad cargado exitosamente.")
        except Exception as e:
            print(f"Error al cargar el modelo {MODEL_FILENAME}. Usando la lógica de simulación (MOCK). Error: {e}")
            global_model = False # Usamos False para indicar que debe usarse el MOCK
            
    return global_model

def preprocess_image_for_model(image_path):
    """
    Carga, recorta (simulado) y redimensiona la imagen para la inferencia del modelo.
    
    Args:
        image_path (str): Ruta local (o simulada) de la imagen del rostro.
        
    Returns:
        np.array: Imagen preprocesada lista para el modelo (shape: (1, 128, 128, 3)).
    """
    # NOTA: En la app Flet real, 'image_path' debería ser la ruta del archivo local
    # subido por el FilePicker para poder cargarlo con cv2.
    
    # 1. Cargar imagen (Simulado con placeholder si no es una ruta real)
    # Puesto que Flet en modo web no da una ruta de archivo simple para cv2/numpy, 
    # esta parte debe ser manejada con un backend que reciba el archivo binario.
    # Por simplicidad y para que funcione con Flet en este entorno, mantendremos 
    # la lógica de simulación si no se proporciona un modelo real.
    
    # Si estamos en modo simulación (Mock), devolvemos None.
    if global_model is False:
        return None
        
    # Aquí iría el código real si tuvieras el archivo binario:
    # img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # === SIMULACIÓN DE IMAGEN PREPROCESADA PARA PRUEBA ===
    # Si el modelo está cargado, simulamos una imagen 128x128x3.
    # EN UNA APP REAL DE ESCRITORIO CON FLET, DEBES USAR CV2.IMREAD(IMAGE_PATH)
    dummy_img = np.random.rand(TARGET_SIZE[0], TARGET_SIZE[1], 3) * 255
    img_array = np.array(dummy_img, dtype='float32') / 255.0
    
    # Agregar dimensión de batch
    return np.expand_dims(img_array, axis=0)


def predict_age_real(image_path, chronological_age):
    """
    Usa el modelo real de Deep Learning para predecir la edad biológica.
    
    Args:
        image_path (str): Ruta de la imagen del rostro subida.
        chronological_age (int): Edad cronológica.
        
    Returns:
        int: Edad biológica estimada.
    """
    model = load_age_model()
    
    # Si el modelo no se pudo cargar, volvemos al MOCK
    if model is False:
        return mock_predict_age(chronological_age)

    # 1. Preprocesar la imagen
    processed_image = preprocess_image_for_model(image_path)
    
    if processed_image is None:
        return mock_predict_age(chronological_age)

    # 2. Inferencia
    try:
        prediction = model.predict(processed_image, verbose=0)
        estimated_age = round(prediction[0][0])
        
        # Asegurar que la edad estimada no sea menor a 1
        return max(1, estimated_age)
    except Exception as e:
        print(f"Error durante la inferencia. Usando MOCK. Error: {e}")
        return mock_predict_age(chronological_age)

# --- Lógica del Modelo Simulado (MOCK) (Mantenida como fallback) ---

def mock_predict_age(chronological_age):
    """
    SIMULACIÓN: Función de Deep Learning (MOCK) como fallback.
    """
    if chronological_age < 18:
        adjustment = random.uniform(-3, 1) 
    elif chronological_age >= 60:
        adjustment = random.uniform(-10, 15)
    else:
        adjustment = random.uniform(-5, 10)
        
    estimated_age = round(chronological_age + adjustment)
    
    return max(1, estimated_age)

def get_recommendations(aging_index):
    """
    Genera recomendaciones no clínicas basadas en el índice de envejecimiento.
    """
    if aging_index > 5:
        # Envejecimiento biológico percibido > Edad cronológica por > 5 años
        return [
            ft.Text("¡Alerta! Tu piel muestra un envejecimiento acelerado.", weight="bold", color=ft.Colors.RED_600),
            ft.Text("- **Hidratación:** Aumenta la ingesta de agua y usa cremas ricas en ácido hialurónico.", size=14),
            ft.Text("- **Protección Solar:** Usa protector solar SPF 50+ diariamente, incluso en días nublados.", size=14),
            ft.Text("- **Estrés:** Considera técnicas de relajación; el estrés crónico impacta la salud de la piel.", size=14),
        ]
    elif aging_index >= 1:
        # Envejecimiento biológico percibido > Edad cronológica (1 a 5 años)
        return [
            ft.Text("Tu piel se ve ligeramente más envejecida que tu edad cronológica.", weight="bold", color=ft.Colors.ORANGE_500),
            ft.Text("- **Antioxidantes:** Incorpora alimentos ricos en vitaminas C y E (frutas cítricas, nueces).", size=14),
            ft.Text("- **Sueño:** Asegúrate de dormir 7-9 horas; la reparación celular ocurre mientras duermes.", size=14),
            ft.Text("- **Chequeo:** Consulta sobre rutinas de cuidado con retinol o péptidos.", size=14),
        ]
    elif aging_index < -5:
        # Envejecimiento biológico percibido < Edad cronológica por > 5 años
        return [
            ft.Text("¡Excelente! Tu piel luce notablemente más joven que tu edad cronológica.", weight="bold", color=ft.Colors.GREEN_600),
            ft.Text("- **Mantener:** Continúa con tus hábitos de vida saludables y protección solar.", size=14),
            ft.Text("- **Dieta:** Enfócate en una dieta balanceada rica en grasas saludables (Omega-3).", size=14),
        ]
    else:
        # Edad biológica y cronológica cercanas (-5 a 0)
        return [
            ft.Text("Tu edad biológica es consistente con tu edad cronológica.", weight="bold", color=ft.Colors.BLUE_600),
            ft.Text("- **Cuidado Básico:** Mantén una rutina de limpieza e hidratación adecuada.", size=14),
            ft.Text("- **Ejercicio:** Realiza actividad física regular para mejorar la circulación y el brillo de la piel.", size=14),
        ]