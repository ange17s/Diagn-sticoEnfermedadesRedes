import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
# AÑADIDO: Importar EarlyStopping para detener el entrenamiento cuando no haya mejora.
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

# REQUERIMIENTOS:
# Este script requiere las siguientes bibliotecas:
# pip install tensorflow scikit-learn opencv-python tqdm

# --- CONFIGURACIÓN DE RUTAS Y PARÁMETROS ---
PROCESSED_DATA_DIR = "UTKFace_Processed"
# CORRECCIÓN: Usamos el formato nativo de Keras (.keras) para evitar el WARNING HDF5.
MODEL_FILENAME = "aging_estimator_model.keras" 
TARGET_SIZE = (128, 128) 

# --- FASE 2.1: CARGA DE DATOS (Usando los datos preprocesados) ---

def load_data(data_dir):
    """
    Carga las imágenes preprocesadas y extrae las etiquetas de edad de los nombres de archivo.
    
    Args:
        data_dir (str): Directorio que contiene las imágenes de rostros preprocesadas.
        
    Returns:
        tuple: (imágenes (X), edades (y)) como arrays de NumPy.
    """
    X = [] # Para almacenar las imágenes (features)
    y = [] # Para almacenar las edades (labels)
    
    print(f"Cargando datos desde: {data_dir}...")
    
    if not os.path.isdir(data_dir):
        print(f"Error: Directorio '{data_dir}' no encontrado. Asegúrate de ejecutar 'data_prep.py' primero.")
        return np.array(X), np.array(y)
        
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    
    if not image_files:
        print(f"Error: No se encontraron imágenes en el directorio '{data_dir}'.")
        return np.array(X), np.array(y)

    for filename in tqdm(image_files, desc="Cargando y extrayendo etiquetas"):
        filepath = os.path.join(data_dir, filename)
        
        # 1. Extraer etiqueta de edad del nombre del archivo (ej: 10_0_0_...jpg)
        try:
            age = int(filename.split('_')[0])
            # Descartar edades extremas o nulas si es necesario (UTKFace cubre 0-116)
            if age > 116 or age < 1:
                continue
        except (ValueError, IndexError):
            continue 

        # 2. Cargar y verificar imagen
        img = cv2.imread(filepath)
        if img is None:
            continue
            
        # 3. Normalización: Convertir a RGB (si no lo está) y almacenar
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img_rgb)
        y.append(age)

    return np.array(X, dtype='float32'), np.array(y, dtype='int32')

# --- FASE 3: DEFINICIÓN Y ENTRENAMIENTO DEL MODELO ---

def define_cnn_model(input_shape):
    """
    Define una arquitectura de red neuronal convolucional (CNN) para la estimación de edad.
    """
    # Usamos tf.keras.models.Sequential en lugar del import directo
    model = tf.keras.models.Sequential([
        # Capa 1: Convolución y Pooling (extracción de características)
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Capa 2: Convolución y Pooling
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Capa 3: Convolución y Pooling
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Capa de Aplanamiento
        tf.keras.layers.Flatten(),
        
        # Capa Densa (Clasificación/Regresión)
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        
        # Capa de Salida: Una neurona para la regresión (predicción de la edad continua)
        tf.keras.layers.Dense(1, activation='linear') 
    ])
    
    # Compilación del modelo (usando MAE, ideal para regresión de edad)
    model.compile(optimizer='adam', loss='mae', metrics=['mae', 'mse'])
    return model

def train_model(X, y, epochs=50, batch_size=64):
    """
    Divide los datos, entrena el modelo y lo guarda.
    
    MODIFICACIÓN: Implementa Early Stopping y aumenta el número de épocas.
    """
    if X.shape[0] == 0:
        print("No hay datos para entrenar. Abortando entrenamiento.")
        return None

    # 1. Normalización de las imágenes a [0, 1]
    X_normalized = X / 255.0

    # 2. División de datos (80% entrenamiento, 20% prueba/validación)
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42
    )

    print(f"Tamaño de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Tamaño de prueba: {X_test.shape[0]} muestras")

    # 3. Definición y resumen del modelo
    input_shape = X_train.shape[1:]
    model = define_cnn_model(input_shape)
    model.summary()
    
    # 4. CALLBACKS: Early Stopping
    # Monitoriza la pérdida de validación ('val_loss' que es el MAE en la validación)
    # Si 'val_loss' no mejora en 5 épocas ('patience=5'), detiene el entrenamiento.
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )
    
    callbacks_list = [early_stopping]

    # 5. Entrenamiento
    print(f"\nIniciando entrenamiento (Máx. {epochs} épocas con Early Stopping)...")
    history = model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list # AÑADIDO: Pasamos la lista de callbacks
    )

    # 6. Guardar el modelo entrenado
    # Usa la variable actualizada con extensión .keras
    model.save(MODEL_FILENAME)
    print(f"\nModelo entrenado guardado como: {MODEL_FILENAME}")
    
    # Devolver el modelo y los datos de prueba para la Fase 4 (Evaluación)
    return model, X_test, y_test, history

def run_training_pipeline():
    """
    Función principal para ejecutar el pipeline de carga y entrenamiento.
    """
    # 1. Cargar datos preprocesados
    X, y = load_data(PROCESSED_DATA_DIR)
    
    # 2. Entrenar el modelo
    # MODIFICADO: Ahora usará un máximo de 50 épocas con Early Stopping
    if X.shape[0] > 0:
        model, X_test, y_test, history = train_model(X, y)
        
        if model:
            # FASE 4: Evaluación (Integrada aquí para mostrar resultados inmediatos)
            print("\n--- FASE 4: EVALUACIÓN ---")
            
            # Evaluar rendimiento en el conjunto de prueba
            loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
            
            print(f"\nResultados de la Evaluación (Conjunto de Prueba):")
            print(f"Pérdida (MAE - Error Absoluto Medio): {mae:.2f} años")
            print(f"Pérdida (MSE - Error Cuadrático Medio): {mse:.2f}")
            print(f"Esto significa que el modelo se equivoca en promedio por {mae:.2f} años.")
            
            # Puedes agregar código aquí para graficar la pérdida de entrenamiento vs validación (history)
            
            return model
            
    return None

if __name__ == "__main__":
    run_training_pipeline()