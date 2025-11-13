import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
PROCESSED_DATA_DIR = "UTKFace_Processed"
MODEL_FILENAME = "aging_estimator_model.keras"
TARGET_SIZE = (128, 128)

# --- FASE 2.1: CARGA DE DATOS ---
def load_data(data_dir):
    X, y = [], []
    print(f"Cargando datos desde: {data_dir}...")

    if not os.path.isdir(data_dir):
        print(f"Error: Directorio '{data_dir}' no encontrado. Ejecuta 'data_prep.py' primero.")
        return np.array(X), np.array(y)

    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    if not image_files:
        print(f"Error: No se encontraron imágenes en el directorio '{data_dir}'.")
        return np.array(X), np.array(y)

    for filename in tqdm(image_files, desc="Cargando y extrayendo etiquetas"):
        filepath = os.path.join(data_dir, filename)
        try:
            age = int(filename.split('_')[0])
            if age > 116 or age < 1:
                continue
        except (ValueError, IndexError):
            continue

        img = cv2.imread(filepath)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img_rgb)
        y.append(age)

    return np.array(X, dtype='float32'), np.array(y, dtype='int32')

# --- FASE 3: DEFINICIÓN Y ENTRENAMIENTO DEL MODELO ---
def define_cnn_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mae', metrics=['mae', 'mse'])
    return model

def train_model(X, y, epochs=50, batch_size=64):
    if X.shape[0] == 0:
        print("No hay datos para entrenar. Abortando entrenamiento.")
        return None

    X_normalized = X / 255.0
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    print(f"Tamaño de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Tamaño de prueba: {X_test.shape[0]} muestras")

    model = define_cnn_model(X_train.shape[1:])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    callbacks_list = [early_stopping]

    print(f"\nIniciando entrenamiento (Máx. {epochs} épocas con Early Stopping)...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list
    )

    model.save(MODEL_FILENAME)
    print(f"\n✅ Modelo entrenado guardado como: {MODEL_FILENAME}")

    return model, X_test, y_test, history

# --- FASE 4: PIPELINE COMPLETO ---
def run_training_pipeline():
    X, y = load_data(PROCESSED_DATA_DIR)

    if X.shape[0] > 0:
        model, X_test, y_test, history = train_model(X, y)

        if model:
            print("\n--- FASE 4: EVALUACIÓN ---")
            loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
            print(f"\nResultados de la Evaluación (Conjunto de Prueba):")
            print(f"MAE (Error Absoluto Medio): {mae:.2f} años")
            print(f"MSE (Error Cuadrático Medio): {mse:.2f}")
            print(f"👉 El modelo se equivoca en promedio por {mae:.2f} años.")

            # --- NUEVO: Graficar curva de pérdida real ---
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['mae'], label='MAE de Entrenamiento', linewidth=2)
            plt.plot(history.history['val_mae'], label='MAE de Validación', linewidth=2)
            plt.xlabel('Épocas', fontsize=13)
            plt.ylabel('Error Absoluto Medio (MAE)', fontsize=13)
            plt.title('Curva de Pérdida Real durante el Entrenamiento', fontsize=15)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig('curva_real_mae.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Curva de pérdida real guardada como: curva_real_mae.png")

            return model
    else:
        print("⚠️ No se encontraron datos válidos para entrenar.")
    return None

# --- OPCIONAL: Curva simulada de ejemplo ---
def plot_loss_history():
    epochs = 28
    train_mae = np.linspace(14.0, 4.5, epochs) + np.random.uniform(-0.1, 0.1, epochs)
    val_mae = np.linspace(16.0, 9.5, epochs) + np.random.uniform(-0.5, 0.5, epochs)
    best_epoch_index = 23
    val_mae[best_epoch_index] = 8.77
    for i in range(best_epoch_index + 1, epochs):
        val_mae[i] = val_mae[i-1] + 0.1

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, epochs + 1), train_mae, label='MAE de Entrenamiento', color='#007ACC', linewidth=2)
    plt.plot(np.arange(1, epochs + 1), val_mae, label='MAE de Validación', color='#D95319', linewidth=2)
    plt.axvline(x=best_epoch_index + 5, color='gray', linestyle=':', linewidth=1.5, label='Punto de Parada')
    plt.axvline(x=best_epoch_index, color='green', linestyle='--', linewidth=1.5, label='Mejor Modelo')
    plt.title('Curva Simulada de Pérdida (MAE)', fontsize=16)
    plt.xlabel('Época', fontsize=14)
    plt.ylabel('MAE', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('figura_curva_perdida_mae.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📈 Gráfico simulado guardado como: figura_curva_perdida_mae.png")

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    # Puedes cambiar entre entrenamiento real o curva simulada
    ejecutar_entrenamiento = True

    if ejecutar_entrenamiento:
        run_training_pipeline()
    else:
        plot_loss_history()
