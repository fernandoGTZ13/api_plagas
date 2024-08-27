from sklearn.preprocessing import label_binarize
import tensorflow as tf
import numpy as np

# Directorio donde están las imágenes
train_dir = "./PLAGAS"

# Crear un generador de datos con aumento de datos
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,  # Reducir el rango de rotación
    width_shift_range=0.1,  # Reducir el rango de desplazamiento
    height_shift_range=0.1,
    shear_range=0.1,  # Reducir el rango de cizalladura
    zoom_range=0.1,  # Reducir el rango de zoom
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=4,  # Batch size más pequeño para reducir el consumo de memoria
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=4,  # Batch size más pequeño para reducir el consumo de memoria
    class_mode='categorical',
    subset='validation'
)

# Cargar el modelo base preentrenado
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

# Congelar las primeras capas del modelo base
base_model.trainable = True
# Elegir cuántas capas descongelar
fine_tune_at = 100  # Puedes ajustar este número
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Crear el modelo completo
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Añadir Dropout
    tf.keras.layers.BatchNormalization(),  # Añadir Batch Normalization
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compilar el modelo con una tasa de aprendizaje baja
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(train_generator,
          validation_data=validation_generator,
          epochs=50,
          callbacks=[early_stopping])

# Función para predecir el tipo de plaga
def predict_image(img_path, threshold=0.7):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizar la imagen

    # Hacer la predicción
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)  # Obtener la confianza más alta

    # Mapear la clase predicha al nombre de la plaga
    class_names = list(train_generator.class_indices.keys())
    predicted_label = class_names[predicted_class[0]]

    # Comprobar si la confianza es menor que el umbral
    if confidence < threshold:
        return "Resultado no encontrado"

    # Verificar si la predicción es errónea comparando la confianza con las otras clases
    sorted_predictions = np.sort(predictions[0])[::-1]  # Ordenar predicciones en orden descendente
    second_best_confidence = sorted_predictions[1]  # Obtener la segunda mejor predicción

    # Si la diferencia entre la mejor predicción y la segunda mejor es pequeña, evitar un falso positivo
    if confidence - second_best_confidence < 0.2:
        return "Resultado no encontrado"

    return predicted_label
