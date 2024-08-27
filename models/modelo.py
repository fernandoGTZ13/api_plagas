import tensorflow as tf
import numpy as np

# Crear un generador de datos con un batch_size más pequeño para reducir el uso de memoria
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,  # Reducir el batch_size a 16 para menor consumo de memoria
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,  # Reducir el batch_size a 16 también aquí
    class_mode='categorical',
    subset='validation'
)

# Cargar el modelo base preentrenado
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

# Congelar las primeras capas del modelo base
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Crear el modelo completo
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compilar el modelo con una tasa de aprendizaje baja
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo con callbacks para manejo de memoria y early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(train_generator,
          validation_data=validation_generator,
          epochs=50,
          callbacks=[early_stopping])

# Función para predecir el tipo de plaga
def predict_image(img_path, threshold=0.7):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)

    class_names = list(train_generator.class_indices.keys())
    predicted_label = class_names[predicted_class[0]]

    if confidence < threshold:
        return "Resultado no encontrado"

    sorted_predictions = np.sort(predictions[0])[::-1]
    second_best_confidence = sorted_predictions[1]

    if confidence - second_best_confidence < 0.2:
        return "Resultado no encontrado"

    return predicted_label
