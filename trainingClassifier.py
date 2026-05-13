import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "DATASET"

# =========================
# LOAD DATA
# =========================
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    f"{DATA_DIR}/TRAIN",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    f"{DATA_DIR}/TRAIN",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    f"{DATA_DIR}/TEST",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

class_names = train_data.class_names
print("Classes:", class_names)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.map(lambda x, y: (preprocess_input(x), y)).prefetch(AUTOTUNE)
val_data = val_data.map(lambda x, y: (preprocess_input(x), y)).prefetch(AUTOTUNE)
test_data = test_data.map(lambda x, y: (preprocess_input(x), y)).prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = data_augmentation(base_model.input)
x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)

x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)

# Binary output
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# =========================
# COMPILE
# =========================
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# =========================
# TRAIN (PHASE 1)
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# =========================
# FINE-TUNING (PHASE 2)
# =========================
base_model.trainable = True

# Freeze early layers
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# =========================
# TEST EVALUATION
# =========================
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.4f}")

# =========================
# SAVE MODEL
# =========================
model.save('new_waste_classifier_mobilenetv2.keras')