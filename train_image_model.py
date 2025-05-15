import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 🔹 Paths
DATA_DIR = "data/combined"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
MODEL_PATH = "models/image_model_mobilenetv2.keras"

# 🔹 Image config
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 🔹 Find common classes
common_classes = sorted(
    list(set(os.listdir(TRAIN_DIR)) & set(os.listdir(VAL_DIR)))
)
print(f"✅ Using {len(common_classes)} common classes between train and val")

# 🔹 Data generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=common_classes
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=common_classes
)

# 🔹 MobileNetV2 base
base_model = MobileNetV2(include_top=False, input_shape=(*IMG_SIZE, 3), weights="imagenet")
base_model.trainable = False  # freeze base model

# 🔹 Classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(len(common_classes), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 🔹 Compile
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# 🔹 Callbacks
os.makedirs("models", exist_ok=True)
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

# 🔹 Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks
)

# 🔹 Save model
model.save(MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")
