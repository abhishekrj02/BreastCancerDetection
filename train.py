"""
Improved Training Script for Breast Cancer Detection Model
Uses DenseNet201 with optimized hyperparameters, class balancing,
data augmentation, and proper callbacks for robust training.

Usage:
    1. Update TRAIN_DIR and TEST_DIR paths below
    2. Run: python train.py
    3. Best weights saved to weight/modeldense1.h5
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

# ──────────────────── Configuration ────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

TRAIN_DIR = "path/to/train"   # <-- UPDATE: folder with 8 subfolders
TEST_DIR  = "path/to/test"    # <-- UPDATE: folder with 8 subfolders

# Class weights computed as: total / (n_classes * class_count)
# Dataset: 5724 samples across 8 classes (highly imbalanced)
CLASS_WEIGHTS = {
    0: 1.10,   # Density1Benign     (648 samples)
    1: 0.44,   # Density1Malignant  (1620 samples)
    2: 3.31,   # Density2Benign     (216 samples)
    3: 0.41,   # Density2Malignant  (1728 samples)
    4: 1.02,   # Density3Benign     (702 samples)
    5: 1.66,   # Density3Malignant  (432 samples)
    6: 2.21,   # Density4Benign     (324 samples)
    7: 13.25,  # Density4Malignant  (54 samples)
}


def build_model():
    """Build DenseNet201 transfer-learning model (same architecture as model.py)."""
    conv_base = DenseNet201(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        pooling="max",
        weights="imagenet",
    )

    # Freeze all layers except last 5 for fine-tuning
    for layer in conv_base.layers[:-5]:
        layer.trainable = False
    for layer in conv_base.layers[-5:]:
        layer.trainable = True

    model = Sequential([
        conv_base,
        BatchNormalization(),
        Dense(2048, activation="relu", kernel_regularizer=l1_l2(0.01)),
        BatchNormalization(),
        Dense(8, activation="softmax"),
    ])

    return model


def get_data_generators():
    """Create train/val/test data generators with augmentation."""

    # Training data with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.1,
    )

    # Test data: only rescale
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def train():
    os.makedirs("model", exist_ok=True)
    os.makedirs("weight", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Building model...")
    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        ModelCheckpoint(
            "weight/modeldense1.h5",
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir="logs"),
    ]

    print("Loading data...")
    train_gen, val_gen, test_gen = get_data_generators()

    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Class indices: {train_gen.class_indices}")

    print("\nStarting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=CLASS_WEIGHTS,
        callbacks=callbacks,
    )

    # Save final model architecture
    model.save("model/model.h5")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = model.evaluate(test_gen)
    print(f"Test Loss:     {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")

    return history


if __name__ == "__main__":
    train()
