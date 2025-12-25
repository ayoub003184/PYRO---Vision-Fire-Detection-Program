


import tensorflow as tf  # Core ML library for model building
from tensorflow.keras.models import Sequential  # To create a linear stack of layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
# Layers used for convolution, pooling, normalization, classification
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For image loading and augmentation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Stop training early if no improvement
import os  # Used to check dataset folder

# Set image input size, batch size and epochs
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

# Folder containing "Fire" and "NoFire" subfolders
DATASET_DIR = "dataset"

# Create training data generator with augmentation to reduce overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest',
    validation_split=0.2  # Use 20% of data for validation
)

# Create validation generator (no augmentation, just rescaling)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load training data from folder
train_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Binary classification (fire/no fire)
    subset='training'
)

# Load validation data
val_data = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Build CNN model using 3 convolution blocks with increasing complexity
model = Sequential([

    # Block 1 - small feature detection
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    # Block 2 - medium features
    Conv2D(64, (5, 5), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (5, 5), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    # Block 3 - large features
    Conv2D(128, (7, 7), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (7, 7), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.4),

    GlobalAveragePooling2D(),

    # Fully connected layer
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    # Output layer - single neuron for binary classification
    Dense(1, activation='sigmoid')
])

# Compile model with optimizer and extra evaluation metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# Add callbacks to stop early and reduce learning rate when stuck
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
]

# Show number of images being used
print(f"Training samples: {train_data.samples}")
print(f"Validation samples: {val_data.samples}")

# Start model training
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save the trained model to file
model.save("fire_model.h5")
print("✅ Model saved as fire_model.h5")