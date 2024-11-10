import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # Input Layer
    layers.Input(shape=(21, 3)),  # Input shape based on your extracted features
    
    # Convolutional layers
    layers.Conv1D(32, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    
    # Flatten the output
    layers.Flatten(),
    
    # Dense (fully connected) layers
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # Dropout layer to reduce overfitting
    
    layers.Dense(34, activation='softmax')  # 36 classes, softmax activation
])
