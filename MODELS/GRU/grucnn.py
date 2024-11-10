import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Reshape, GRU, Dense, Bidirectional
from tensorflow.keras.models import Model

num_classes = 34

# Input layer for the video frames
input_shape = (21, 3)  # Adjust input shape as needed
video_input = Input(shape=input_shape)

# Convolutional layers to capture spatial features
conv1 = Conv1D(32, 3, activation='relu')(video_input)
maxpool1 = MaxPooling1D(2)(conv1)
conv2 = Conv1D(64, 3, activation='relu')(maxpool1)
maxpool2 = MaxPooling1D(2)(conv2)

# Reshape the CNN output for sequence input
reshaped_cnn_output = Reshape((-1, 64))(maxpool2)

# GRU layers to capture temporal features
gru1 = Bidirectional(GRU(64, return_sequences=True))(reshaped_cnn_output)
gru2 = Bidirectional(GRU(64, return_sequences=False))(gru1)

# Classification layer
classification_output = Dense(num_classes, activation='softmax')(gru2)

# Create the model
model = Model(inputs=video_input, outputs=classification_output)
