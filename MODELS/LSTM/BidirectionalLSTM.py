import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

""" # Define the model
model = tf.keras.Sequential()

# Add the first Bidirectional LSTM layer
model.add(Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(50, 225)))

# Add more Bidirectional LSTM layers
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))

# Add a dense layer for classification
num_classes = 21 # Specify the number of classes
model.add(Dense(units=num_classes, activation='softmax')) """

""" num_classes = 21

model = tf.keras.Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(50,225)))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
 """
num_classes = 34

model = tf.keras.Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(21,3)))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dense(num_classes, activation='softmax'))
