import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, GRU, Dense

 
num_classes = 34

model = tf.keras.Sequential()
model.add(Bidirectional(GRU(64, return_sequences=True), input_shape=(21,3)))
model.add(Bidirectional(GRU(128, return_sequences=True)))
model.add(Bidirectional(GRU(64, return_sequences=False)))
model.add(Dense(num_classes, activation='softmax'))

