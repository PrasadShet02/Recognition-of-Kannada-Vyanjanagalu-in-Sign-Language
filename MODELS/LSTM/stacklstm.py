import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model

# Define the model
num_classes = 34  # Number of classes
input_shape = (21,3)  # Adjust input shape as needed
video_input = Input(shape=input_shape)

# First LSTM layer
lstm1 = Bidirectional(LSTM(64, return_sequences=True))(video_input)

# Second LSTM layer
lstm2 = Bidirectional(LSTM(128, return_sequences=True))(lstm1)

# Third LSTM layer
lstm3 = Bidirectional(LSTM(64, return_sequences=True))(lstm2)

# Four LSTM layer
lstm4 = Bidirectional(LSTM(128, return_sequences=True))(lstm3)

# Five LSTM layer
lstm5 = Bidirectional(LSTM(64, return_sequences=False))(lstm4)

# Add a Dense layer to reshape the output before TimeDistributed
#dense_layer = Dense(64, activation='relu')(lstm3)

# Apply TimeDistributed to the Dense layer
#time_dist = TimeDistributed(Dense(64, activation='relu', input_shape = input_shape))(lstm5)

# Classification layer
classification_output = Dense(num_classes, activation='softmax')(lstm5)

# Create the model
model = Model(inputs=video_input, outputs=classification_output)
