
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

num_classes = 34

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(21,3)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))