import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#in conventional nn we cannot directly feed images as input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]).astype('float32')
#normalization
X_train /= 255
X_test /= 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = 10 #10 digits to recognize

def classification_model():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1]*X_train.shape[2],)))
    model.add(Dense(X_train.shape[1]*X_train.shape[2], activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = classification_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)

print(f'Accuracy: {scores[1]}% \n Error: {1 - scores[1]}')    