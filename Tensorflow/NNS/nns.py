import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#data preprocessing to be able to be handled by nn
train_images = train_images / 255.0
test_images = test_images / 255.0

#sequential is just the most basic type of nn passing data from left to right
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #input layer
    keras.layers.Dense(128, activation="relu"), #hidden layer
    keras.layers.Dense(10, activation="softmax") #output layer
])

model.compile(optimizer = "adam",
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])

#training the model
model.fit(train_images, train_labels, epochs=1)

#testing how accurate the model really is
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print("Test accuracy: ", test_acc)

#predicting on new dataset using our trained model
predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[0])])
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()