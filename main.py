import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics="accuracy")

model.fit(x_train,y_train,validation_data=(x_test,y_test) ,epochs=5)
model.save('handwritten.h5')

trainedModel = tf.keras.models.load_model('handwritten.h5')

loss, accuracy = trainedModel.evaluate(x_test, y_test)

import os
from google.colab import drive
drive.mount('/content/drive')

image_dir = '/content/drive/MyDrive/Handwritten_digit_project/digits'

imge_number = 1
while os.path.isfile(f"{image_dir}/{imge_number}.png"):
  try:
    img = cv2.imread(f"{image_dir}/{imge_number}.png")[:,:,0]
    img = np.invert(np.array([img]))
    prediction = trainedModel.predict(img)
    print(f"Digit is probably a {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
  except:
    print("Error!!!")
  finally:
    imge_number += 1