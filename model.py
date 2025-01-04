import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
%matplotlib inline

(X_train,y_train) , (X_test, y_test) = keras.datasets.mnist.load_data() #load all handwritten digits dataset
#plt.matshow(X_train[2]) #scaling
X_train = X_train /255
X_test = X_test/ 255

#we will now flatten our dataset
X_train_flattened=X_train.reshape(len(X_train),28*28)
X_test_flattened=X_test.reshape(len(X_test),28*28)
X_test_flattened.shape

model=keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,),activation='sigmoid')
                       ])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train_flattened , y_train, epochs=5)
model.evaluate(X_test_flattened, y_test)
plt.matshow(X_test[1])

y_predicted=model.predict(X_test_flattened)
y_predicted[1]
np.argmax(y_predicted[1])

model.save("mnist_model.h5")