import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
x_data = digits.data
y_data = digits.target
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.3)
x_val=x_train[:100]
x_train=x_train[100:]
y_val=y_train[:100]
y_train=y_train[100:]
model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[8,8]))
model.add(keras.layers.Dense(300,activation='relu'))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
tb_hist= keras.callbacks.TensorBoard(log_dir='../graph', histogram_freq=0, write_graph=True, write_images=True)
history=model.fit(x_train,y_train,epochs=30,validation_data=(x_val,y_val),callbacks=[tb_hist])




