import sys
assert sys.version_info >= (3, 5)
from tensorflow import keras
import sklearn
assert sklearn.__version__ >= "0.20"
import tensorflow as tf
assert tf.__version__ >= "2.0"
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def dense(label_dim, weight_init, activation) :
    return tf.keras.layers.Dense(units=label_dim, use_bias=True, kernel_initializer=weight_init,activation=activation)
def plot_history(histories, key='accuracy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  plt.show()

digits = load_digits()
x_data = digits.data
y_data = digits.target
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
x_valid, x_train = x_train[:50], x_train[50:]
y_valid, y_train = y_train[:50], y_train[50:]

#sequential모델을 만듬
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[8, 8]),
keras.layers.Dense(300, activation="relu"),
keras.layers.Dense(100, activation="relu"),
keras.layers.Dense(10, activation="softmax")
])
def makemodeldrop(weight_init):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[8, 8]))
    model.add(dense(300, weight_init, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(dense(100, weight_init, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(dense(10,weight_init, activation="softmax"))
    model.summary()
    return model

#갖고있는 트레이닝모델로 30번을 돌려보면서 확인
model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
tb_hist = keras.callbacks.TensorBoard(log_dir='../graph', histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(x_train, y_train, epochs=30,validation_data=(x_valid, y_valid),callbacks=[tb_hist])

print('§ Weight 초기화를 Xavier하고 30% 드롭아웃한 모델 ')
second_model=makemodeldrop('glorot_uniform')
second_model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
tb_hist1 = keras.callbacks.TensorBoard(log_dir='../graph', histogram_freq=0, write_graph=True, write_images=True)
history1 = second_model.fit(x_train, y_train, epochs=30,validation_data=(x_valid, y_valid),callbacks=[tb_hist1])

plot_history([('Xavier', history1),('Random', history)])
#그래프를 보면 가중치를 초기화하고 30%드롭한 모델이 초기 데이터보다 train 데이터와 val 데이터간의 간격이 더 적어서
#과적합이 덜 생기는 것을 보아 더 효과적이라고 할 수 있다.