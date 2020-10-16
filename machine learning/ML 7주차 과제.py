import sys
assert sys.version_info >= (3, 5)
from tensorflow import keras
import sklearn
assert sklearn.__version__ >= "0.20"
import tensorflow as tf
assert tf.__version__ >= "2.0"
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def Dense(label_dim, activation) :
    return tf.keras.layers.Dense(units=label_dim, use_bias=True,activation=activation)

def dense(label_dim, weight_init, activation) :
    return tf.keras.layers.Dense(units=label_dim, use_bias=True, kernel_initializer=weight_init,activation=activation)

def modelpredict(model, X_train, y_train, X_valid, y_valid):
    tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
    history = model.fit(X_train, y_train, epochs=30,validation_data=(X_valid, y_valid), callbacks=[tb_hist])
    return history

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

def makemodel():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[8, 8]))
    model.add(Dense(300,activation="relu"))
    model.add(Dense(100,activation="relu"))
    model.add(Dense(10,activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    return model
def makemodeldrop(weight_init):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[8, 8]))
    model.add(dense(300, weight_init, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(dense(100, weight_init, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(dense(10,weight_init, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    #model.summary()
    return model

if __name__=="__main__":
    digits = load_digits()
    x_data = digits.data
    y_data = digits.target
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
    x_valid, x_train = x_train[:50], x_train[50:]
    y_valid, y_train = y_train[:50], y_train[50:]
    print('weight 초기화를 하지 않고 (RandomNormal) dropout도 하지 않은 모델')
    model=makemodel()
    history=modelpredict(model,x_train,y_train,x_test,y_test)

    print('§ Weight 초기화를 Xavier하고 30% 드롭아웃한 모델 ')
    second_model=makemodeldrop('glorot_uniform')
    history1=modelpredict(second_model,x_train,y_train,x_test,y_test)

    plot_history([('Random', history),('Xavier', history1)])
#그래프를 보면 가중치를 초기화하고 30%드롭한 모델이 초기 데이터보다 train 데이터와 val(test) 데이터간의 간격이 더 적어서
#과적합이 덜 생기는 것을 보아 더 효과적이라고 할 수 있다.

