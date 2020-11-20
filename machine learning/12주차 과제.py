import numpy as np
from tensorflow.keras import Sequential
from keras.layers import SimpleRNN,Dense


def make_model(x_data, y_data):
    model = Sequential()
    model.add(SimpleRNN(50, return_sequences=True,input_shape=(3,1)))
    model.add(Dense(1))
    print(model.summary())
    model.compile(loss='mse',optimizer='adam')
    model.fit(x_data, y_data, batch_size=128, epochs=500,verbose=0)
    y_hat=model.predict([[[0.5],[0.6],[0.7]],[[0.6],[0.7],[0.8]]])
    print('Vanilla simpleRNN')
    print(y_hat)

def stack_model(x_data, y_data):
    model = Sequential()
    model.add(SimpleRNN(50, return_sequences=True,input_shape=(3,1)))
    model.add(SimpleRNN(50, return_sequences=True))
    model.add(Dense(1))
    print(model.summary())
    model.compile(loss='mse',optimizer='adam')
    model.fit(x_data, y_data, batch_size=128, epochs=500,verbose=0)
    y_hat=model.predict([[[0.5],[0.6],[0.7]],[[0.6],[0.7],[0.8]]])
    print('Stacked simpleRNN')
    print(y_hat)



def main():
    x_data=np.array([[0.1, 0.2, 0.3],[0.2, 0.3, 0.4],[0.3, 0.4, 0.5],[0.4,0.5,0.6]])
    y_data=np.array([[0.2, 0.3, 0.4],[0.3, 0.4, 0.5],[0.4, 0.5, 0.6],[0.5,0.6,0.7]])
    x_data=x_data.reshape(4,3,1)
    y_data=y_data.reshape(4,3,1)

    make_model(x_data, y_data)
    stack_model(x_data, y_data)
    #같은 에포크라면 stack된 RNN이 더 정확성이 좋다.
main()