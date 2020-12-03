import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN
test_data=np.array([[[0.5,0.6,0.7]],[[0.6,0.7,0.8]]])
test_data=test_data.reshape(2,3,1)
x_data = np.array([[[0.1, 0.2, 0.3]], [[0.2, 0.3, 0.4]], [[0.3, 0.4, 0.5]], [[0.4, 0.5, 0.6]]])
y_data = np.array([[[0.2, 0.3, 0.4]], [[0.3, 0.4, 0.5]], [[0.4, 0.5, 0.6]], [[0.5, 0.6, 0.7]]])
x_data=x_data.reshape(4,3,1)
y_data=y_data.reshape(4,3,1)

model=Sequential()
model.add(SimpleRNN(50,input_shape=(3,1),return_sequences=True))
model.add(SimpleRNN(40,return_sequences=True))
model.add(SimpleRNN(30,return_sequences=True))
model.add(Dense(1))
model.summary()
model.compile(loss='mse',optimizer='adam',metrics='mse')
model.fit(x_data,y_data,epochs=100,verbose=0)
y=model.predict(test_data)
print(y)
