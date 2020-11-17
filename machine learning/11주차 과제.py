import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
# convert into dataset matrix
def convertToMatrix(data, step):
    X, Y = [], []
    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d, ])
        Y.append(data[d,])
    return np.array(X), np.array(Y)
step =4
N = 1000
Tp = 800
t = np.arange(0, N)
x = np.sin(0.02 * t) + 2 * np.random.rand(N)
df = pd.DataFrame(x)
plt.plot(df)
values = df.values
train, test = values[0:Tp, :], values[Tp:N, :]
# 맨마지막의 데이터에 추가 적으로 복사로더 데이터 넣기
test = np.append(test, np.repeat(test[-1,], step))
train = np.append(train, np.repeat(train[-1,], step))
# 데이터 자르기
trainX, trainY = convertToMatrix(train, step)
testX, testY = convertToMatrix(test, step)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
testX = np.reshape(testX, (testX.shape[0],testX.shape[1],1))
totalX=np.concatenate([trainX,testX],axis=0)
trainX.shape
trainY.shape
model = Sequential()
model.add(SimpleRNN(50, return_sequences=False, input_shape=(4,1)))
model.add(Dense(1))
model.summary()
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.fit(trainX,trainY,epochs=100, verbose=2)
y=model.predict(totalX)
plt.plot(y)

plt.show()
