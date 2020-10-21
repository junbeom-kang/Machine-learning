from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

#데이터 만들기
first=np.random.randint(100,size=(100,3))
second=np.random.randint(50,100,size=(100,3))
make_zero=np.zeros(100).reshape(100,1)
make_one=np.ones(100).reshape(100,1)

#데이터 합치기
newdata=np.hstack((np.vstack((first,second)),np.vstack((make_zero,make_one))))

#데이터 x부분과 y부분 분리하기
X=newdata[:,:3]
y=newdata[:,3]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
#로지스틱 리그레션모델 만들기
model = LogisticRegression(max_iter=100)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print(accuracy_score(Y_test,y_pred))
#print(model.score(X_test,y_pred))
#print(model.score(y_pred,Y_test))
