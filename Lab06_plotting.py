import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

data=pd.read_csv('heart.csv')
X=data.drop(['output'],axis=1)
Y=data['output']
print("X = ", X.shape)
print("Y = ",Y.shape)

#############################################################
sc=MinMaxScaler()
X1=sc.fit_transform(X)
X_train,X_test,Y_train,Y_test=train_test_split(X1,Y,test_size=0.2 , random_state=20)

##############################################################

NNmodel=Sequential()
NNmodel.add(Dense(units=30,activation='sigmoid',input_shape=(13,)))
NNmodel.add(Dense(25,activation='relu')) 
NNmodel.add(Dense(units=1,activation='sigmoid')) 
op=SGD(learning_rate=0.3)
NNmodel.compile(loss='binary_crossentropy',optimizer=op,metrics=['accuracy'])
summary=NNmodel.fit(X_train,Y_train,epochs=4,verbose=1)
NNmodel.fit(X_train,Y_train,epochs=10,verbose=1)
result=NNmodel.evaluate(X_test,Y_test)
print("loss : ",result[0])
print("Accuracy : " ,result[1])
pd.DataFrame(summary.history).plot()
plt.show()


