from matplotlib.pyplot import clf
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
X=np.array([[0,0],
            [1,0],
            [1,1],
            [0,1]])
orr=np.array([-1,1,1,1])

xor=np.array([-1,1,1,-1])
W=np.zeros(30,dtype=float)
breast_cancer=load_breast_cancer()
X=breast_cancer.data
Y=breast_cancer.target
Y=np.where(Y==0,-1,Y)

b=0
alpha=1 #learning rate(0-1)
theta=0.2 #threshold
epochs=500 
e=1
while e<=epochs:
    print("epoch=",e)
    y_pred=[]
    for i in range(len(X)): #0,1,2,3
        x=X[i]
        t=Y[i]
        y_in=np.sum(x*W)+b
        if y_in>theta:
            y=1
        elif y_in<-theta:
            y=-1
        else:
            y=0
        y_pred.append(y)
        if y!=t:
            b=b+alpha*t
            W=W+alpha*t*x
    print("W=",W,"b=",b)
    y_pred=np.array(y_pred)
    if np.array_equal(Y,y_pred):
        break
    e=e+1

acc=accuracy_score(Y,y_pred)
print("aacuracy:",acc)


