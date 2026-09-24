import numpy as np
X=np.array([[0,0],[1,0],[1,1],[0,1]])

# Y=np.array([-1,-1,1,-1])
Y=np.array([-1,1,1,1])
W=np.zeros(2,dtype=float)
#W=np.array(object:[0,0],dtype=float)
b=0
count=0
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
        count+=1

    if count==3:
        break
    e=e+1



