from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

alldata=load_breast_cancer()
X=alldata.data
Y=alldata.target
print("Shape of X:",X.shape)
print("Shape of Y:", Y.shape)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
scalar=StandardScaler()
X_train=scalar.fit_transform(X_train)
X_test=scalar.transform(X_test)
clf=Perceptron(max_iter=500,eta0=1.0,random_state=42)  
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
print("Predicted values:",y_pred)
acc=accuracy_score(Y_test,y_pred)
print("Accuracy:",acc)









