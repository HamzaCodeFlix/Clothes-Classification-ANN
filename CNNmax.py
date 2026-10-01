from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation 
from keras.utils import to_categorical 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
 
 
data = pd.read_csv("Images28.csv", header=None) 
print("Shape of data:", data.shape) 
X = data.iloc[:, 1:].values 
y = data.iloc[:, 0].values 
print("Shape of X:", X.shape) 
print("Shape of y:", y.shape) 
 
# normalization 
 
X = X / 255.0 
print("Minimum value:", X.min()) 
print("Maximum value:", X.max()) 
 
# reshape images 
 
X = X.reshape(-1, 28, 28, 3) 
print("Shape of X after reshape:", X.shape) 
 
# train test split 
 
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42) 
print("Training data:", trainX.shape) 
print("Testing data:", testX.shape) 
 
# converting labels into categorical values 
 
trainY = to_categorical(trainY, num_classes=4) 
testY = to_categorical(testY, num_classes=4) 
 
# creating CNN model 
 
model = Sequential( 
    [ 
        Conv2D(32, (3, 3), input_shape=(28, 28, 3)), 
        Activation('relu'), 
        MaxPooling2D(pool_size=(2, 2)), 
        Conv2D(64, (3, 3)), 
        Activation('relu'), 
        MaxPooling2D(pool_size=(2, 2)), 
        Flatten(), 
        Dense(45), 
        Activation('relu'), 
        Dense(4), 
        Activation('softmax') 
    ] 
) 
 
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) 
model.fit(trainX,trainY,epochs=5,batch_size=30,verbose=1) 
 
# training performance 
 
train_performance = model.evaluate(trainX, trainY) 
print("Training accuracy:", train_performance[1]) 
print("Training loss:", train_performance[0]) 
 
# testing performance 
 
test_performance = model.evaluate(testX, testY) 
print("Testing accuracy:", test_performance[1]) 
print("Testing loss:", test_performance[0]) 
pd.DataFrame(model.history.history).plot() 
plt.show() 
 
model.save("CNNmax.keras") 
print("CNN Max Pooling model saved successfully!")