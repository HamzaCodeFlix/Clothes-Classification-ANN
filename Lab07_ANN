from tabnanny import verbose
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Activation
import matplotlib.pyplot as plt
import pandas as pd

# creating a neural network model

(trainX, trainY), (testX, testY) = mnist.load_data()
print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)
print(trainY)

plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(trainX[i], cmap='gray')
plt.show()
############################################################
trainX = trainX.reshape((60000,784))
testX = testX.reshape((10000,784))
trainY=to_categorical(trainY,num_classes=10)
testY=to_categorical(testY,num_classes=10)

###########################################################

model=Sequential(
    [
        Dense(45, input_dim=784),
        Activation('relu'),
        Dense(10),
        Activation('softmax')
    ]
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX,trainY, epochs=5, batch_size=30, verbose=1)
performance=model.evaluate(testX,testY)
print("accuracy: ", performance[1])
print("loss: ", performance[0])
pd.DataFrame(model.history.history).plot()
plt.show()
model.save('my_model.keras')

