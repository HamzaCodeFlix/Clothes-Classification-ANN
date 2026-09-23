from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv("images28.csv")
print(data.shape)
print(data.head())
X = data.iloc[:, 1:].values
Y = data.iloc[:, 0].values
print("X shape:", X.shape)
print("Y shape:", Y.shape)
X = X / 255.0
trainX, testX, trainY, testY = train_test_split(X,Y,test_size=0.2,random_state=42)
num_classes = len(set(Y))
print("Number of classes:", num_classes)
trainY = to_categorical(trainY, num_classes=num_classes)
testY = to_categorical(testY, num_classes=num_classes)

model = Sequential(
    [
        Input(shape=(2352,)),
        Dense(45),
        Activation('relu'),
        Dense(num_classes),
        Activation('softmax')
    ]
)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(trainX,trainY,epochs=5,batch_size=30,verbose=1)
performance = model.evaluate(testX, testY)
print("Accuracy:", performance[1])
print("Loss:", performance[0])
pd.DataFrame(model.history.history).plot()
plt.show()
model.save('my_modeldataclothes.keras')
