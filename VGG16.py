from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from keras.utils import img_to_array, load_img, to_categorical
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("Images224.csv")

print("Shape of data:", data.shape)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

X = X / 255.0

print("Minimum value:", X.min())
print("Maximum value:", X.max())

X = X.reshape(-1, 224, 224, 3)

print("Shape of X after reshape:", X.shape)

trainX, testX, trainY, testY = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", trainX.shape)
print("Testing data:", testX.shape)

trainY = to_categorical(trainY, num_classes=4)
testY = to_categorical(testY, num_classes=4)

model = Sequential([
    Conv2D(64, (2, 2), input_shape=(224, 224, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (2, 2)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (2, 2)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(512, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(512, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Activation('relu'),

    Dense(4096),
    Dense(4096),

    Dense(4),
    Activation('softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    trainX,
    trainY,
    epochs=5,
    batch_size=20,
    verbose=1
)

train_performance = model.evaluate(trainX, trainY, verbose=0)

print("\n========== TRAINING PERFORMANCE ==========")
print("Training accuracy:", train_performance[1])
print("Training loss:", train_performance[0])

test_performance = model.evaluate(testX, testY, verbose=0)

print("\n========== TESTING PERFORMANCE ==========")
print("Testing accuracy:", test_performance[1])
print("Testing loss:", test_performance[0])

print("\n========== MODEL SCORES ==========")
print("Training score:", train_performance[1])
print("Testing score:", test_performance[1])

print("\n========== FINAL RESULTS ==========")
print("Training accuracy:", train_performance[1])
print("Training accuracy percentage:", train_performance[1] * 100, "%")

print("Testing accuracy:", test_performance[1])
print("Testing accuracy percentage:", test_performance[1] * 100, "%")

print("Training loss:", train_performance[0])
print("Testing loss:", test_performance[0])

model.save("VGG.keras")

print("\nVGG model saved successfully!")

plt.figure(figsize=(10, 6))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['loss'], label='Training Loss')

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("VGG Model Training Accuracy and Loss")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))

plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("VGG Model Accuracy - 5 Epochs")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))

plt.plot(history.history['loss'], label='Training Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VGG Model Loss - 5 Epochs")
plt.legend()
plt.grid()
plt.show()

img = load_img(r"C:\Users\Muhammad Hamza\Pictures\shirts.jpg", target_size=(28, 28))
img = img_to_array(img)

img = np.array(img)
img = img / 255.0
img = img.reshape(1, 2352)

prediction = model.predict(img)

print("Prediction:", np.argmax(prediction))
print("Confidence:", np.max(prediction) * 100, "%")

plt.imshow(img.reshape(28, 28, 3))
plt.show()
