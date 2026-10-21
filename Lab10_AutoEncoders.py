
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model, datasets, losses
from keras.layers import Input, Dense, Flatten, Reshape
from random import randint

data = pd.read_csv("Images224.csv")
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

print("Shape of x_train:", x_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_test:", y_test.shape)

# Designing the Autoencoder model
input_layer = Input(shape=(x_train.shape[1:]))
flattened = Flatten()(input_layer)
hidden = Dense(100, activation='relu')(flattened)
latent = Dense(20, activation='relu')(hidden)
encoder = Model(inputs=input_layer, outputs=latent, name='encoder')

print(encoder.summary())

#######################################################################

# Designing decoder

input_layer_decoder = Input(shape=encoder.output.shape[1:])
upsampled = Dense(100, activation='relu')(input_layer_decoder)
unsampled = Dense(784, activation='relu')(upsampled)
constructed = Reshape(x_train.shape[1:])(unsampled)
decoder = Model(inputs=input_layer_decoder, outputs=constructed, name='decoder')

print(decoder.summary())

###############################################################################

autoencoder = Model(
    inputs=encoder.input,
    outputs=decoder(encoder(input_layer)),
    name='autoencoder'
)

print(autoencoder.summary())

autoencoder.compile(
    optimizer='adam',
    loss=losses.MeanSquaredError()
)

summary = autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=100,
    validation_data=(x_test, x_test)
)

# Display loss graph
plt.figure(figsize=(8, 5))
plt.plot(summary.history['loss'], label='Training Loss')
plt.plot(summary.history['val_loss'], label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

###############################################################################

# Display original and reconstructed images

fig, axs = plt.subplots(3, 2, figsize=(10, 15))

for i in range(3):
    sample = x_test[randint(0, x_test.shape[0] - 1)]

    reconstructed = autoencoder.predict(
        np.expand_dims(sample, axis=0),
        verbose=0
    )

    axs[i][0].imshow(sample, cmap='gray')
    axs[i][0].set_title('Original Image')
    axs[i][0].axis('off')

    axs[i][1].imshow(reconstructed[0], cmap='gray')
    axs[i][1].set_title('Reconstructed Image')
    axs[i][1].axis('off')

plt.tight_layout()
plt.show()


