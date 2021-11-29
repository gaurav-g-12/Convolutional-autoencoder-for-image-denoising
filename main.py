import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def normalize(images):
    images = images.astype('float32')/255.0
    images = np.expand_dims(images, -1)
    return images

def add_noise(images):
    noise_factor = 0.3
    images = images + noise_factor*np.random.normal(loc=0, scale=1, size=(len(images), 28, 28, 1))
    images = np.clip(images, a_min=0, a_max=1)
    return images

def display(image1, image2):
    plt.figure(figsize=(20,4))
    index = np.random.randint(low=0, high=len(image1), size=10)
    array1 = image1[index,:]
    array2 = image2[index,:]
    n=10

    for i in range(n):
        plt.subplot(2,10,i+1)
        plt.imshow(array1[i])

        plt.subplot(2,10,i+1+n)
        plt.imshow(array2[i])
    plt.show()

x_train = normalize(x_train)
x_test = normalize(x_test)

x_test_noise = add_noise(x_test)
x_train_noise = add_noise(x_train)

display(x_train, x_train_noise)


input = tf.keras.Input((28,28,1))
x = tf.keras.layers.Conv2D(32,3,padding='same', activation = 'relu')(input)
x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
x = tf.keras.layers.Conv2D(32,3,padding='same', activation = 'relu')(x)
x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)

x = tf.keras.layers.Conv2DTranspose(32,3,strides=2,padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(32,3,strides=2,padding='same', activation='relu')(x)
output = tf.keras.layers.Conv2D(1,3,activation = 'sigmoid', padding = 'same')(x)

model = tf.keras.Model(inputs=input, outputs=output)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=['accuracy']
)

model.fit(x_train_noise, x_train, epochs=1, shuffle=True)


test_image = x_test_noise[6]
test_image = np.expand_dims(test_image, 0)

model_output = model.predict(test_image)

plt.figure(figsize=(5,5))
for i in range(1):
    plt.subplot(1, 2, 1)
    plt.imshow(test_image[0])

    plt.subplot(1, 2, 2)
    plt.imshow(model_output[0])
plt.show()





