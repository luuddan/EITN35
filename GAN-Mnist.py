import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


#Import
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy.random import randint
from numpy.random import rand
from numpy import zeros
from numpy import ones
from matplotlib import pyplot
import random
from numpy import vstack
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#Input dimension generator
input_dim = 100

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

discriminator = Sequential()
discriminator.add(Dense(1024, input_dim=784, activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(0.4))
discriminator.add(Dense(512, activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(0.4))
discriminator.add(Dense(256, activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(0.4))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

discriminator.summary()

#Generator-model
generator = Sequential()
generator.add(Dense(256, input_dim=input_dim, activation=LeakyReLU(alpha=0.2)))
generator.add(Dense(512, activation=LeakyReLU(alpha=0.2)))
generator.add(Dense(1024, activation=LeakyReLU(alpha=0.2)))
generator.add(Dense(784, activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
generator.summary()

#GAN-model
discriminator.trainable = False
inputs = Input(shape=(input_dim, ))
hidden = generator(inputs)
output = discriminator(hidden)
gan = Model(inputs, output)
gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
gan.summary()

#Load and preprocess data from MNIST
(x_train, _), (_, _) = mnist.load_data()


x_train = (x_train - 127.5)/127.5
print(x_train.shape)
x_train = x_train.reshape(60000, 784)
print(x_train.shape)

# select real samples
def generate_real_samples(x_train, n_samples):
    # choose random instances
    ix = randint(0, x_train.shape[0], n_samples)
    # retrieve selected images
    X = x_train[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y
# generate n noise samples with class labels
def generate_noise_samples(n_samples):
    # generate uniform random numbers in [0,1]
    X = rand(784 * n_samples)
    # reshape into a batch of grayscale images
    X = X.reshape((n_samples, 784))
    # generate 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(input_dim, n_samples):
    # generate points in the latent space
    x_input = randn(input_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, input_dim)
    return x_input
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    x = model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return x, y


# Train discriminator
def train_discriminator(model, dataset, number_iteration, batch_size):
    half_batch = int(batch_size / 2)
    for i in range(number_iteration):
        # Sample from real images
        x_real, y_real = generate_real_samples(x_train, (half_batch))
        # Sample from noise
        x_noise, y_noise = generate_noise_samples((half_batch))
        # Train with real images
        _, real_acc = model.train_on_batch(x_real, y_real)
        print(x_real.shape)
        # Train with noise
        _, noise_acc = model.train_on_batch(x_noise, y_noise)
        print('>%d real=%.0f%% noise=%.0f%%' % (i + 1, real_acc * 100, noise_acc * 100))


train_discriminator(discriminator, x_train, 25, 256)

n_samples = 25
x, _ = generate_fake_samples(generator, input_dim, n_samples)
x = x.reshape(n_samples, 28, 28)
for i in range(n_samples):
    pyplot.subplot(5, 5, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(x[i], cmap='gray_r')

pyplot.show()

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    x_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))

def train_gan(generator, discriminator, gan, x_train, input_dim, n_epochs, batch_size):
    batch_per_epoch = int(x_train.shape[0]/batch_size)
    half_batch = int(batch_size/2)
    for i in range(n_epochs):
        for j in range(batch_per_epoch):
            x_real, y_real = generate_real_samples(x_train, half_batch)
            x_fake, y_fake = generate_fake_samples(generator, input_dim, half_batch)
            x, y = vstack((x_real, x_fake)), vstack((y_real, y_fake))
            discriminator_loss = discriminator.train_on_batch(x, y)
            x_gan = generate_latent_points(input_dim, batch_size)
            y_gan = ones((batch_size, 1))
            generator_loss = gan.train_on_batch(x_gan, y_gan)
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, batch_per_epoch, discriminator_loss[0], generator_loss[0]))
            # evaluate the model performance, sometimes
            if (j==1):
                summarize_performance(i, generator, discriminator, x_train, input_dim)

train_gan(generator,discriminator,gan,x_train,input_dim, 20, 256)

n_samples = 25
pyplot.figure(figsize=(10, 10))
x, _ = generate_fake_samples(generator, input_dim, n_samples)
x = x.reshape(n_samples, 28, 28)
for i in range(n_samples):
    pyplot.subplot(5, 5, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(x[i], cmap='gray_r')

pyplot.show()

n_samples = 25
pyplot.figure(figsize=(10, 10))
x, _ = generate_real_samples(x_train, n_samples)
x = x.reshape(n_samples, 28, 28)
for i in range(n_samples):
    pyplot.subplot(5, 5, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(x[i], cmap='gray_r')

pyplot.show()