from keras.models import Sequential
import sys
import time
from keras.optimizers import Adam, SGD
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dropout
from keras.models import Model
import seaborn as sns
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import *
from keras.engine.topology import Layer
from keras import backend as K
from keras.regularizers import l2
from keras_preprocessing.image import ImageDataGenerator
import cv2
import os
from skimage import io
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import numpy.random as rng
from sklearn.utils import shuffle
from load_data import *

K.set_image_data_format('channels_last')

datagen = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    # zca_whitening=True,
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    shear_range=0.2,  # set range for random shear
    zoom_range=0.1,  # set range for random zoom
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # rescale=1. / 255,
)

np.set_printoptions(threshold=sys.maxsize)

evaluate_every = 10  # interval for evaluating on one-shot tasks
loss_every = 20  # interval for printing loss (iterations)
batch_size = 64  # 32 to 64
n_iter = 20000
N_way = 20  # how many classes for testing one-shot tasks>
n_val = 250  # how many one-shot tasks to validate on?
best = -1


def initialize_weights(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def initialize_bias(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                     kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3),
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    model.add(Dropout(0.4))
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    return siamese_net


model = get_siamese_model((105, 105, 1))
model.summary()
# optimizer = SGD(0.0004, momentum=0.6, nesterov=True, decay=0.0003 // n_iter)
optimizer = Adam(lr=0.00006)
model.compile(loss="binary_crossentropy", optimizer=optimizer)


class SiameseLoader:
    """For loading batches and testing tasks to a siamese net"""

    def __init__(self, path, data_subsets=["train", "val"]):
        self.data = {}
        self.categories = {}
        self.info = {}

        for name in data_subsets:
            file_path = os.path.join(path, name + ".pickle")
            print("loading data from {}".format(file_path))
            with open(file_path, "rb") as f:
                (X, c) = pickle.load(f)
                self.data[name] = X
                self.categories[name] = c

    def get_batch(self, batch_size, s="train"):
        """Create batch of n pairs, half same class, half different class"""
        X = self.data[s]
        n_classes, n_examples, w, h = X.shape

        # randomly sample several classes to use in the batch
        categories = rng.choice(n_classes, size=(batch_size,), replace=False)
        # initialize 2 empty arrays for the input image batch
        pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]
        # initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets = np.zeros((batch_size,))
        targets[batch_size // 2:] = 1
        for i in range(batch_size):
            category = categories[i]
            idx_1 = rng.randint(0, n_examples)
            pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, 1)
            idx_2 = rng.randint(0, n_examples)
            # pick images of same class for 1st half, different for 2nd
            if i >= batch_size // 2:
                category_2 = category
            else:
                # add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                category_2 = (category + rng.randint(1, n_classes)) % n_classes
            pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, 1)
        return pairs, targets

    def generate(self, batch_size, s="train"):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size, s)
            yield (pairs, targets)

    def make_oneshot_task(self, N, s="val", language=None):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        X = self.data[s]
        n_classes, n_examples, w, h = X.shape
        indices = rng.randint(0, n_examples, size=(N,))
        if language is not None:
            low, high = self.categories[s][language]
            if N > high - low:
                raise ValueError("This language ({}) has less than {} letters".format(language, N))
            categories = rng.choice(range(low, high), size=(N,), replace=False)

        else:  # if no language specified just pick a bunch of random letters
            categories = rng.choice(range(n_classes), size=(N,), replace=False)
        true_category = categories[0]
        ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))
        test_image = np.asarray([X[true_category, ex1, :, :]] * N).reshape(N, w, h, 1)
        support_set = X[categories, indices, :, :]
        support_set[0, :, :] = X[true_category, ex2]
        support_set = support_set.reshape(N, w, h, 1)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image, support_set]

        return pairs, targets

    def test_oneshot(self, model, N, k, s="val", verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k, N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N, s)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct += 1
        percent_correct = (100.0 * n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct, N))
        return percent_correct

    def train(self, model, epochs, verbosity):
        model.fit_generator(self.generate(batch_size))


loader = SiameseLoader(data_path)


def concat_images(X):
    """Concatenates a bunch of images into a big matrix for plotting purposes."""
    nc, h, w, _ = X.shape
    X = X.reshape(nc, h, w)
    n = np.ceil(np.sqrt(nc)).astype("int8")
    img = np.zeros((n*w, n*h))
    x = 0
    y = 0
    for example in range(nc):
        img[x*w:(x+1)*w, y*h:(y+1)*h] = X[example]
        y += 1
        if y >= n:
            y = 0
            x += 1
    return img


def plot_oneshot_task(pairs):
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.matshow(pairs[0][0].reshape(105, 105), cmap='gray')
    img = concat_images(pairs[1])
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.matshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
#
#
# pairs, targets = loader.make_oneshot_task(20, "train", "Japanese_(katakana)")
# plot_oneshot_task(pairs)


weights_path = 'save_weight/'
print("Starting training process!")
print("-------------------------------------")
t_start = time.time()

# weights_path_0 = os.path.join(weights_path, "model_weights_64_dropout.h5")
# model.load_weights(weights_path_0)

for i in range(1, n_iter):
    (inputs, targets) = loader.get_batch(batch_size)
    # loss = model.train_on_batch(inputs, targets)
    # datagen.fit(inputs[0])
    # datagen.fit(inputs[1])
    his = model.fit_generator(generator=datagen.flow(x=inputs, y=targets, batch_size=batch_size), steps_per_epoch=1, verbose=2)
    # his = model.fit_generator(loader.generate(batch_size), steps_per_epoch=1)
    loss = his.history['loss'][0]
    # print("\n ------------- \n")
    # print("Loss: {0}".format(loss))
    if i % evaluate_every == 0:
        print("Time for {0} iterations: {1}".format(i, time.time() - t_start))
        val_acc = loader.test_oneshot(model, N_way, n_val, verbose=True)
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            print("Saving weights to: {0} \n".format(weights_path))
            model.save_weights(filepath=os.path.join(weights_path, 'model_weights_64_dropout.h5'))
            best = val_acc

    if i % loss_every == 0:
        print("iteration {}, training loss: {:.2f},".format(i, loss))

weights_path_2 = os.path.join(weights_path, "model_weights_64_dropout.h5")
model.load_weights(weights_path_2)


def nearest_neighbour_correct(pairs, targets):
    """returns 1 if nearest neighbour gets the correct answer for a one-shot task
        given by (pairs, targets)"""
    L2_distances = np.zeros_like(targets)
    for i in range(len(targets)):
        L2_distances[i] = np.sum(np.sqrt(pairs[0][i] ** 2 - pairs[1][i] ** 2))
    if np.argmin(L2_distances) == np.argmax(targets):
        return 1
    return 0


def test_nn_accuracy(N_ways, n_trials, loader):
    """Returns accuracy of one shot """
    print("Evaluating nearest neighbour on {} unique {} way one-shot learning tasks ...".format(n_trials, N_ways))

    n_right = 0

    for i in range(n_trials):
        pairs, targets = loader.make_oneshot_task(N_ways, "val")
        correct = nearest_neighbour_correct(pairs, targets)
        n_right += correct
    return 100.0 * n_right / n_trials


ways = np.arange(1, 30, 2)
resume = False
val_accs, train_accs, nn_accs = [], [], []
trials = 450
for N in ways:
    val_accs.append(loader.test_oneshot(model, N, trials, "val", verbose=True))
    train_accs.append(loader.test_oneshot(model, N, trials, "train", verbose=True))
    nn_accs.append(test_nn_accuracy(N, trials, loader))

print(np.mean(val_accs))

# plot the accuracy vs num categories for each
plt.plot(ways, val_accs, "m")
plt.plot(ways, train_accs, "y")
plt.plot(ways, nn_accs, "c")

plt.plot(ways, 100.0 / ways, "r")
plt.show()

fig, ax = plt.subplots(1)
ax.plot(ways, val_accs, "m", label="Siamese(val set)")
ax.plot(ways, train_accs, "y", label="Siamese(train set)")
plt.plot(ways, nn_accs, label="Nearest neighbour")

ax.plot(ways, 100.0/ways, "g", label="Random guessing")
plt.xlabel("Number of possible classes in one-shot tasks")
plt.ylabel("% Accuracy")
plt.title("Omiglot One-Shot Learning Performance of a Siamese Network")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
inputs, targets = loader.make_oneshot_task(20, "val")
plt.show()

plot_oneshot_task(inputs)