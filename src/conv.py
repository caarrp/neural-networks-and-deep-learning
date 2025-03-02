"""conv.py
~~~~~~~~~~

Code for many of the experiments involving convolutional networks in
Chapter 6 of the book 'Neural Networks and Deep Learning', by Michael
Nielsen.  The code essentially duplicates (and parallels) what is in
the text, so this is simply a convenience, and has not been commented
in detail.  Consult the original text for more details.

"""

from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers

import network3
from network3 import sigmoid, tanh, ReLU, Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10

#helper function for tensor flow usage
def shared_to_tf_data(shared_data):
    x, y = shared_data
    x = x.eval()
    y = y.eval()
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(mini_batch_size)

# Convert data to TensorFlow format
train_dataset = shared_to_tf_data(training_data)
val_dataset = shared_to_tf_data(validation_data)
test_dataset = shared_to_tf_data(test_data)

def shallow(n=3, epochs=60):
    nets = []
    for j in range(n):
        print("A shallow net with 100 hidden neurons")
        model = models.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(100, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
        nets.append(model)
    return nets


def basic_conv(n=3, epochs=60):
    for j in range(n):
        print("Conv + FC architecture")
        model = models.Sequential([
            layers.Conv2D(20, (5, 5), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
    return model

def omit_FC():
    for j in range(3):
        print("Conv only, no FC")
        model = models.Sequential([
            layers.Conv2D(20, (5, 5), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_dataset, epochs=60, validation_data=val_dataset)
    return model

def dbl_conv(activation_fn=sigmoid):
    for j in range(3):
        print("Conv + Conv + FC architecture")
        model = models.Sequential([
            layers.Conv2D(20, (5, 5), activation=activation_fn, input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(40, (5, 5), activation=activation_fn),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(100, activation=activation_fn),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_dataset, epochs=60, validation_data=val_dataset)
    return model


# The following experiment was eventually omitted from the chapter,
# but I've left it in here, since it's an important negative result:
# basic l2 regularization didn't help much.  The reason (I believe) is
# that using convolutional-pooling layers is already a pretty strong
# regularizer.
def regularized_dbl_conv():
    for lmbda in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        for j in range(3):
            print("Conv + Conv + FC num %s, with regularization %s" % (j, lmbda))
            model = models.Sequential([
                layers.Conv2D(20, (5, 5), activation='relu', input_shape=(28, 28, 1)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(40, (5, 5), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lmbda)),
                layers.Dense(10, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(train_dataset, epochs=60, validation_data=val_dataset)


def dbl_conv_relu():
    for lmbda in [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        for j in range(3):
            print("Conv + Conv + FC num %s, relu, with regularization %s" % (j, lmbda))
            model = models.Sequential([
                layers.Conv2D(20, (5, 5), activation='relu', input_shape=(28, 28, 1)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(40, (5, 5), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lmbda)),
                layers.Dense(10, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(train_dataset, epochs=60, validation_data=val_dataset)

#### Some subsequent functions may make use of the expanded MNIST
#### data.  That can be generated by running expand_mnist.py.


def expanded_data(n=100):
    """n is the number of neurons in the fully-connected layer. We'll try
    n=100, 300, and 1000.
    """

    # Load expanded training data
    expanded_training_data, _, _ = network3.load_data_shared("../data/mnist_expanded.pkl.gz")
    
    # Convert Theano shared data to TensorFlow Dataset
    train_dataset = shared_to_tf_data(expanded_training_data)
    
    # Train the model 3 times
    for j in range(3):
        print("Training with expanded data, %s neurons in the FC layer, run num %s" % (n, j))
        
        # Define the model
        model = models.Sequential([
            layers.Conv2D(20, (5, 5), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(40, (5, 5), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(n, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train the model
        model.fit(train_dataset, epochs=60, validation_data=val_dataset)
    
    return model


def expanded_data_double_fc(n=100):
    """n is the number of neurons in both fully-connected layers.  We'll
    try n=100, 300, and 1000.

    """

    expanded_training_data, _, _ = network3.load_data_shared("../data/mnist_expanded.pkl.gz")
    train_dataset = shared_to_tf_data(expanded_training_data)
    for j in range(3):
        print("Training with expanded data, %s neurons in two FC layers, run num %s" % (n, j))
        model = models.Sequential([
            layers.Conv2D(20, (5, 5), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(40, (5, 5), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(n, activation='relu'),
            layers.Dense(n, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_dataset, epochs=60, validation_data=val_dataset)


def double_fc_dropout(p0, p1, p2, repetitions):
    expanded_training_data, _, _ = network3.load_data_shared("../data/mnist_expanded.pkl.gz")
    train_dataset = shared_to_tf_data(expanded_training_data)
    nets = []
    for j in range(repetitions):
        print("\n\nTraining using a dropout network with parameters ", p0, p1, p2)
        print("Training with expanded data, run num %s", j)
        model = models.Sequential([
            layers.Conv2D(20, (5, 5), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(40, (5, 5), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(1000, activation='relu'),
            layers.Dropout(p0),
            layers.Dense(1000, activation='relu'),
            layers.Dropout(p1),
            layers.Dense(10, activation='softmax'),
            layers.Dropout(p2)
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_dataset, epochs=40, validation_data=val_dataset)
        nets.append(model)
    return nets


def ensemble(nets):
    """Takes as input a list of nets, and then computes the accuracy on
    the test data when classifications are computed by taking a vote
    amongst the nets.  Returns a tuple containing a list of indices
    for test data which is erroneously classified, and a list of the
    corresponding erroneous predictions.

    Note that this is a quick-and-dirty kluge: it'd be more reusable
    (and faster) to define a {Theano redacted} function taking the vote.  But
    this works.

    """
    test_x, test_y = test_data
    test_x = test_x.eval()
    test_y = test_y.eval()
    predictions = []
    for net in nets:
        preds = net.predict(test_x)
        predictions.append(np.argmax(preds, axis=1))
    all_test_predictions = list(zip(*predictions))
    def plurality(p): return Counter(p).most_common(1)[0][0]
    plurality_test_predictions = [plurality(p) for p in all_test_predictions]
    error_locations = [j for j in range(10000) if plurality_test_predictions[j] != test_y[j]]
    erroneous_predictions = [plurality(all_test_predictions[j]) for j in error_locations]
    print("Accuracy is {:.2f}%".format((1 - len(error_locations) / 10000.0) * 100))
    return error_locations, erroneous_predictions

def plot_errors(error_locations, erroneous_predictions=None):
    test_x, test_y = test_data[0].eval(), test_data[1].eval()
    fig = plt.figure()
    error_images = [np.array(test_x[i]).reshape(28, -1) for i in error_locations]
    n = min(40, len(error_locations))
    for j in range(n):
        ax = plt.subplot2grid((5, 8), (j/8, j % 8))
        ax.matshow(error_images[j], cmap = matplotlib.cm.binary)
        ax.text(24, 5, test_y[error_locations[j]])
        if erroneous_predictions:
            ax.text(24, 24, erroneous_predictions[j])
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    return plt
    
def plot_filters(net, layer, x, y):

    """Plot the filters for net after the (convolutional) layer number
    layer.  They are plotted in x by y format.  So, for example, if we
    have 20 filters after layer 0, then we can call show_filters(net, 0, 5, 4) to
    get a 5 by 4 plot of all filters."""
    filters = net.layers[layer].w.eval()
    fig = plt.figure()
    for j in range(len(filters)):
        ax = fig.add_subplot(y, x, j)
        ax.matshow(filters[j][0], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    return plt


#### Helper method to run all experiments in the book

def run_experiments():

    """Run the experiments described in the book.  Note that the later
    experiments require access to the expanded training data, which
    can be generated by running expand_mnist.py.

    """
    shallow()
    basic_conv()
    omit_FC()
    dbl_conv(activation_fn=sigmoid)
    # omitted, but still interesting: regularized_dbl_conv()
    dbl_conv_relu()
    expanded_data(n=100)
    expanded_data(n=300)
    expanded_data(n=1000)
    expanded_data_double_fc(n=100)    
    expanded_data_double_fc(n=300)
    expanded_data_double_fc(n=1000)
    nets = double_fc_dropout(0.5, 0.5, 0.5, 5)
    # plot the erroneous digits in the ensemble of nets just trained
    error_locations, erroneous_predictions = ensemble(nets)
    plt = plot_errors(error_locations, erroneous_predictions)
    plt.savefig("ensemble_errors.png")
    # plot the filters learned by the first of the nets just trained
    plt = plot_filters(nets[0], 0, 5, 4)
    plt.savefig("net_full_layer_0.png")
    plt = plot_filters(nets[0], 1, 8, 5)
    plt.savefig("net_full_layer_1.png")