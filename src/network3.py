"""network3.py
~~~~~~~~~~~~~~

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

Written for Theano 0.6 and 0.7, needs some changes for more recent
versions of Theano.

"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from keras import activations
relu = activations.relu
sigmoid = activations.sigmoid
softmax = activations.softmax
tanh = activations.tanh

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return tf.maximum(0.0, z)

mini_batch_size = 10

#### Constants

#### Load the MNIST data

#### Load the MNIST data
def load_data(filename="../data/mnist.pkl.gz"):
    """Load the MNIST data from a .pkl.gz file and return it as TensorFlow datasets."""
    with gzip.open(filename, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    
    def to_tf_dataset(data):
        """Convert NumPy arrays into TensorFlow datasets."""
        x, y = data
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(mini_batch_size)
    
    return [to_tf_dataset(training_data), to_tf_dataset(validation_data), to_tf_dataset(test_data)]
#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.model = self.build_model()

    def build_model(self):
        """Build the TensorFlow model using the specified layers."""
        model = models.Sequential()
        for layer in self.layers:
            model.add(layer)
        return model

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=eta),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        
        # Train the model
        history = self.model.fit(
            training_data,
            epochs=epochs,
            validation_data=validation_data
        )

        # Evaluate on test data
        test_loss, test_accuracy = self.model.evaluate(test_data)
        print(f"Test accuracy: {test_accuracy:.2%}")

   
#### Define layer types

class ConvPoolLayer(layers.Layer):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
    """Combination of a convolutional and a max-pooling layer."""
    def __init__(self, filters, kernel_size, pool_size, activation_fn):
        super(ConvPoolLayer, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, activation=activation_fn)
        self.pool = layers.MaxPooling2D(pool_size)

class FullyConnectedLayer(layers.Layer):
    """Fully connected layer with optional dropout."""
    def __init__(self, units, activation_fn, dropout_rate=0.0):
        super(FullyConnectedLayer, self).__init__()
        self.dense = layers.Dense(units, activation=activation_fn)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs):
        x = self.conv(inputs)
        return self.pool(x)

class SoftmaxLayer(layers.Layer):
    """Softmax output layer."""
    def __init__(self, units):
        super(SoftmaxLayer, self).__init__()
        self.dense = layers.Dense(units, activation='softmax')

    def call(self, inputs):
        return self.dense(inputs)
     # no dropout in the convolutional layers

#### Miscellanea
def size(data):
    """Return the size of the dataset `data`."""
    return len(data)

def dropout_layer(inputs, dropout_rate):
    """Apply dropout to the inputs."""
    return layers.Dropout(dropout_rate)(inputs)