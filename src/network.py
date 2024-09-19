from abc import abstractmethod

import numpy as np
import tensorflow as tf


class BaseNetwork(tf.Module):

    @abstractmethod
    def evaluate(self):
        ...

    @abstractmethod
    def loss(self):
        ...

    def __init__(self, layers, activation):

        super().__init__()
        self._layers = layers

        # network
        self._activation = activation
        self._depth = len(self._layers) - 1

        # initialize weights and biases
        self._weights = []
        for i in range(self._depth):
            input_dim = self._layers[i]
            output_dim = self._layers[i + 1]
            std_dv = np.sqrt((2.0 / (input_dim + output_dim)))
            w = tf.random.normal([input_dim, output_dim],
                                 dtype='float64') * std_dv
            w = tf.Variable(w, trainable=True, name='w' + str(i + 1))
            b = tf.Variable(tf.cast(tf.zeros([output_dim]), dtype='float64'),
                            trainable=True,
                            name='b' + str(i + 1))
            self._weights.append(w)
            self._weights.append(b)

    def forward(self, x):
        y = x
        for i in range(self._depth - 1):
            y = self._activation(
                tf.add(tf.matmul(y, self._weights[2 * i]),
                       self._weights[2 * i + 1]))
        return tf.add(tf.matmul(y, self._weights[-2]), self._weights[-1])

    def get_weights(self):
        parameters_1d = []
        for i in range(self._depth):
            w_1d = tf.reshape(self._weights[2 * i], [-1])
            b_1d = tf.reshape(self._weights[2 * i + 1], [-1])
            parameters_1d = tf.concat([parameters_1d, w_1d], 0)
            parameters_1d = tf.concat([parameters_1d, b_1d], 0)
        return parameters_1d

    def set_weights(self, parameters):
        for i in range(self._depth):
            shape_w = tf.shape(self._weights[2 * i]).numpy()
            size_w = tf.size(self._weights[2 * i]).numpy()
            shape_b = tf.shape(self._weights[2 * i + 1]).numpy()
            size_b = tf.size(self._weights[2 * i + 1]).numpy()
            pick_w = parameters[0:size_w]
            self._weights[2 * i].assign(tf.reshape(pick_w, shape_w))
            parameters = np.delete(parameters, np.arange(size_w), 0)
            pick_b = parameters[0:size_b]
            self._weights[2 * i + 1].assign(tf.reshape(pick_b, shape_b))
            parameters = np.delete(parameters, np.arange(size_b), 0)

    def optimize(self, parameters):
        self.set_weights(parameters)
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            loss = self.loss()
        grads = tape.gradient(loss, self.trainable_variables)
        del tape

        grads_1d = []
        for i in range(self._depth):
            grads_w_1d = tf.reshape(grads[2 * i], [-1])
            grads_b_1d = tf.reshape(grads[2 * i + 1], [-1])
            grads_1d = tf.concat([grads_1d, grads_w_1d], 0)
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0)

        return loss.numpy(), grads_1d.numpy()
