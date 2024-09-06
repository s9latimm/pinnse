import numpy as np
import os

import tensorflow as tf
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class Network(tf.Module):

    def __init__(self, iterations, border, train, loss):

        super().__init__()
        self._layers = np.array([2, 20, 20, 20, 20, 20, 20, 20, 20, 1])

        # border points
        self._f_border = border

        # training
        self._x_train = train

        # loss function
        self._loss_pde = loss

        # network
        self._activation = tf.nn.tanh
        self._pbar = tqdm(total=iterations, position=0, leave=True)
        self._depth = len(self._layers) - 1

        # initialize network
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

    def evaluate(self, x):
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

    def _set_weights(self, parameters):
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

    def _loss_border(self, border):
        return tf.reduce_mean(
            tf.square(border[:, 2:3] - self.evaluate(border[:, 0:2])))

    def loss(self):
        loss_u = self._loss_border(self._f_border)
        loss_f = self._loss_pde(self, self._x_train)
        return loss_u + loss_f

    def optimize(self, parameters):
        self._set_weights(parameters)
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

    def callback(self, _):
        self._pbar.update(1)
