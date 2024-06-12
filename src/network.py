import numpy as np
import os

import tensorflow as tf
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class NNetwork(tf.Module):

    def __init__(self, iterations, ub, lb, X_u_train, u_train, X_f_train,
                 X_u_test, u):

        super().__init__()
        self.X_u_test = X_u_test
        self._layers = np.array([2, 20, 20, 20, 20, 20, 20, 20, 20, 1])
        self.X_u_train = X_u_train
        self.X_f_train = X_f_train
        self.u_train = u_train
        self.pbar = tqdm(total=iterations, position=0, leave=True)
        self.ub = ub
        self.lb = lb
        self.u = u
        self._weights = []  # Weights and biases
        self._depth = len(self._layers) - 1
        self._parameters = 0  # total number of parameters

        for i in range(self._depth):
            in_dim = self._layers[i]
            out_dim = self._layers[i + 1]

            # Xavier standard deviation
            std_dv = np.sqrt((2.0 / (in_dim + out_dim)))

            # weights = normal distribution * Xavier standard deviation + 0
            w = tf.random.normal([in_dim, out_dim], dtype='float64') * std_dv

            w = tf.Variable(w, trainable=True, name='w' + str(i + 1))

            b = tf.Variable(tf.cast(tf.zeros([out_dim]), dtype='float64'),
                            trainable=True,
                            name='b' + str(i + 1))

            self._weights.append(w)
            self._weights.append(b)

            self._parameters += in_dim * out_dim + out_dim

    def evaluate(self, x):

        x = (x - self.lb) / (self.ub - self.lb)

        a = x

        for i in range(len(self._layers) - 2):
            z = tf.add(tf.matmul(a, self._weights[2 * i]),
                       self._weights[2 * i + 1])
            a = tf.nn.tanh(z)

        a = tf.add(
            tf.matmul(a, self._weights[-2]),
            self._weights[-1])  # For regression, no activation to last layer
        return a

    def get_weights(self):

        parameters_1d = []  # [.... W_i,b_i.....  ] 1d array

        for i in range(self._depth):
            w_1d = tf.reshape(self._weights[2 * i], [-1])  # flatten weights
            b_1d = tf.reshape(self._weights[2 * i + 1], [-1])  # flatten biases

            parameters_1d = tf.concat([parameters_1d, w_1d],
                                      0)  # concat weights
            parameters_1d = tf.concat([parameters_1d, b_1d], 0)  # concat biases

        return parameters_1d

    def _set_weights(self, parameters):

        for i in range(self._depth):
            shape_w = tf.shape(
                self._weights[2 * i]).numpy()  # shape of the weight tensor
            size_w = tf.size(
                self._weights[2 * i]).numpy()  # size of the weight tensor

            shape_b = tf.shape(
                self._weights[2 * i + 1]).numpy()  # shape of the bias tensor
            size_b = tf.size(
                self._weights[2 * i + 1]).numpy()  # size of the bias tensor

            pick_w = parameters[0:size_w]  # pick the weights
            self._weights[2 * i].assign(tf.reshape(pick_w, shape_w))  # assign
            parameters = np.delete(parameters, np.arange(size_w), 0)  # delete

            pick_b = parameters[0:size_b]  # pick the biases
            self._weights[2 * i + 1].assign(tf.reshape(pick_b,
                                                       shape_b))  # assign
            parameters = np.delete(parameters, np.arange(size_b), 0)  # delete

    def _loss_bc(self, x, y):

        loss_u = tf.reduce_mean(tf.square(y - self.evaluate(x)))
        return loss_u

    def _loss_pde(self, x_to_train_f):

        g = tf.Variable(x_to_train_f, dtype='float64', trainable=False)

        x_f = g[:, 0:1]
        t_f = g[:, 1:2]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)
            tape.watch(t_f)

            g = tf.stack([x_f[:, 0], t_f[:, 0]], axis=1)

            z = self.evaluate(g)

            u_x = tape.gradient(z, x_f)
            u_t = tape.gradient(z, t_f)
            u_xx = tape.gradient(u_x, x_f)

        del tape

        u = (self.evaluate(g))

        f = u_t + u * u_x - (0.01 / np.pi) * u_xx

        loss_f = tf.reduce_mean(tf.square(f))

        return loss_f

    def loss(self, x, y, g):

        loss_u = self._loss_bc(x, y)
        loss_f = self._loss_pde(g)

        loss = loss_u + loss_f

        return loss

    def optimize(self, parameters):

        self._set_weights(parameters)

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)

            loss = self.loss(self.X_u_train, self.u_train, self.X_f_train)

        grads = tape.gradient(loss, self.trainable_variables)

        del tape

        grads_1d = []  # flatten grads

        for i in range(self._depth):
            grads_w_1d = tf.reshape(grads[2 * i], [-1])  # flatten weights
            grads_b_1d = tf.reshape(grads[2 * i + 1], [-1])  # flatten biases

            grads_1d = tf.concat([grads_1d, grads_w_1d],
                                 0)  # concat grad_weights
            grads_1d = tf.concat([grads_1d, grads_b_1d],
                                 0)  # concat grad_biases

        return loss.numpy(), grads_1d.numpy()

    def callback(self, _):

        loss_value = self.loss(self.X_u_train, self.u_train, self.X_f_train)

        u_pred = self.evaluate(self.X_u_test)
        error_vec = np.linalg.norm(
            (self.u - u_pred), 2) / np.linalg.norm(self.u, 2)

        tqdm.write(f'{loss_value.numpy():.16f}, '
                   f'{error_vec:.16f}')
        self.pbar.update(1)
