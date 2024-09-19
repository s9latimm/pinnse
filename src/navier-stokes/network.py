import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class Network(tf.Module):

    def __init__(self, layers, iterations, border, train):

        super().__init__()
        self._layers = layers

        # training
        self._x_train = train

        # border
        self._border = border

        # network
        self._activation = tf.nn.tanh
        self._pbar = tqdm(total=iterations, position=0, leave=True)
        self._depth = len(self._layers) - 1

        self.u_err = []
        self.v_err = []
        self.f_err = []
        self.g_err = []
        self.m_err = []

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

        self._viscosity = 0.08
        self._density = 1.

    def evaluate(self, x_train):
        g = tf.Variable(x_train, dtype='float64', trainable=False)

        x = g[:, 0:1]
        y = g[:, 1:2]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)

            psi = self.forward(tf.stack([x[:, 0], y[:, 0]], axis=1))
            u = psi[:, 0:1]
            v = psi[:, 1:2]
            p = psi[:, 2:3]

            # psi = psi_and_p[:, 0:1]
            # p = psi_and_p[:, 1:2]
            #
            # u = tape.gradient(psi, y)
            # v = -tape.gradient(psi, x)

            u_x = tape.gradient(u, x)
            u_xx = tape.gradient(u_x, x)
            u_y = tape.gradient(u, y)
            u_yy = tape.gradient(u_y, y)

            v_x = tape.gradient(v, x)
            v_xx = tape.gradient(v_x, x)
            v_y = tape.gradient(v, y)
            v_yy = tape.gradient(v_y, y)

            p_x = tape.gradient(p, x)
            p_y = tape.gradient(p, y)

        del tape

        f = self._density * (u * u_x +
                             v * u_y) + p_x - self._viscosity * (u_xx + u_yy)
        g = self._density * (u * v_x +
                             v * v_y) + p_y - self._viscosity * (v_xx + v_yy)

        m = v_x - v_y

        return u, v, p, f, g, m

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

    def _loss_border(self):
        u, v, p, f, g, m_pred = self.evaluate(self._border)

        u_err = tf.reduce_sum(tf.square(self._border[:, 2:3] - u))
        v_err = tf.reduce_sum(tf.square(self._border[:, 3:4] - v))

        self.u_err.append(u_err.numpy())
        self.v_err.append(v_err.numpy())

        return u_err + v_err

    def _loss_pde(self):
        _, _, _, f, g, m = self.evaluate(self._x_train)

        f_err = tf.reduce_sum(tf.square(f))
        g_err = tf.reduce_sum(tf.square(g))
        m_err = tf.reduce_sum(tf.square(m))

        self.f_err.append(f_err.numpy())
        self.g_err.append(g_err.numpy())
        self.m_err.append(m_err.numpy())

        return f_err + g_err + m_err

    def loss(self):
        loss_u = self._loss_border()
        loss_f = self._loss_pde()
        return loss_f + loss_u

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

    def callback(self, loss):
        # self._pbar.write(f'loss: {loss}')
        self._pbar.update(1)
