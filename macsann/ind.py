# -*- coding: utf-8 -*-
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from random import randint


'''individuals class'''
class ind:
    def __init__(self, layers, activation_func, optimi, input_shape, last_activation,
                 metric_func, loss_func, output_neurons, max_hidden_neurons,
                 dropout_rate, ind_id, weights=None):
        self.layers = layers
        self.activation_func = activation_func
        self.optimi = optimi
        self.input_shape = input_shape
        self.last_activation = last_activation
        self.metric_func = metric_func
        self.loss_func = loss_func
        self.output_neurons = output_neurons
        self.max_hidden_neurons = max_hidden_neurons
        self.dropout_rate = dropout_rate
        self.weights = weights
        self.ind_id = ind_id
        self.metric = None
        self.fitness = None

    def build_weights(self):
        model = Sequential()
        for i in range(0, len(self.weights), 2):
            if i == 0:
                model.add(Dense(
                    self.weights[i].shape[1], activation=self.activation_func, input_shape=self.input_shape))
                model.add(Dropout(self.dropout_rate))
            elif i == len(self.weights)-2:
                # capa de salida
                model.add(Dense(self.output_neurons, activation=self.last_activation))
            else:
                model.add(
                    Dense(self.weights[i].shape[1], activation=self.activation_func))
                model.add(Dropout(self.dropout_rate))

        model.compile(loss=self.loss_func,
                      optimizer=self.optimi,
                      metrics=[self.metric_func])
        model.set_weights(self.weights)
        return model

    def __del__(self):
        self.modelo = None

    def build_model(self):
        model = Sequential()
        for i in range(self.layers):
            NEURONS = randint(1, self.max_hidden_neurons)
            if i == 0:
                model.add(Dense(NEURONS, activation=self.activation_func,
                                input_shape=self.input_shape, bias_initializer='random_uniform'))
                model.add(Dropout(self.dropout_rate))
            else:
                model.add(Dense(NEURONS, activation=self.activation_func,
                                bias_initializer='random_uniform'))
                model.add(Dropout(self.dropout_rate))
        # capa de salida
        model.add(Dense(self.output_neurons, activation=self.last_activation,
                        bias_initializer='random_uniform'))

        model.compile(loss=self.loss_func,
                      optimizer=self.optimi,
                      metrics=[self.metric_func])
        return model
