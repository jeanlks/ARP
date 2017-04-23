# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plot


class NaiveBayesClassifier(object):

    def __init__(self, train_dataset, dataset_cols_names):
        self.train_dataset = train_dataset

        self.classes = []
        self.variables_labels = dataset_cols_names
        self.train_inputs, self.train_outputs = self.get_data(self.train_dataset)
        self.inputs_by_class = False

    def get_data(self, path):

        data_matrix = pd.read_csv(path).as_matrix()
        inputs = []
        outputs = []
        for row in data_matrix:
            inputs.append(row[:4])
            outputs.append(row[4])

        self.classes = np.unique(outputs)
        return inputs, outputs


    @staticmethod
    def gaussian_vectorized(variance, mean, values):
        """
        Método que aplica a função gaussiana em uma variável para um range de valores.
        :type variance: Float
        :param variance: a variancia da variavel
        :type mean: Float
        :param mean: a média da variavel
        :type values: Array of Float
        :param values: os valors de X (array com vários valores para a mesma variavel)
        :return: vetor contendo o resultado da gaussiana para cada valor no vetor de entrada (values)
        """
        result = []
        for value in values:
            result.append(NaiveBayesClassifier.gaussian(variance, mean, value))
        return result

    @staticmethod
    def gaussian_multi(covariance_matrix, means, values):


    def get_inputs_by_class(self):


        if self.inputs_by_class:
            return self.inputs_by_class

        variables_labels = self.variables_labels
        inputs_transpose = np.transpose(self.train_inputs)
        result = {}
        for idx, var_data in enumerate(inputs_transpose):
            if not variables_labels[idx] in result:
                result[variables_labels[idx]] = {}
            for idx_var, value in enumerate(var_data):
                if not self.train_outputs[idx_var] in result[variables_labels[idx]]:
                    result[variables_labels[idx]][self.train_outputs[idx_var]] = []
                result[variables_labels[idx]][self.train_outputs[idx_var]].append(value)

        self.inputs_by_class = result
        return result


    def remove_zeros(self, matrix):
        result = []
        for vector in matrix:
            line = []
            for item in vector:
                val = item
                if val < 0.0001:
                    val = float("NaN")
                line.append(val)
            result.append(line)
        return result

    def plot_two_var_normal(self, vars_combination):


        # criando range 2D para plotar os valores resultantes da gaussiana
        x_values = np.arange(-1, 10, 0.1)
        y_values = np.arange(-1, 10, 0.1)
        x_values, y_values = np.meshgrid(x_values, y_values)

        inputs_by_class = self.get_inputs_by_class()

        fig = plot.figure()
        for class_name in self.classes:
            inputs = []

            # separando apenas as variáveis informadas em um array (inputs)
            for var in vars_combination:
                input_values = inputs_by_class[var][class_name]
                inputs.append(input_values)

            # para cada classe é calculada a matriz de covariancia e as médias das variáveis informadas
            covariance_matrix = np.cov(inputs)
            means = np.mean(inputs, 1)

            # para cada x e y no range, calculando o valor resultante da gaussiana multivariada
            z_values = []
            for x in x_values[0]:
                z_line = []
                for y in x_values[0]:
                    z_line.append(self.gaussian_multi(covariance_matrix, means, [x, y]))
                z_values.append(z_line)

            # adicionando superfice ao gráfico para plotar a gaussiana dessa classe.
            ax = fig.gca(projection='3d')
            plot.xlabel(vars_combination[0])
            plot.ylabel(vars_combination[1])

            ax.plot_surface(y_values, self.remove_zeros(x_values), self.remove_zeros(z_values), antialiased=False, linewidth=0)

        plot.show()
