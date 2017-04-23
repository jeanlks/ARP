# -*- coding: utf-8 -*-
from classifier import NaiveBayesClassifier

nbc = NaiveBayesClassifier("iris-treinamento.txt",
                           ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])

vars_combinations = [
    ['Sepal Length', 'Sepal Width'],
    ['Sepal Length', 'Petal Width'],
    ['Sepal Length', 'Petal Length'],
    ['Petal Length', 'Petal Width'],
    ['Petal Length', 'Sepal Width'],
    ['Petal Width', 'Sepal Width']
]
for vars_combination in vars_combinations:
    nbc.plot_two_var_normal(vars_combination)

