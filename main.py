import numpy as np
from numpy import genfromtxt
from mlp import *

AND_dataset = np.genfromtxt(
    'datasets/problemAND.csv', delimiter=",", encoding='UTF-8-sig')

OR_dataset = np.genfromtxt(
    'datasets/problemOR.csv', delimiter=",", encoding='UTF-8-sig')

XOR_dataset = np.genfromtxt(
    'datasets/problemXOR.csv', delimiter=",", encoding='UTF-8-sig')

caracteres_limpo = np.genfromtxt(
    'datasets/caracteres-limpo.csv', delimiter=',', encoding='UTF-8-sig')

caracteres_ruido = np.genfromtxt(
    "datasets/caracteres-ruido.csv", delimiter=',', encoding='UTF-8-sig')

caracteres_ruido20 = np.genfromtxt(
    'datasets/caracteres_ruido20.csv', delimiter=',', encoding='UTF-8-sig')

dataset = np.concatenate(
    (caracteres_limpo, caracteres_ruido, caracteres_ruido20))

hyperparameters = gen_hyperparameters_dict(63, [49, 39, 33, 29], 7)
mlp = MultilayerPerceptron(hyperparameters)
mlp.train(dataset, maxEpochs=1000, learning_rate=0.001,
          test_size=0.2, random_state=None, momentum=0.90)
