# carregando os datasets
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

dataset = np.concatenate((caracteres_limpo,caracteres_ruido,caracteres_ruido20))

# for i,input in enumerate(dataset):
#     print("{}: {}".format(i,input))

# print(np.shape(caracteres_limpo))
hyperparameters = gen_hyperparameters_dict(63, [1, 1], 7)
mlp = MultilayerPerceptron(hyperparameters)
mlp.train(dataset, 1000, 0.1, 0.2, 5, 0.8)
