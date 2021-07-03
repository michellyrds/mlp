# carregando os datasets
import numpy as np
from numpy import genfromtxt
from mlp import *
import csv

AND_dataset = np.genfromtxt(
    "datasets/problemAND.csv", delimiter=",", encoding="UTF-8-sig")
caracteres_limpo = np.genfromtxt(
    "datasets/caracteres-limpo.csv", delimiter=',', encoding="UTF-8-sig")

# print(np.shape(caracteres_limpo))
hyperparameters = gen_hyperparameters_dict(63, [1,1], 7)
mlp = MultilayerPerceptron(hyperparameters)
mlp.train(caracteres_limpo, 1, 0.1)

