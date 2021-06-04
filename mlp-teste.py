#carregando os datasets
import numpy as np
from numpy import genfromtxt
from mlp import MultilayerPerceptron

AND_dataset = np.genfromtxt("datasets/problemAND.csv", delimiter=",", encoding = "UTF-8-sig")
#print(AND_dataset)
OR_dataset = np.genfromtxt("datasets/problemOR.csv", delimiter=",", encoding = "UTF-8-sig")

XOR_dataset = np.genfromtxt("datasets/problemXOR.csv", delimiter=",", encoding = "UTF-8-sig")

teste = MultilayerPerceptron(2, 2)
teste.createPerceptrons(1)

# separar dataset em conjunto de treinamento e label

i = 0
while i < len(AND_dataset):
    teste.epoca(AND_dataset[i])
    # teste.backpropagation()