#carregando os datasets
import numpy as np
from numpy import genfromtxt
from mlp import *


#AND_dataset = np.g
AND_dataset = np.genfromtxt("datasets/problemAND.csv", delimiter=",", encoding = "UTF-8-sig")

# print("-------------------")
#OR_dataset = np.genfromtxt("datasets/problemOR.csv", delimiter=",", encoding = "UTF-8-sig")

#XOR_dataset = np.genfromtxt("datasets/problemXOR.csv", delimiter=",", encoding = "UTF-8-sig")

h = gen_hyperparameters_dict(2,[2],1)
mlp = MultilayerPerceptron(h)
output = mlp.forward_propagate(AND_dataset[0])

# print("Input: ", AND_dataset)
print("Output: ", output)

