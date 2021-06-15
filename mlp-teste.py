#carregando os datasets
import numpy as np
from numpy import genfromtxt
from mlp import MultilayerPerceptron

AND_dataset = np.genfromtxt("datasets/problemAND.csv", delimiter=",", encoding = "UTF-8-sig")
#print(AND_dataset)
#OR_dataset = np.genfromtxt("datasets/problemOR.csv", delimiter=",", encoding = "UTF-8-sig")

#XOR_dataset = np.genfromtxt("datasets/problemXOR.csv", delimiter=",", encoding = "UTF-8-sig")


mlp = MultilayerPerceptron()

output = mlp.forward_propagate(AND_dataset)

print("Input: ", AND_dataset)
print("Output: ", output)

