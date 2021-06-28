#carregando os datasets
import numpy as np
from numpy import genfromtxt
from mlp import *
import csv

def preprocessing(dataset):
    gen = np.genfromtxt(dataset, delimiter=",", encoding = "UTF-8-sig")
    features = gen[:, 0:2]
    targets = gen[:, -1]
    
    with open("datasets/resultadoData.csv", 'w', encoding='UTF-8-sig') as d:
        escrivao = csv.writer(d)
        escrivao.writerows(features)

    with open("datasets/resultadoLabels.csv", 'w', encoding='UTF-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(targets)
        
    pass

dataset = "datasets/problemAND.csv"
preprocessing(dataset)

AND_dataset = np.genfromtxt("datasets/problemAND.csv", delimiter=",", encoding = "UTF-8-sig")
Dados_dataset = np.genfromtxt("datasets/resultadoData.csv", delimiter=",", encoding = "UTF-8-sig")
Labels_dataset = np.genfromtxt("datasets/resultadoLabels.csv", delimiter=",", encoding = "UTF-8-sig")

# print("Input: ", AND_dataset)
# print("")
# print("Dados: ", Dados_dataset)
# print("")
# print("Labels: ", Labels_dataset)


# print("-------------------")


h = gen_hyperparameters_dict(2,[2],1)
mlp = MultilayerPerceptron(h)
mlp.train(AND_dataset)

#predições: chamar o forward_propagate
#print("Output: ", output)
