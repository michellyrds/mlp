import numpy as np
from numpy import genfromtxt
from mlp import *
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import statsmodels.api as sm

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

features, target = mlp.preprocessing(caracteres_ruido)
testes = mlp.predict(features)
print(target)
print(testes)
'''
testes = round(mlp.predict(features))
m_c = confusion_matrix(target, testes)
print(m_c)

df_cm = pd.DataFrame(m_c, index = [i for i in "PN"],
                  columns = [i for i in "PN"])
plt.figure(figsize = (7,6))
sn.heatmap(df_cm, annot=True)
plt.show()
'''
