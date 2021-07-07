import numpy as np
from numpy import genfromtxt
from mlp import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn.metrics as mt

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

y1 = caracteres_limpo[:-14, :]
y2 = caracteres_ruido[7:-7, :]
y3 = caracteres_ruido20[14:, :]
dataset_teste = np.concatenate(
    (y1, y2, y3))

caracteres_limpo = caracteres_limpo[7:, :]
caracteres_ruido = caracteres_ruido[:, :]
caracteres_ruido20 = caracteres_ruido20[:-7, :]

dataset = np.concatenate(
    (caracteres_limpo, caracteres_ruido, caracteres_ruido20))

hyperparameters = gen_hyperparameters_dict(63, [49, 39, 33, 29], 7)
mlp = MultilayerPerceptron(hyperparameters)
mlp.train(dataset, maxEpochs=2000, learning_rate=0.001,
          test_size=0.2, random_state=None, momentum=0.90)

features, target = mlp.preprocessing(dataset_teste)
testes = mlp.predict(features)

print("Acuracia depois de testes:")
print(mt.accuracy_score(target, testes))
print("Precisao depois de testes:")
print(mt.precision_score(target, testes, average='micro'))
print("Recall depois de testes:")
print(mt.recall_score(target, testes, average='micro'))
print("F1_score depois de testes:")
print(mt.f1_score(target, testes, average='micro'))
print("Roc_Auc_score depois de testes:")
print(mt.roc_auc_score(target, testes, average='micro'))


m_c = mt.confusion_matrix(target.argmax(axis=1), testes.argmax(axis=1))
print(m_c)

df_cm = pd.DataFrame(m_c, index = [i for i in "ABCDEJK"],
                  columns = [i for i in "ABCDEJK"])
plt.figure(figsize = (7,6))
sn.heatmap(df_cm, annot=True)
plt.show()

