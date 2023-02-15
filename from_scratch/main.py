import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn.metrics as mt
from load_data import get_data_caracteres

from mlp import MultilayerPerceptron

caracteres_limpo, caracteres_ruido, caracteres_ruido20 = get_data_caracteres()

y1 = caracteres_limpo[:-14, :]
y2 = caracteres_ruido[7:-7, :]
y3 = caracteres_ruido20[14:, :]
dataset_teste = np.concatenate((y1, y2, y3))

caracteres_limpo = caracteres_limpo[7:, :]
caracteres_ruido = caracteres_ruido[:, :]
caracteres_ruido20 = caracteres_ruido20[:-7, :]

dataset = np.concatenate((caracteres_limpo, caracteres_ruido, caracteres_ruido20))

mlp = MultilayerPerceptron(
    hyperparameters={
        "n_inputs": 63,
        "n_hidden_layers": [49, 39, 33, 29],
        "n_outputs": 7,
    },
    seed=12345,
)
mlp.train(
    dataset,
    maxEpochs=50,
    learning_rate=0.001,
    test_size=0.33,
    random_state=12345,
    accMin=0.90,
)

features, target = mlp.split_dataset(dataset_teste)
testes = mlp.predict(features)

"""
    Análise dos resultados
"""

print("\n---------------- Validação ----------------")
print("Acurácia: {}".format((mt.accuracy_score(target, testes))))
print("Precisão: {}".format(mt.precision_score(target, testes, average="micro")))
print("Recall: {}".format(mt.recall_score(target, testes, average="micro")))
print("F1_score: {}".format(mt.f1_score(target, testes, average="micro")))
print("Roc_Auc_score: {}".format(mt.roc_auc_score(target, testes, average="micro")))

m_c = mt.confusion_matrix(target.argmax(axis=1), testes.argmax(axis=1))
print(m_c)

df_cm = pd.DataFrame(m_c, index=[i for i in "ABCDEJK"], columns=[i for i in "ABCDEJK"])
plt.figure(figsize=(7, 6))
sn.heatmap(df_cm, annot=True)
plt.show()
