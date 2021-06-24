#execucao do MLP e avaliar o modelo
from mlp import(
    MultilayerPerceptron
)

hyperparameters = {
    "n_espaco_vetorial": 2,
    "threshold": 0.5


}



# função para separar o label

'''
def separandoData(dataset, n_labels):
    i = 0
    while(i<n_labels):
        last_column = dataset.iloc[: , -1]
        last_columns.append(last_column)
        dataset.drop(last_column)
'''



# possível avaliação que resultará em matrizes de confusão

'''
testes = round(modelo.predict(teste))

m_c = confusion_matrix(labels, testes)
print(m_c)

df_cm = pd.DataFrame(m_c, index = [i for i in "PN"],
                  columns = [i for i in "PN"])
plt.figure(figsize = (7,6))
sn.heatmap(df_cm, annot=True)
plt.show()
'''