#execucao do MLP e avaliar o modelo
from mlp import(
    Perceptron,
    MultilayerPerceptron
)

hyperparameters = {
    "n_espaco_vetorial": 2,
    "threshold": 0.5


}




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