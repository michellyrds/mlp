import numpy as np
# importando biblioteca de manipulação de matrizes e etc
from funcao_step import (
    d_sigmoid,
    sigmoid,
    tanh,
    d_tanh
)
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
"""
    Dict que implementa os hiperparametros do MLP
"""

"""
    **Hiperparametros do modelo**
    n_inputs: numero de perceptrons na camada de entrada
    n_camada_escondida: lista de int no qual
        len(n_camada_escondida): número de camadas escondidas
        n_camada_escondida[i] = k, onde k é o número de perceptrons na camada escondida i
    n_outputs: numero de perceptrons na camada de saída
"""


def gen_hyperparameters_dict(n_inputs: int, n_camada_escondida: list, n_outputs: int) -> dict:

    hyperparameters = dict([
        ('n_inputs', n_inputs),
        ('n_camada_escondida', n_camada_escondida),
        ('n_outputs', n_outputs)
    ])
    return hyperparameters


class MultilayerPerceptron:

    def __init__(self, hyperparameters: dict, seed=None):

        self.n_inputs = hyperparameters['n_inputs']
        self.n_camada_escondidas = hyperparameters['n_camada_escondida']
        self.n_outputs = hyperparameters['n_outputs']
        self.inputs = []
        self.labels = []

        # representação da arquitetura da rede
        camadas = [self.n_inputs] + self.n_camada_escondidas + [self.n_outputs]

        np.random.seed(seed)
        pesos = []  # matriz de pesos
        for i in range(len(camadas)-1):
            # matriz aleatoria [-1,1)
            w = np.random.uniform(-1, 1, (camadas[i], camadas[i+1]))
            # w = np.random.rand(camadas[i], camadas[i + 1])
            pesos.append(w)
        self.pesos = pesos

        # ativações por camada
        ativacoes = []
        for i in range(len(camadas)):
            a = np.zeros(camadas[i])
            ativacoes.append(a)

        self.ativacoes = ativacoes

        # derivadas por camada
        derivadas = []
        for i in range(len(camadas)-1):
            d = np.zeros((camadas[i], camadas[i+1]))
            derivadas.append(d)

        self.derivadas = derivadas

    def preprocessing(self, dataset):
        self.inputs.clear()
        self.labels.clear()

        for input in dataset:
            self.inputs.append(input[:-self.n_outputs])
            self.labels.append(input[self.n_inputs:])

    def forward_propagate(self, inputs):
        ativacoes = inputs

        self.ativacoes[0] = ativacoes

        for i, w in enumerate(self.pesos[:-1]):
            net_inputs = np.dot(ativacoes, w)
            ativacoes = sigmoid(net_inputs)
            self.ativacoes[i+1] = ativacoes

        # nao devemos aplicar a funcao de ativação na camada de saida
        w = self.pesos[-1]
        net_inputs = np.dot(ativacoes, w)
        ativacoes = net_inputs
        # !
        return ativacoes

    def back_propagate(self, erro):
        """
        E = erro quadrático

        y - a[i+1] = erro (label - output da predição)
        s'(h_[i+1]) = derivada da função step
        a_i = ativacoes da camada i

        dE/dW_i = (y - a[i+1]) * s'(h_[i+1])*a_i
        s'(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1]))
        s(h_[i+1]) = a_[i+1]

        dE/dW_[i-1] = (y - a_[i+1] * s'(h_[i+1])) * W_i * s'(h_i) * a_[i-1]

        """

        for i in reversed(range(len(self.derivadas))):
            ativacoes = self.ativacoes[i+1]

            delta = erro * d_sigmoid(ativacoes)
            delta_t = np.reshape(np.shape(delta[0]), -1).T
            # np.reshape(np.shape(delta[0]), -1).T # transposta da matriz coluna

            ativacao_atual = self.ativacoes[i]
            # transforma em uma matriz coluna
            ativacao_atual = np.reshape(np.shape(ativacao_atual[0]), -1)
            # se nao funfar, tentar:
            # ativacao_atual_coluna = ativacao_atual.reshape(ativacao_atual.shape[0], -1)
            self.derivadas[i] = np.dot(ativacao_atual, delta_t)
            # !
            erro = np.dot(delta, np.transpose(self.pesos[i]))  # transposta

    def mean_squad_error(self, label, output):
        return np.average((label - output)**2)

    def gradient_descent(self, learning_rate: float):

        for i in range(len(self.pesos)):
            w = self.pesos[i]
            derivadas = self.derivadas[i]
            w += derivadas * learning_rate

    def train(self, dataset, learning_rate, test_size, seed=None):
        # train, validation = train
        random.Random(seed).shuffle(dataset)
        #self.preprocessing(dataset)

        #X_train, X_test, y_train, y_test = train_test_split(self.inputs, self.labels, test_size=test_size)

        kfolds = KFold(n_splits=9, shuffle=True, random_state=seed)
        kfolds.get_n_splits(dataset)

        for train_index, test_index in kfolds.split(dataset):
            X_train, X_test = dataset[train_index], dataset[test_index]
            y_train, y_test = dataset[train_index],dataset[test_index]


        for i in range(True):
            erro_quadrado = 0
            erro_quadrado_ant = 0

            for j, input in enumerate(X_train):

                output = self.forward_propagate(input)

                erro = y_train[j] - output

                self.back_propagate(erro)

                self.gradient_descent(learning_rate)

                erro_quadrado_ant = erro_quadrado
                erro_quadrado += self.mean_squad_error(y_train[j], output)

            print("Erro: {} na época {}".format(
                erro_quadrado/(len(X_train)), i+1))

    def predict(self, input):

        output = self.forward_propagate(input)

        return output

    def save_model(self):  # michelly
        # salvar a arquitetura do modelo
        # salvar a matriz de pesos
        pass
