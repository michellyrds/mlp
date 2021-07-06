import numpy as np
from funcao_step import *
import random
from sklearn.model_selection import (
    train_test_split, 
    KFold
)
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


class MultilayerPerceptron(object):

    def __init__(self, hyperparameters: dict, seed=None):

        self.n_inputs = hyperparameters['n_inputs']
        self.n_camada_escondidas = hyperparameters['n_camada_escondida']
        self.n_outputs = hyperparameters['n_outputs']

        # representação da arquitetura da rede
        camadas = [self.n_inputs] + self.n_camada_escondidas + [self.n_outputs]

        np.random.seed(seed)
        pesos = []  # matriz de pesos
        for i in range(len(camadas)-1):
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
        X, y = dataset[:, :-self.n_outputs], dataset[:, self.n_inputs:]

        return X, y

    def forward_propagate(self, inputs):
        ativacoes = inputs

        self.ativacoes[0] = ativacoes

        for i, w in enumerate(self.pesos[:-1]):
            net_inputs = np.dot(ativacoes, w)
            ativacoes = sigmoid(net_inputs)
            self.ativacoes[i+1] = ativacoes

        # aplicando a função de ativação bipolar na camada de saída
        w = self.pesos[-1]
        net_inputs = np.dot(ativacoes, w)
        ativacoes = bipolar(0,net_inputs)
        self.ativacoes[-1] = ativacoes
        # # !
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

            delta_t = delta.reshape(delta.shape[0], -1).T

            ativacao_atual = self.ativacoes[i]

            # transforma em uma matriz coluna
            ativacao_atual = ativacao_atual.reshape(
                ativacao_atual.shape[0], -1)

            self.derivadas[i] = np.dot(ativacao_atual, delta_t)

            erro = np.dot(delta, self.pesos[i].T)  # transposta

    def mean_squad_error(self, label, output):
        return np.average((label - output)**2)

    def gradient_descent(self, learning_rate: float):

        for i in range(len(self.pesos)):
            w = self.pesos[i]
            derivadas = self.derivadas[i]
            w += (derivadas * learning_rate)
            self.pesos[i] = w

    def train_CV(self, dataset, learning_rate, test_size, seed=None):
        pass

    def train(self, dataset, maxEpochs, learning_rate, test_size, random_state=None, momentum=0.7):
        # dataset de treinamento, um de teste e um de validação
        X, y = self.preprocessing(dataset)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        sum_error = 0
        for i in range(maxEpochs):
            print("\n---------------- Época {} ----------------".format(i+1))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            sum_error_train = 0

            for j, input in enumerate(X_train):

                output = self.forward_propagate(input)

                error = y_train[j] - output

                self.back_propagate(error)

                self.gradient_descent(learning_rate)

                sum_error_train += self.mean_squad_error(y_train[j], output)

            error_rate_train = sum_error_train/(len(X_train))
            print("Erro médio no treinamento: {}".format(error_rate_train))

            # calculate the momentum
            sum_error_test = 0
            for j, input in enumerate(X_test):
                
                output = self.forward_propagate(input)
                error = y_test[j] - output

                sum_error_test += self.mean_squad_error(y_test[j], output)

                
            error_rate_test = sum_error_test/(len(X_test))
            print("Erro médio na validação: {}".format(error_rate_test))

            sum_error += error_rate_test
            #calculando a acúracia total estimada usando random sampling (média das acurácias obtidas em cada iteração)
            acc = 1 - (sum_error/(i+1)) 
            if(acc >= momentum):
                print("Rede neural convergiu na época {} com acurácia de {}".format(i+1, acc))
                return
            
        
        print("\nTreinamento finalizado. Acurácia do modelo: {}".format(1 - (sum_error/maxEpochs)))

    def predict(self, input):
        output = []
        
        for j, teste in enumerate(input):
                output.append(self.forward_propagate(teste))

        return np.asarray(output)

    def save_model(self):  # michelly
        # salvar a arquitetura do modelo
        # salvar a matriz de pesos
        pass
