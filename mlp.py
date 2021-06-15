import numpy as np #importando biblioteca de manipulação de matrizes e etc
from funcao_step import (
    funcao_step,
    d_funcao_step
)
import csv

class Perceptron: #declarando a classe do nosso neurônio multilayer perceptron
    pass

class MultilayerPerceptron:
    #arrumar parametros da camada escondida
    def __init__(self, n_input = 2, n_camada_escondida = [4,2], n_output = 1): #numeros de perceptrons na camada de entrada, escondida e saída
        
        self.n_input = n_input
        self.n_camada_escondida = n_camada_escondida
        self.n_output = n_output

        camadas = [self.n_input] + self.n_camada_escondida + [self.n_output]

        # inicializando os pesos aleatorios
        #self.w = np.random.uniform([-1, 1, self.n_input + 1])
        w = []

        for i in range(len(camadas) - 1):
            #wL = np.random.uniform([-1, 1], camadas[i], camadas[i + 1])
            #se der errado, usar: 
            wL = np.random.rand(camadas[i], camadas[i-1])
            w.append(wL)
        self.w = w
        print("Rede neural:")
        print(self.w)

    def forward_propagate(self, input):
        ativacao = input

        for wL in self.w:
            #arrumar essa desgraça aqui
            net_input = np.dot(ativacao, wL) #multiplicacao de matrizes
            print(net_input)
            ativacao = funcao_step(net_input)

        return ativacao

