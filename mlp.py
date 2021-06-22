import numpy as np #importando biblioteca de manipulação de matrizes e etc
from funcao_step import (
    funcao_step,
    d_funcao_step
)
import csv

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
    #arrumar parametros da camada escondida
    
    def __init__(self, hyperparameters: dict): #numeros de perceptrons na camada de entrada, escondida e saída
        
        self.n_inputs = hyperparameters['n_inputs']
        self.n_camada_escondidas = hyperparameters['n_camada_escondida']
        self.n_outputs = hyperparameters['n_outputs']

        camadas = [self.n_inputs] + self.n_camada_escondidas + [self.n_outputs]
        print("------------------------------")
        print("Esboço da rede neural:")
        print(camadas)
        print("------------------------------")
        # inicializando os pesos aleatorios
        # self.w = np.random.uniform([-1, 1, self.n_inputs + 1])
        pesos = [] #matriz de pesos

        for i in range(len(camadas)-1):
            #wL = np.random.uniform([-1, 1], camadas[i], camadas[i + 1])
            #se der errado, usar: 
            w = np.random.rand(camadas[i], camadas[i-1])
            pesos.append(w)
        self.pesos = pesos


    def train(dataset, train_size=0.7, test_size=0.3, random_state=0):
        pass #uke 
    # E AI APAGA AI APAGA IA ENTAO
    # SUA PROPRIEDADE:?? KKK AQUI NotADirectoryErrorAQUI É USP
    # SÓ TEM COMUNISTA
    
    # Propriedade sagrada
    # my property 
    #se gostou deixe seu like compartilhe o video com seus amigos ative o sininho para receber notificações se inscreva no canal e é isso meninas
    #oiiiiiiiiiiiiiiii
    #arrasta pra cima
    #Sara da 10
    # AAHHHHHHI
    



    def preprocessing(dataset): #Nanda, Ale, Raul 
        pass

    def forward_propagate(self, inputs): #1x2
        ativacoes = inputs #1x2
        print("ativacoes.shape:")
        print(ativacoes.shape)
        print("ativacoes:")
        print(ativacoes)
        for wL in self.w:
            #----------------------arrumar essa desgraça aqui--------------------------
            net_inputs = ativacoes.multiply(wL)
            #net_inputs = np.dot(ativacoes, wL) #multiplicacao de matrizes
            ativacoes = funcao_step(net_inputs)

        #aqui retorna o output da camada de saida
        return ativacoes

