<<<<<<< HEAD
"""
    Implementação do multilayer perceptron
"""

import numpy as np #importando biblioteca de manipulação de matrizes e etc


class MultilayerPerceptron: #declarando a classe do nosso multilayer perceptron

    def __init__(self, dataset):
        pass

    def fit(self, X, y): # etapa de treino
        pass

    def predict(self, X): #treinamento de fato
        pass
        
=======
import numpy as np #importando biblioteca de manipulação de matrizes e etc
import random
import funcao_step as fs

class Perceptron: #declarando a classe do nosso neurônio multilayer perceptron

    def __init__(self, pesos: np.vectorize, bias):
        self.pesos = pesos
        self.bias = bias
        self.somatoria
        pass

    def fit(self, X): # etapa de treino
        i = 0
        for valor in X:
            self.somatoria += self.pesos[i] * valor
            i += 1
        self.somatoria += self.bias[0]
        self.somatoria = fs.funcao_step(self.somatoria)
        #A CONFIRMAR ↓
        #self.somatoria = fs.d_funcao_step(self.somatoria)
        pass
       
class MultilayerPerceptron: #declarando a classe do nosso  multilayer perceptron

    def __init__(self, numPerIni, numPerFim):
        self.numPerIni = numPerIni
        self.numPerFim = numPerFim
        self.matrizP = []
        pass

    def createPerceptrons(self, qtd_CmE):
        camada = []
        i = 0
        while i < qtd_CmE:
            j = 0
            while j < self.numPerIni:
                perceptron = Perceptron(self.pesosAleatorios(self.numPerIni), self.pesosAleatorios(1))
                camada.append(perceptron)
                # continues...
                j += 1
            self.matrizP.append(camada)
            i += 1
    
    def pesosAleatorios(self, numPerIni): #atribui os pesos aleatórios iniciais
        numAleatorio = []
        i = 0
        while i < numPerIni:
           numAleatorio[i] = random.randrange(-1, 1)
           i += 1
        return numAleatorio

<<<<<<< HEAD
    
>>>>>>> a893382 (28/05)
=======
    def epoca(self, instancia):  
        i = 0
        while i < len(self.matrizP):
            j = 0
            while j < len(self.matrizP):
                self.matrizP[i][j].fit(instancia)
                j += 1
            i += 1
            
>>>>>>> 5d4600e (alteracoes)
