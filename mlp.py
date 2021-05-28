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

class Perceptron: #declarando a classe do nosso neurônio multilayer perceptron

    def __init__(self, pesos, bias):
        self.pesos = pesos
        self.bias = bias
        pass

    def fit(self, X): # etapa de treino
        i = 0
        for valor in x:
            self.somatoria += self.pesos[i] * valor
            i += 1
        self.somatoria += self.bias
        self.predict(self, self.somatoria)
        pass

    def predict(self, X): # testes
        # função de ativação bipolar
        if (x >= 0):
            return 1
        else:
            return -1
        pass
        
class MultilayerPerceptron: #declarando a classe do nosso  multilayer perceptron

    def __init__(self, numPerIni, numPerFim):
        self.numPerIni = numPerIni
        self.numPerFim = numPerFim
        pass

    def createPerceptrons(self):
        i = 0
        while i < self.numPerIni:
            perceptron = Perceptron(self.pesosAleatorios(self.numPerIni, randon.randrange(-1, 1)))
            # continues...
        pass
    
    def pesosAleatorios(numPerIni):
        for i in numPerIni:
           numAleatorio[i] = random.randrange(0, 1)

    
>>>>>>> a893382 (28/05)
