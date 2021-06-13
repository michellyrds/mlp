import numpy as np #importando biblioteca de manipulação de matrizes e etc
from funcao_step import (
    funcao_step,
    d_funcao_step
)
import csv

class Perceptron: #declarando a classe do nosso neurônio multilayer perceptron
    pass

class MultilayerPerceptron:
    
    def __init__(self, input_size, eta=0.01, threshold=1e-3):
        self.w = np.random.uniform([-1,1,input_size+1])
        self.fnet = np.vectorize(funcao_step)
        self.dfnet = np.vectorize(d_funcao_step)
        self.eta = eta
        self.threshold = threshold
        self.erro_quad = 0

    #def init_weights(self)

    def train(self, dataset):
        n = dataset.shape[0]

#https://www.codeproject.com/Articles/821348/Multilayer-Perceptron-in-Python

