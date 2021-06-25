from math import exp

def funcao_step(x): # funcao sigmoide
    ativacao = []
    for element in x: 
        ativacao.append(1.0 / (1.0 + exp((-element))))

    return ativacao

def d_funcao_step(x): #derivada da funcao step
    return exp(-x) / ((1.0 + exp(-x))**2)