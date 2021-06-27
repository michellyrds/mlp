from math import exp

def sigmoid(input): # funcao sigmoide
    ativacao = []
    for x in input: 
        ativacao.append(
            1.0 / (1.0 + exp((-x))) #f(x)
        )

    return ativacao

def d_sigmoid(x): #derivada da funcao sigmoide
    
    return exp(-x) / ((1.0 + exp(-x))**2)

def tanh(input):
    ativacao = []
    for x in input:
        ativacao.append(
            (exp(x) - exp(-x))/(exp(x) + exp(-x)) #f(x)
        )

def d_tanh(x):
    return 1 - ((exp(x) - exp(-x))/(exp(x) + exp(-x))**2)