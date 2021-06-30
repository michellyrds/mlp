from math import exp

def sigmoid(input): # funcao sigmoide input = X = (x1, x2, x3, ...)
    ativacao = []
    for x in input: 
        ativacao.append(
            1.0 / (1.0 + exp((-x))) #f(x)
        )

    return ativacao

def d_sigmoid(input): #derivada da funcao sigmoide
    ativacao = []
    
    for x in input:
        ativacao.append(
            exp(-x) / ((1.0 + exp(-x))**2) #f'(x)
        )
    return ativacao

def tanh(input):
    ativacao = []
    for x in input:
        ativacao.append(
            (exp(x) - exp(-x))/(exp(x) + exp(-x)) #f(x)
        )

def d_tanh(input):
    ativacao = []

    for x in input:
        ativacao.append(
            1 - ((exp(x) - exp(-x))/(exp(x) + exp(-x))**2) #f'(x)
        )

    return ativacao