from math import exp

def funcao_step(x): # funcao sigmoide 
    return 1.0/(1.0+exp(-x))

def d_funcao_step(x): #derivada da funcao step
    return exp(-x)/(1.0+exp(-x))**2