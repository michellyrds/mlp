from math import exp

import numpy as np


def sigmoid(input):
    ativacao = []
    for x in input:
        ativacao.append(1.0 / (1.0 + exp((-x))))  # f(x)
    return np.asarray(ativacao)


def d_sigmoid(input):
    ativacao = []

    for x in input:
        ativacao.append(exp(-x) / ((1.0 + exp(-x)) ** 2))  # f'(x)
    return np.asarray(ativacao)


def tanh(input):
    ativacao = []
    for x in input:
        ativacao.append((exp(x) - exp(-x)) / (exp(x) + exp(-x)))  # f(x)
    return np.asarray(ativacao)


def d_tanh(input):
    ativacao = []

    for x in input:
        ativacao.append(1 - ((exp(x) - exp(-x)) / (exp(x) + exp(-x)) ** 2))  # f'(x)
    return np.asarray(ativacao)


def bipolar(theta, input):
    ativacao = []

    for x in input:
        ativacao.append(1 if x >= theta else 0)
    return np.asarray(ativacao)
