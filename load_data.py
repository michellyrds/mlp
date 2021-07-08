import numpy as np

def get_data():
    AND_dataset = np.genfromtxt(
        'datasets/problemAND.csv', delimiter=",", encoding='UTF-8-sig')

    OR_dataset = np.genfromtxt(
        'datasets/problemOR.csv', delimiter=",", encoding='UTF-8-sig')

    XOR_dataset = np.genfromtxt(
        'datasets/problemXOR.csv', delimiter=",", encoding='UTF-8-sig')

    return AND_dataset, OR_dataset, XOR_dataset


def get_data_caracteres():
    caracteres_limpo = np.genfromtxt(
        'datasets/caracteres-limpo.csv', delimiter=',', encoding='UTF-8-sig')

    caracteres_ruido = np.genfromtxt(
        'datasets/caracteres-ruido.csv', delimiter=',', encoding='UTF-8-sig')

    caracteres_ruido20 = np.genfromtxt(
        'datasets/caracteres_ruido20.csv', delimiter=',', encoding='UTF-8-sig')

    return caracteres_limpo, caracteres_ruido, caracteres_ruido20
