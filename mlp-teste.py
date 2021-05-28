#carregando os datasets
import numpy as np
from numpy import genfromtxt

AND_dataset = np.genfromtxt("datasets/problemAND.csv", delimiter=",", encoding = "UTF-8-sig")
     
OR_dataset = np.genfromtxt("datasets/problemOR.csv", delimiter=",", encoding = "UTF-8-sig")

XOR_dataset = np.genfromtxt("datasets/problemXOR.csv", delimiter=",", encoding = "UTF-8-sig")
