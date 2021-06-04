import numpy as np #importando biblioteca de manipulação de matrizes e etc
import random
class Teste: 
    #eu esqueci que função vamos usar
    def funcao_Ativacao(x):
        return
    
    def derivada_funcao_Ativacao(x):
        return

    def pesosAleatorios(numPerIni):
        numAleatorio = []
        for i in numPerIni:
            numAleatorio[i] = random.randrange(0, 1)
        return numAleatorio


    def arquitetura(tamanho_entrada,
                    qntd_neuronios_camada_escondida,
                    qntd_neuronios_camada_saida,
                    funcao_Ativacao,
                    derivada_funcao_Ativacao):
        tamanho_entrada = tamanho_entrada
        qntd_neuronios_camada_escondida
        #... bla bla bla
        
        # x neuronios na camada escondida
        #       peso do neuronio x associado ao neuronio y da camada de entrada
        # 0     w_11    w_12    theta_1
        # x-1   w_21    w_22    theta_2

        #matriz_Camada_Escondida
        #quantidade de numeros aleatorios a serem gerados = (qntd_neuronios_camada_escondida * tamanho_entrada+1), 
        # n_linhas = qntd_neuronios_camada_escondida, 
        # n_colunas = tamanho_entrada+1

        pesos1 = pesosAleatorios(qntd_neuronios_camada_escondida * tamanho_entrada+1)
        matriz_Camada_Escondida = []
        #linhas
        for num in range(qntd_neuronios_camada_escondida):
            linha2 = []
            #colunas
            for num2 in range(tamanho_entrada + 1):
                linha2.append(pesos1[num2])
            matriz_Camada_Escondida.append(linha2)
                
        #matriz_Camada_Escondida
        #quantidade de numeros aleatorios a serem gerados = (qntd_neuronios_camada_saida * qntd_neuronios_camada_escondida + 1), 
        # n_linhas = qntd_neuronios_camada_saida, 
        # n_colunas = tamanho da camada escondida +1
        pesos2 = pesosAleatorios(qntd_neuronios_camada_saida * qntd_neuronios_camada_escondida + 1)
        matriz_camada_saida = []
        for num in range(qntd_neuronios_camada_saida):
            linha2 = []
            for num2 in range(qntd_neuronios_camada_escondida + 1):
                linha2.append(pesos2[num2])
            matriz_camada_saida.append(linha2)