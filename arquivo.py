# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:38:23 2024

@author: christiane
"""

# arquivo = open("att48_d.txt", "r")
# conteudo = arquivo.read()
# print(conteudo[0])
# arquivo.close()

arquivo = open("att48_d.txt", "r")
#arqMatriz = open("matriz.txt","w")

matriz = []
linha = arquivo.readline()
print('inicio')
while linha != "":
    #print(linha)
    linha = linha.replace('      ',',')
    linha = linha.replace('\n',',')
    linha2  = '['+linha+']'
    matriz.append(linha2)
    #print(linha2)
    linha = arquivo.readline()
    
arquivo.close()

print(matriz)

print('fim')