# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:06:56 2024

@author: christiane
"""

import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm import mutate, order_crossover, generate_random_population, calculate_fitness, sort_population, default_problems
from draw_functions import draw_paths, draw_plot, draw_cities
import sys
import numpy as np
import pygame
from benchmark_att48 import *
from capitais import *
import sys

import math
import copy 
from typing import Tuple


# print(localizacao[4])
# print(localizacao)
# print(localizacao.index((15,23)))
#print(ordem_cidades)
#print(matriz_distancia)


# def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
#     """
#     Calculate the Euclidean distance between two points.

#     Parameters:
#     - point1 (Tuple[float, float]): The coordinates of the first point.
#     - point2 (Tuple[float, float]): The coordinates of the second point.

#     Returns:
#     float: The Euclidean distance between the two points.
#     """
#     return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def distancia_cidades(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Consulta a distância entre duas cidades na Matriz de Distâncias

    Parameters:
    - point1 (Tuple[float, float]): The coordinates of the first point.
    - point2 (Tuple[float, float]): The coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    return matriz_distancia[localizacao.index((point1[0],point1[1]))][localizacao.index((point2[0],point2[1]))]

cidade1: Tuple[float, float]
cidade2: Tuple[float, float]

cidade1 = (92,41)
cidade2 = (25,35)

ordem1 = localizacao.index((cidade1[0],cidade1[1]))
ordem2 = localizacao.index((cidade2[0],cidade2[1]))

print(ordem1)
print(ordem2)

print(matriz_distancia[localizacao.index((cidade1[0],cidade1[1]))][localizacao.index((cidade2[0],cidade2[1]))])