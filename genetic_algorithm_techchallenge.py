

import random
import math
import copy 
from typing import List, Tuple
from capitais import *
import sys

def knn_hotstart(ponto_inicial: int, ordem_cidades: List[int]) -> List[int]:
    """
    Gera um caminho a partir de uma cidade inicial usando o knn (Nearest Neighbour)

    Parametros:
    - ponto_inicial (int): Cidade que inicia o percurso
    - ordem_cidades (List[int]): Uma lista de inteiros que representa a localização das cidades

    Returns:
    List[int]: Melhor percurso usando a técnica knn, a partir da cidade informada
    """    

    matriz = matriz_distancia
    ordem = ordem_cidades
    cidade_inicial = ponto_inicial

    indice_cidade = ordem.index(cidade_inicial)

    caminho_inicial = [cidade_inicial]

    cidades = ordem.copy()
    cidades.remove(cidade_inicial)

    cidade = indice_cidade
    lista = matriz[cidade].copy()

    lista.remove(0)

    while len(cidades) > 0:
        menor = min(lista)
        
        posicao_matriz = matriz[cidade].index(menor)
       
        posicao_caminho = ordem[posicao_matriz]
        
        if posicao_caminho in cidades:
            proxima_cidade = posicao_caminho
            caminho_inicial.append(proxima_cidade)
            indice_cidade = ordem.index(proxima_cidade)
            
            cidade = indice_cidade
            lista = matriz[cidade].copy()

            lista.remove(0)
            
            cidades.remove(posicao_caminho)
        else:
            lista.remove(menor)

    return caminho_inicial
        

def generate_random_population(ordem_cidades: List[int], population_size: int) -> List[int]:
    """
    Gera aleatoriamente um conjunto de percursos a partir de uma lista de cidades

    Parametros:
    - ordem_cidades (List[int]): Uma lista de inteiros que representa a localização das cidades
    - population_size (int): O tamanho da população, número de percursos a ser gerado

    Returns:
    List[List[int]]: Lista de percursos, onde cada percurso representa a ordem do trajeto

    """
    return [random.sample(ordem_cidades, len(ordem_cidades)) for _ in range(population_size)]


def distancia_cidades(cidade1: int, cidade2: int) -> float:
    """
    Consulta a distância entre duas cidades na Matriz de Distâncias

    Parametros:
    - cidade1 (Int): O indice da primeira cidade.
    - cidade2 (Int): O indice da segunda cidade.

    Returns:
    float: Distância entre as duas cidades.
    """
    return matriz_distancia[cidade1 - 1][cidade2 - 1]



def calculate_fitness(path: List[int]) -> float:
    """
    Calcula fitness de um caminho utilizando a Matriz de Distâncias.

    Parametros:
    - path (List[Int]): Lista de inteiros que representa um percurso, onde cada inteiro é o indice da cidade.

    Returns:
    float: A distancia total do trajeto.
    """
    
    distance = 0
    n = len(path)
       
    for i in range(n-1):
        distance += distancia_cidades(path[i], path[i+1])

    distance += distancia_cidades(path[n-1], path[0])
    
    return distance


def order_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """
    Executa o cruzamento de dois percursos para criar um novo percurso

    Parameters:
    - parent1 (List[int]): Primeiro percurso
    - parent2 (List[int]): Segundo percurso

    Returns:
    List[int]: Percurso resultante do cruzamento.
    """
    length = len(parent1)
    
    # Choose two random indices for the crossover
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)

    # Initialize the child with a copy of the substring from parent1
    child = parent1[start_index:end_index]

    # Fill in the remaining positions with genes from parent2
    remaining_positions = [i for i in range(length) if i < start_index or i >= end_index]
    remaining_genes = [gene for gene in parent2 if gene not in child]

    for position, gene in zip(remaining_positions, remaining_genes):
        child.insert(position, gene)

    return child


# child = order_crossover(parent1, parent2)
# print("Parent 1:", [0, 1, 2, 3, 4, 5, 6, 7, 8])
# print("Parent 1:", parent1)
# print("Parent 2:", parent2)
# print("Child   :", child)


# # Example usage:
# population = generate_random_population(5, 10)

# print(calculate_fitness(population[0]))


# population = [(random.randint(0, 100), random.randint(0, 100))
#           for _ in range(3)]



# TODO: implement a mutation_intensity and invert pieces of code instead of just swamping two. 
def mutate(solution:  List[int], mutation_probability: float) ->  List[int]:
    """
    Modifica uma solução invertendo um segmento da sequência com uma determinada probabilidade de mutação

    Parametros:
    - solution (List[int]): Um percurso a ser modificado
    - mutation_probability (float): A probabilidade de mutação de cada percurso

    Returns:
    List[int]: O percurso modificado.
    """
    mutated_solution = copy.deepcopy(solution)

    # Check if mutation should occur    
    if random.random() < mutation_probability:
        
        # Ensure there are at least two cities to perform a swap
        if len(solution) < 2:
            return solution
    
        # Select a random index (excluding the last index) for swapping
        index = random.randint(0, len(solution) - 2)
        
        # Swap the cities at the selected index and the next index
        mutated_solution[index], mutated_solution[index + 1] = solution[index + 1], solution[index]   
        
    return mutated_solution

### Demonstration: mutation test code    
# # Example usage:
# original_solution = [(1, 1), (2, 2), (3, 3), (4, 4)]
# mutation_probability = 1

# mutated_solution = mutate(original_solution, mutation_probability)
# print("Original Solution:", original_solution)
# print("Mutated Solution:", mutated_solution)


def sort_population(population: List[List[int]], fitness: List[float]) -> Tuple[List[List[int]], List[float]]:
    """
    Clasifica a população baseado nos valores de fitness

    Parametros:
    - population (List[List[int]]): A população de percursos, onde cada percurso é uma solução encontrada
    - fitness (List[float]): A distância correspondente a cada percurso
    
    Returns:
    Tuple[List[List[int]], List[float]]: Contem o percurso e a distância correspondente
    """
    # Combine lists into pairs
    combined_lists = list(zip(population, fitness))

    # Sort based on the values of the fitness list
    sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1])

    # Separate the sorted pairs back into individual lists
    sorted_population, sorted_fitness = zip(*sorted_combined_lists)

    return sorted_population, sorted_fitness


if __name__ == '__main__':
    N_CITIES = 10
    
    POPULATION_SIZE = 100
    N_GENERATIONS = 100
    MUTATION_PROBABILITY = 0.3
    cities_locations = [(random.randint(0, 100), random.randint(0, 100))
              for _ in range(N_CITIES)]
    
    # CREATE INITIAL POPULATION
    population = generate_random_population(cities_locations, POPULATION_SIZE)

    # Lists to store best fitness and generation for plotting
    best_fitness_values = []
    best_solutions = []
    
    for generation in range(N_GENERATIONS):
  
        
        population_fitness = [calculate_fitness(individual) for individual in population]    
        
        population, population_fitness = sort_population(population,  population_fitness)
        
        best_fitness = calculate_fitness(population[0])
        best_solution = population[0]
           
        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)    

        print(f"Generation {generation}: Best fitness = {best_fitness}")

        new_population = [population[0]]  # Keep the best individual: ELITISM
        
        while len(new_population) < POPULATION_SIZE:
            
            # SELECTION
            parent1, parent2 = random.choices(population[:10], k=2)  # Select parents from the top 10 individuals
            
            # CROSSOVER
            child1 = order_crossover(parent1, parent2)
            
            ## MUTATION
            child1 = mutate(child1, MUTATION_PROBABILITY)
            
            new_population.append(child1)
            
    
        print('generation: ', generation)
        population = new_population
    


