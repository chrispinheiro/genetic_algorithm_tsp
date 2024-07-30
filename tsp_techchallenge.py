import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm_techchallenge import mutate, order_crossover, generate_random_population, calculate_fitness, sort_population, knn_hotstart
from draw_functions import draw_paths, draw_plot, draw_cities
import sys
import numpy as np
import pygame
from capitais import *
import sys
from typing import List, Tuple


# Define constant values
# pygame
WIDTH, HEIGHT = 800, 400
NODE_RADIUS = 10
FPS = 30
PLOT_X_OFFSET = 450


# GA
N_CITIES = 5
POPULATION_SIZE = 100
# Usando número de gerações para parar o processo
#N_GENERATIONS = None
N_GENERATIONS = 50
MUTATION_PROBABILITY = 0.5

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


# Capitais Brasileiras
WIDTH, HEIGHT = 1500, 800
att_cities_locations = np.array(localizacao_capitais)
max_x = max(point[0] for point in att_cities_locations)
max_y = max(point[1] for point in att_cities_locations)
scale_x = (WIDTH - PLOT_X_OFFSET - NODE_RADIUS) / max_x
scale_y = HEIGHT / max_y
cities_locations = [(int(point[0] * scale_x + PLOT_X_OFFSET),
                      int(point[1] * scale_y)) for point in att_cities_locations]

target_solution = ordem_cidades

#target_solution = knn_hotstart(20, ordem_cidades)

fitness_target_solution = calculate_fitness(target_solution)

print(f"Best Solution: {fitness_target_solution}")
# ----- Capitais Brasileiras


# Desenha pontos e caminhos na tela
def cities_coordenadas(caminho : List[int]) -> List[Tuple[float, float]]:
   
    coordenadas = []
    n = len(caminho)
    for i in range(n):
        xy = localizacao_capitais[i]     
        coordenadas.append((int(xy[0] * scale_x + PLOT_X_OFFSET),int(xy[1] * scale_y)))

    return coordenadas


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP Solver using Pygame")
clock = pygame.time.Clock()
generation_counter = itertools.count(start=1)  # Start the counter at 1


# Create Initial Population
# TODO:- use some heuristic like Nearest Neighbour our Convex Hull to initialize
#population = generate_random_population(cities_locations, POPULATION_SIZE)
#population = generate_random_population(att_48_cities_order, POPULATION_SIZE)

population_knn = []

for x in range(len(ordem_cidades)):
    individuo = knn_hotstart(x+1, ordem_cidades)
    population_knn.append(individuo)

population = generate_random_population(ordem_cidades, POPULATION_SIZE-len(population_knn))

population.extend(population_knn)

#population = generate_random_population(ordem_cidades, POPULATION_SIZE)

best_fitness_values = []
best_solutions = []


# Main game loop
running = True
while running:
   
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    generation = next(generation_counter)
    
    #Interrompe execução após N_GENERATIONS gerações
    if int(generation) == int(N_GENERATIONS):
        running = False

    screen.fill(WHITE)

    population_fitness = [calculate_fitness(
        individual) for individual in population]

    population, population_fitness = sort_population(
        population,  population_fitness)

    best_fitness = calculate_fitness(population[0])
    best_solution = population[0]

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    draw_plot(screen, list(range(len(best_fitness_values))),
              best_fitness_values, y_label="Fitness - Distance (pxls)")

    draw_cities(screen, cities_locations, RED, NODE_RADIUS)
    draw_paths(screen, cities_coordenadas(best_solution), BLUE, width=3)
    draw_paths(screen, cities_coordenadas(population[1]), rgb_color=(128, 128, 128), width=1)

    print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)}")

    new_population = [population[0]]  # Keep the best individual: ELITISM

    while len(new_population) < POPULATION_SIZE:
        
        # selection
        # simple selection based on first 10 best solutions
        #parent1, parent2 = random.choices(population[:10], k=2)
        
        #parent1, parent2 = random.choices(population, k=2)

        # solution based on fitness probability
        probability = 1 / np.array(population_fitness)
        parent1, parent2 = random.choices(population, weights=probability, k=2)
        
        #child1 = order_crossover(parent1, parent1)
        child1 = order_crossover(parent1, parent2)
        
        # child1 = order_crossover(parent1, parent2)
        # child2 = order_crossover(parent2, parent1)
        
        
        child1 = mutate(child1, MUTATION_PROBABILITY)
        new_population.append(child1)
        
        # child2 = mutate(child2, MUTATION_PROBABILITY)
        # new_population.append(child2)

    population = new_population

    pygame.display.flip()
    clock.tick(FPS)


# TODO: save the best individual in a file if it is better than the one saved.

# exit software
pygame.quit()
sys.exit()
