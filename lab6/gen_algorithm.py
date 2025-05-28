from dataclasses import dataclass
import numpy as np
from typing import List, Literal, Tuple

@dataclass
class Individual:
    genes: np.ndarray
    fitness: int
    sum_weight: int
    def copy(self):
        return Individual(self.genes.copy(), self.fitness, self.sum_weight)

class GeneticKnapsack:
    def __init__(self, costs: np.ndarray, weights: np.ndarray, max_weight: int, 
                 population_size: int, generations: int, mutation_rate: float, 
                 crossover_rate: float, 
                 selection_method: Literal['Tournament', 'Roulette'] = 'Tournament',
                 crossover_method: Literal['SinglePoint', 'TwoPoint'] = 'SinglePoint',
                 tournament_size: int | None = None,
                 ):
        self.costs = np.array(costs)
        self.weights = np.array(weights)
        self.max_weight = max_weight
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        
        if self.selection_method == 'Tournament' and self.tournament_size is None:
            self.tournament_size = np.clip(int(population_size * 0.1), 2, 100)
        

    def create_initial_population(self) -> List[Individual]:
        population = []
        genes_length = self.costs.shape[0]
        while len(population) < self.population_size:
            genes = np.random.randint(0, 2, size=genes_length)
            weight = np.sum(self.weights * genes)
            fitness = np.sum(self.costs * genes)
            ind = Individual(genes, fitness, weight)
            self.fix_individual(ind)
            population.append(ind)
        return population

    def evaluate_population(self, population: List[Individual]) -> float:
        return np.mean([ind.fitness for ind in population])

    def mutate_population(self, population: List[Individual]):
        for ind in population:
            if np.random.rand() < self.mutation_rate:
                while True:
                    idx = np.random.randint(0, len(ind.genes))
                    if ind.genes[idx] == 0:
                        if ind.sum_weight + self.weights[idx] > self.max_weight:
                            continue
                        ind.genes[idx] = 1
                        ind.sum_weight += self.weights[idx]
                        ind.fitness += self.costs[idx]
                    else:
                        ind.genes[idx] = 0
                        ind.sum_weight -= self.weights[idx]
                        ind.fitness -= self.costs[idx]
                    break

    def tournament_selection(self, population: List[Individual]) -> List[Individual]:
        selected = []
        while len(selected) < len(population):
            tournament = np.random.choice(population, size=self.tournament_size, replace=False)
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected.append(winner)
        return selected
    
    def roulette_selection(self, population: List[Individual]) -> List[Individual]:
        fitness_sum = sum(ind.fitness for ind in population)
        probabilities = [ind.fitness / fitness_sum for ind in population]
        selected_indices = np.random.choice(len(population), size=len(population), p=probabilities, replace=True)
        selected = [population[i] for i in selected_indices]
        return selected

    def fix_individual(self, ind: Individual):
        while ind.sum_weight > self.max_weight:
            index = np.random.choice(np.where(ind.genes == 1)[0])
            ind.genes[index] = 0
            ind.sum_weight -= self.weights[index]
            ind.fitness -= self.costs[index]

    def crossover(self, population: List[Individual]) -> List[Individual]:
        new_population = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            if i + 1 < len(population):
                parent2 = population[i + 1]
                if np.random.rand() < self.crossover_rate:
                    if self.crossover_method == 'SinglePoint':
                        cp = np.random.randint(1, len(parent1.genes) - 1)
                        child1_genes = np.concatenate((parent1.genes[:cp], parent2.genes[cp:]))
                        child2_genes = np.concatenate((parent2.genes[:cp], parent1.genes[cp:]))
                    elif self.crossover_method == 'TwoPoint':
                        cp1 = np.random.randint(1, len(parent1.genes) - 2)
                        cp2 = np.random.randint(cp1 + 1, len(parent1.genes) - 1)
                        child1_genes = np.concatenate((parent1.genes[:cp1], parent2.genes[cp1:cp2], parent1.genes[cp2:]))
                        child2_genes = np.concatenate((parent2.genes[:cp1], parent1.genes[cp1:cp2], parent2.genes[cp2:]))
                    else:
                        raise ValueError("Invalid crossover method")

                    child1_weight = np.sum(self.weights * child1_genes)
                    child2_weight = np.sum(self.weights * child2_genes)
                    child1_fitness = np.sum(self.costs * child1_genes)
                    child2_fitness = np.sum(self.costs * child2_genes)

                    child1 = Individual(child1_genes, child1_fitness, child1_weight)
                    child2 = Individual(child2_genes, child2_fitness, child2_weight)

                    self.fix_individual(child1)
                    self.fix_individual(child2)

                    new_population.append(child1)
                    new_population.append(child2)
                else:
                    new_population.append(parent1)
                    new_population.append(parent2)
            else:
                new_population.append(parent1)
        return new_population

    def run(self) -> Tuple[int, int, List[int], List[int]]:
        population = self.create_initial_population()
        best = max(population, key=lambda ind: ind.fitness).copy()

        for gen in range(self.generations):
            if self.selection_method == 'Tournament':
                population = self.tournament_selection(population)
            elif self.selection_method == 'Roulette':
                population = self.roulette_selection(population)
            else:
                raise ValueError("Invalid selection method")

            population = self.crossover(population)
            self.mutate_population(population)

            current = max(population, key=lambda ind: ind.fitness)
            if current.fitness > best.fitness:
                best = current.copy()

            # print(f"Gen {gen}: Best fitness = {current.fitness}, Avg fitness = {self.evaluate_population(population):.2f}, Best weight = {current.sum_weight}")

        selected_costs = self.costs[best.genes == 1]
        selected_weights = self.weights[best.genes == 1]
        return int(np.sum(selected_costs)), int(np.sum(selected_weights)), selected_costs.tolist(), selected_weights.tolist()
