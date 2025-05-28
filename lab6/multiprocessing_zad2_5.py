import queue
import numpy as np
from typing import List, Tuple
import math
import datetime
import pandas as pd
import itertools
from itertools import product
from random_gen import RandomNumberGenerator
from gen_algorithm import GeneticKnapsack
from typing import Literal
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

seed = 42
np.random.seed(seed)
random_gen = RandomNumberGenerator(seedVaule=seed)


n = 1000
C, W, B = random_gen.generate_input(n=n)

SELECTION_METHOD = 'Tournament'
CROSSOVER_METHOD = 'SinglePoint'
TOURNAMENT_SIZE = 5
GENERATIONS = 1000  # reduce for speed

population_sizes = [25, 50, 100, 200]
mutation_rates = [0.01, 0.05, 0.1, 0.2]
crossover_rates = [0.1, 0.3, 0.5, 0.7, 0.9]



def run_experiment(params):
    pop_size, mut_rate, cross_rate = params
    start_time = time.time()

    results = []
    for _ in range(5):  # Repeat 5 times
        gen = GeneticKnapsack(
            costs=C, weights=W, max_weight=B,
            population_size=pop_size,
            generations=GENERATIONS,
            mutation_rate=mut_rate,
            crossover_rate=cross_rate,
            tournament_size=TOURNAMENT_SIZE,
            crossover_method=CROSSOVER_METHOD
        )
        solution_cost, solution_weight, *_ = gen.run()
        results.append((solution_cost, solution_weight))

    avg_cost = sum(r[0] for r in results) / 5
    avg_weight = sum(r[1] for r in results) / 5
    duration = time.time() - start_time

    return {
        'population_size': pop_size,
        'mutation_rate': mut_rate,
        'crossover_rate': cross_rate,
        'avg_solution_cost': avg_cost,
        'avg_solution_weight': avg_weight,
        'time_seconds': duration
    }

if __name__ == '__main__':
    print(f"Running experiments with {cpu_count()} CPU cores...")

    param_combinations = list(product(population_sizes, mutation_rates, crossover_rates))
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(run_experiment, param_combinations), total=len(param_combinations)))

    results_df = pd.DataFrame(results)
    results_df.to_csv('results/proper_combinations_results.csv', index=False)