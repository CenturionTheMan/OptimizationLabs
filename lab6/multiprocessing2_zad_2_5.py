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

GENERATIONS = 1000
REPEATS = 5

df = pd.read_csv('results/proper_combinations_results.csv', delimiter=',')
df_sorted = df.sort_values(by='avg_solution_cost', ascending=False)
best_solution = df_sorted.iloc[0]

BEST_POP = best_solution['population_size']
BEST_MUT = best_solution['mutation_rate']
BEST_CROSS = best_solution['crossover_rate']

def run_experiment(args):
    n, selection_method, crossover_method, tournament_size = args

    C, W, B = random_gen.generate_input(n=n)

    results = []
    start = time.time()
    for _ in range(REPEATS):
        gen = GeneticKnapsack(
            costs=C,
            weights=W,
            max_weight=B,
            population_size=BEST_POP,
            generations=GENERATIONS,
            mutation_rate=BEST_MUT,
            crossover_rate=BEST_CROSS,
            selection_method=selection_method,
            crossover_method=crossover_method,
            tournament_size=tournament_size
        )
        cost, weight, *_ = gen.run()
        results.append((cost, weight))

    avg_cost = sum(r[0] for r in results) / REPEATS
    avg_weight = sum(r[1] for r in results) / REPEATS
    duration = time.time() - start

    return {
        'problem_size': n,
        'selection_method': selection_method,
        'crossover_method': crossover_method,
        'tournament_size': tournament_size if selection_method == 'Tournament' else None,
        'avg_solution_cost': avg_cost,
        'avg_solution_weight': avg_weight,
        'time_seconds': duration
    }

if __name__ == '__main__':
    # variables:
    selection_methods = ['Tournament', 'Roulette']
    crossover_methods = ['SinglePoint', 'TwoPoint'] 
    problem_sizes = list(range(100, 2001, 200))  # 100 2000 co 200
    tournament_sizes = [3, 5, 7]

    for selection in selection_methods:
        param_list = []
        for n, crossover in product(problem_sizes, crossover_methods):
            if selection == 'Tournament':
                for t_size in tournament_sizes:
                    param_list.append((n, selection, crossover, t_size))
            else:
                param_list.append((n, selection, crossover, None))

        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(run_experiment, param_list), total=len(param_list)))

        df_results = pd.DataFrame(results)
        out_path = f'results/best_params/{selection}.csv'
        df_results.to_csv(out_path, index=False)
        print(f"Saved results for selection method '{selection}' to: {out_path}")
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

GENERATIONS = 1000
REPEATS = 5

df = pd.read_csv('results/combinations_results.csv', delimiter=',')
df_sorted = df.sort_values(by='solution_cost', ascending=False)
best_solution = df_sorted.iloc[0]

BEST_POP = best_solution['population_size']
BEST_MUT = best_solution['mutation_rate']
BEST_CROSS = best_solution['crossover_rate']

def run_experiment(args):
    n, selection_method, crossover_method, tournament_size = args

    C, W, B = random_gen.generate_input(n=n)

    results = []
    start = time.time()
    for _ in range(REPEATS):
        gen = GeneticKnapsack(
            costs=C,
            weights=W,
            max_weight=B,
            population_size=BEST_POP,
            generations=GENERATIONS,
            mutation_rate=BEST_MUT,
            crossover_rate=BEST_CROSS,
            selection_method=selection_method,
            crossover_method=crossover_method,
            tournament_size=tournament_size
        )
        cost, weight, *_ = gen.run()
        results.append((cost, weight))

    avg_cost = sum(r[0] for r in results) / REPEATS
    avg_weight = sum(r[1] for r in results) / REPEATS
    duration = time.time() - start

    return {
        'problem_size': n,
        'selection_method': selection_method,
        'crossover_method': crossover_method,
        'tournament_size': tournament_size if selection_method == 'Tournament' else None,
        'avg_solution_cost': avg_cost,
        'avg_solution_weight': avg_weight,
        'time_seconds': duration
    }

if __name__ == '__main__':
    # variables:
    selection_methods = ['Tournament', 'Roulette']
    crossover_methods = ['SinglePoint', 'TwoPoint'] 
    problem_sizes = list(range(100, 2001, 200))  # 100 2000 co 200
    tournament_sizes = [3, 5, 7]

    for selection in selection_methods:
        param_list = []
        for n, crossover in product(problem_sizes, crossover_methods):
            if selection == 'Tournament':
                for t_size in tournament_sizes:
                    param_list.append((n, selection, crossover, t_size))
            else:
                param_list.append((n, selection, crossover, None))

        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(run_experiment, param_list), total=len(param_list)))

        df_results = pd.DataFrame(results)
        out_path = f'results/best_params/{selection}.csv'
        df_results.to_csv(out_path, index=False)
        print(f"Saved results for selection method '{selection}' to: {out_path}")