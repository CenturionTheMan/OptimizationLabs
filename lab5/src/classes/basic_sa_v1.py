import datetime
import math
from typing import List, Literal, Tuple
from classes.task import Task
import numpy as np

class BasicSA:
    def __init__(self,
                 T0: float, 
                 alpha: float, 
                 max_i: int, 
                 cooling: Literal['linear', 'geometric']):
        self.T0 = T0
        self.alpha = alpha
        self.max_i = max_i
        self.cooling = cooling

    def get_neighbor(self,tasks_order: List[Task]):
        tmp = tasks_order.copy()
        index1 = np.random.randint(0, len(tasks_order))
        index2 = np.random.randint(0, len(tasks_order))
        while index1 == index2:
            index2 = np.random.randint(0, len(tasks_order))
        tmp[index1], tmp[index2] = tmp[index2], tmp[index1]
        return tmp

    def get_fitness(self,tasks_order: List[Task]) -> float:
        c = 0
        result = 0
        for task in tasks_order:
            result += max(0, c - task.time) * task.weight
            c += task.time
        return result


    def run(self, taks: List[Task]) -> Tuple[List[Task], float, float, float]:
        """
        Run the simulated annealing algorithm.
        :param taks: List of tasks to schedule.
        :return: Tuple containing the best order of tasks, its fitness, time of finding the best solution, and initial fitness.
        """
        iteration = 0
        T = self.T0

        current = np.random.choice(taks, len(taks), replace=False).tolist()
        current_fit = self.get_fitness(current)
        initial_fitness = current_fit
        
        time_start = datetime.datetime.now()
        best_found_time = 0
        best = current.copy()
        best_fit = current_fit
            
        while iteration < self.max_i:
            possible = self.get_neighbor(current)        
            possible_fit = self.get_fitness(possible)

            if possible_fit < current_fit:
                current_fit = possible_fit
                current = possible
                
                if current_fit < best_fit:
                    best_found_time = (datetime.datetime.now() - time_start).total_seconds()
                    best = current.copy()
                    best_fit = current_fit
            else:
                delta = possible_fit - current_fit
                try:
                    prob = math.exp(-delta / T)
                except OverflowError:
                    prob = 0
                if np.random.rand() < prob:
                    current_fit = possible_fit
                    current = possible

            
            iteration += 1
            
            if self.cooling == 'linear':
                T -= self.alpha
            elif self.cooling == 'geometric':
                T *= self.alpha
            else:
                raise ValueError("Unknown cooling type")
        return best, best_fit, best_found_time, initial_fitness