import datetime
import math
from typing import List, Literal, Tuple
from classes.task import Task
import numpy as np

class BasicRS:
    def __init__(self,
                 neighbor: Literal['close', 'wide']='close',
                 initial: Literal['single', 'group']='single',
                 time_limit_s: int = 10, 
                 initial_size: int = 1000,
                 no_impro_iter = 1000
                 ):
        self.time_limit_s = time_limit_s
        self.initial = initial
        self.initial_size = initial_size
        self.neighbor = neighbor
        self.no_impro_iter = no_impro_iter

    def get_neighbor_wide(self,tasks_order: List[Task]):
        tmp = tasks_order.copy()
        index1, index2 = np.random.choice(len(tasks_order), 2, replace=False)
        tmp[index1], tmp[index2] = tmp[index2], tmp[index1]
        return tmp
    
    def get_neighbor_close(self,tasks_order: List[Task]):
        tmp = tasks_order.copy()
        index = np.random.randint(0, len(tasks_order)-1)
        tmp[index], tmp[index+1] = tmp[index+1], tmp[index]
        return tmp

    def get_fitness(self,tasks_order: List[Task]) -> float:
        c = 0
        result = 0
        for task in tasks_order:
            result += max(0, c - task.time) * task.weight
            c += task.time
        return result

    def get_initial(self, tasks: List[Task]) -> List[Task]:
        if self.initial == 'single':
            tmp = np.random.choice(tasks, len(tasks), replace=False).tolist()
            tmp_fit = self.get_fitness(tmp)
            return tmp, tmp_fit
        elif self.initial == 'group':
            tmp = [np.random.choice(tasks, len(tasks), replace=False).tolist() for _ in range(self.initial_size)]
            fitnesses = [self.get_fitness(t) for t in tmp]
            best_fit = min(fitnesses)
            return tmp[fitnesses.index(best_fit)], best_fit
        else:
            raise ValueError("Unknown initial solution type")


    def run(self, taks: List[Task]) -> Tuple[List[Task], float, float, float]:
        """
        Run the simulated annealing algorithm.
        :param taks: List of tasks to schedule.
        :return: Tuple containing the best order of tasks, its fitness, time of finding the best solution, and initial fitness.
        """
        iteration = 0
        reps_till_last_improvement = 0
        
        time_start = datetime.datetime.now()
        
        current, current_fit = self.get_initial(taks)
        
        ###################################### for plots
        initial_fitness = current_fit
        best_found_time = datetime.datetime.now()
        ###################################### for plots

        while (datetime.datetime.now() - time_start).total_seconds() < self.time_limit_s and self.no_impro_iter > reps_till_last_improvement:
            if self.neighbor == 'wide':
                possible = self.get_neighbor_wide(current)
            elif self.neighbor == 'close':
                possible = self.get_neighbor_close(current)
            else:
                raise ValueError("Unknown neighbor type")

            possible_fit = self.get_fitness(possible)

            if possible_fit < current_fit:
                current_fit = possible_fit
                current = possible.copy()
                reps_till_last_improvement = 0
                best_found_time = datetime.datetime.now()
            else:
                reps_till_last_improvement += 1

            iteration += 1

        return current, current_fit, best_found_time, initial_fitness