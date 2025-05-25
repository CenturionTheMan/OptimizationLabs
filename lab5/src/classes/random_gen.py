import math
from typing import List, Tuple
from classes.task import Task

class RandomNumberGenerator:
    def __init__(self, seedVaule=None):
        self.__seed=seedVaule
    def nextInt(self, low, high):
        m = 2147483647
        a = 16807
        b = 127773
        c = 2836
        k = int(self.__seed / b)
        self.__seed = a * (self.__seed % b) - k * c;
        if self.__seed < 0:
            self.__seed = self.__seed + m
        value_0_1 = self.__seed
        value_0_1 =  value_0_1/m
        return low + int(math.floor(value_0_1 * (high - low + 1)))
    def nextFloat(self, low, high):
        low*=100000
        high*=100000
        val = self.nextInt(low,high)/100000.0
        return val
    
    def get_data(self, n:int) -> Tuple[List[int], List[int], List[int]]:
        s = 0
        
        W = []
        P = []
        D = []
        
        for i in range(n):
            w = self.nextInt(1, 10)
            p = self.nextInt(1,100)
            s += p
            W.append(w)
            P.append(p)
        
        for i in range(n):
            d = self.nextInt(int(s/4), int(s/2))
            D.append(d)
            
        return W, P, D
    
    def get_random_tasks(self, n: int) -> List[Task]:
        W, P, D = self.get_data(n)
        tasks = [Task(p, w, d) for p, w, d in zip(P, W, D)]
        return tasks
            