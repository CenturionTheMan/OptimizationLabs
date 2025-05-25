import math
from typing import List, Tuple

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

    def generate_input(self, n: int) -> Tuple[List[int], List[int], int]:
        S = 0
        c = []
        w = []
        for i in range(n):
            c.append(self.nextInt(1, 10))
            w.append(self.nextInt(1, 10))
            S += w[i]
        B = self.nextInt(math.floor(S / 2), math.floor(S / 2))
        return c, w, B