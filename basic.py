import torch.nn as nn
import torch.functional as F
import torch
from typing import List


l1 = [3, [66, 55, 44], (7, 8, 9)]
l2 = list(l1)
l1.append(100)
l1[1].remove(55)


import copy
class Bus:
    def __init__(self, passengers = None):
        if passengers is None:
            self.passengers = []
        else:
            self.passengers = list(passengers)

    def pick(self, name):
        self.passengers.append(name)

    
    def remove(self, name):
        self.passengers.remove(name)


b = Bus(['Alice', 'Bill', 'Claire', 'David'])
b1 = copy.copy(b)
b2 = copy.deepcopy(b)

b.remove('Claire')
print(b.passengers)
print(b1.passengers)
print(b2.passengers)