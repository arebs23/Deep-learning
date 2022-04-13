

DATA = [
    ('Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species'),
    (5.8, 2.7, 5.1, 1.9, 'virginica'),
    (5.1, 3.5, 1.4, 0.2, 'setosa'),
    (5.7, 2.8, 4.1, 1.3, 'versicolor'),
    (6.3, 2.9, 5.6, 1.8, 'virginica'),
    (6.4, 3.2, 4.5, 1.5, 'versicolor'),
    (4.7, 3.2, 1.3, 0.2, 'setosa'),
]


from typing import List
class Iris:
    def __init__(self, features: List[float], labels:str):
        self.features = features
        self.labels = labels

    def summa(self):
        return sum(self.features)
    


# Solution 
result= {}
for *features, labels in DATA[1:]:
    iris = Iris(features, labels)
    result[iris.labels] = iris.summa()
print(f"result: {result}")