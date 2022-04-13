"""
* Assignment: OOP Method Sequence
* Required: yes
* Complexity: easy
* Lines of code: 9 lines
* Time: 8 min

English:
    1. Create class `Iris` with `features: list[float]` and `label: str` fields
    2. For each row in `DATA` create `Iris` instance with row values
    3. Set class attributes at the initialization from positional arguments
    4. Create method which sums values of all `features`
    5. In `result` gather species and sum of each row
    6. Run doctests - all must succeed

Polish:
    1. Stwórz klasę `Iris` z polami `features: list[float]` i `label: str`
    2. Dla każdego wiersza w `DATA` twórz instancję `Iris` z danymi z wiersza
    3. Ustaw atrybuty klasy przy inicjalizacji z argumentów pozycyjnych
    4. Stwórz metodę sumującą wartości wszystkich `features`
    5. W `result` zbieraj nazwę gatunku i sumę z każdego wiersza
    6. Uruchom doctesty - wszystkie muszą się powieść

Tests:
    >>> import sys; sys.tracebacklimit = 0

    >>> result  # doctest: +NORMALIZE_WHITESPACE
    {'setosa': 9.4,
     'versicolor': 16.299999999999997,
     'virginica': 19.3}
"""

# DATA = [
#     (4.7, 3.2, 1.3, 0.2, 'setosa'),
#     (7.0, 3.2, 4.7, 1.4, 'versicolor'),
#     (7.6, 3.0, 6.6, 2.1, 'virginica'),
# ]

# result = {}


# Solution
# class Iris:
#     def __init__(self, features, label):
#         self.features = features
#         self.label = label

#     def sum(self):
#         return sum(self.features)


# for *features, label in DATA:
#     iris = Iris(features, label)
#     result[iris.label] = iris.sum()

sizes = [1, 32, 64, 128, 256, 512]

for in_f,out_f in zip(sizes, sizes[1:]):
    print(in_f,out_f)