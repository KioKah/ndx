import numpy as np


class Result:
    def __init__(self, counter_array, starting_value):
        self.counter_array = counter_array
        self.starting_value = starting_value


def convolve(a, b):
    # Add
    l_a = len(a)
    l_b = len(b)
    counter_array = [0] * (l_a + l_b - 1)
    for s in range(l_a):
        for o in range(l_b):
            counter_array[s + o] += a[s] * b[o]
    return counter_array


def generate_convolutions(a, starting_value, size):
    results_list = []

    # Initial result (1@a)
    current_result = a.copy()

    # Loop to generate and store results
    for i in range(starting_value, starting_value + size):
        if i == 1:
            results_list.append(Result(current_result, i))
        else:
            for _ in range(i - 1):
                current_result = convolve(current_result, a)
            results_list.append(Result(current_result, i))
            # Reset current_result to base array for next iteration
            current_result = a.copy()

    return results_list


# Example usage:
A = [1, 1, 1, 1, 1, 1]  # The base array
STARTING_VALUE = 15
SIZE = 2

results = generate_convolutions(A, STARTING_VALUE, SIZE)
for result in results:
    print(f"{result.starting_value}@a: {result.counter_array}")
