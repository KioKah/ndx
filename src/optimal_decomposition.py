import numpy as np
from time import time
from typing import List, Dict, Tuple

# TODO : Finish documentation


class OptimalDecomposition:
    def __init__(self, n):
        self.optimal_decompositions = self.compute_optimal_decompositions(n)
        self.ordered_operations = self.optimal_decompositions[0][n]
        self.costs = self.optimal_decompositions[0][n]

    @staticmethod
    def compute_optimal_decompositions(
        n: int,
    ) -> Tuple[List[Dict[int, Tuple[int, int]]], List[int]]:
        """Calculate the optimal decomposition of all numbers from 0 to n.

        This function calculates the optimal decomposition of a all numbers from 0 to n into a sums of positive integers.
        The decomposition is considered optimal if it minimizes the number of sums required to obtain the number.
        You start from one and once a number has been reached, it can be used in further operations.
        For 0 and 1, no operations are required.
        To get the optimal decomposition of 6, you can start by adding 1 and 1 to get 2, then add 1 and 2 to get 3, then add 3 and 3 to get 6.
        This decomposition requires 3 operations, which is the minimum number of operations to obtain 6.

        Args:
            `n` (int): The largest number to be decomposed.

        Returns:
        Tuple[List[Dict[int, Tuple[int, int]]], List[int]]: A tuple containing two lists:
            ordered_operations (List[Dict[int, Tuple[int, int]]]): A list of dictionaries representing the ordered operations
                for each target number from 0 to n. Each dictionary maps a key (target number) to a tuple of two integers
                representing the decomposition.
            costs (List[int]): A list of integers representing the costs (number of operations) for each target number
                from 0 to n.

        Example:
            >>> optimal_decomposition(4)
            ([{}, {}, {2: (1, 1)}, {2: (1, 1), 3: (2, 1)}, {2: (1, 1), 4: (2, 2)}], [0, 0, 1, 2, 2])

        """
        operations = [{} for _ in range(n + 1)]
        costs = [np.inf] * (n + 1)
        costs[0] = 0
        costs[1] = 0

        for target in range(2, n + 1):
            for j in range(1, target // 2 + 1):
                i = target - j  # i>j

                ij_operations = {}
                all_keys = set(operations[i].keys()).union(operations[j].keys())
                for key in all_keys:
                    if key in operations[i] and key in operations[j]:
                        # If the key is in both dictionaries, choose the tuple with the higher product
                        a1, b1 = operations[i][key]
                        a2, b2 = operations[j][key]
                        if a1 * b1 >= a2 * b2:
                            ij_operations[key] = operations[i][key]
                        else:
                            ij_operations[key] = operations[j][key]
                    elif key in operations[i]:  # If the key is only in d1
                        ij_operations[key] = operations[i][key]
                    else:  # If the key is only in d2
                        ij_operations[key] = operations[j][key]
                ij_operations[target] = (i, j)

                ij_operations_cost = len(ij_operations)

                if ij_operations_cost < costs[target]:
                    operations[target] = ij_operations
                    costs[target] = ij_operations_cost
                elif ij_operations_cost == costs[target]:
                    if sum(p[0] + p[1] for p in ij_operations.values()) < sum(
                        p[0] + p[1] for p in operations[target].values()
                    ):
                        operations[target] = ij_operations
                        costs[target] = ij_operations_cost

        ordered_operations = [
            {key: operations[target][key] for key in sorted(operations[target].keys())}
            for target in range(n + 1)
        ]

        return ordered_operations, costs

    def items(self):
        return self.ordered_operations.items()


if __name__ == "__main__":
    N = 128

    start = time()
    od = OptimalDecomposition(N)
    OPERATIONS = od.optimal_decompositions[0]
    COSTS = od.optimal_decompositions[1]
    print(time() - start)

    def powers_of_two(idx: int) -> int:
        return idx.bit_length() - 1 + str(bin(idx)).count("1") - 1

    for i in range(1, N + 1):
        print(f"Optimal decomposition for n = {i}:")
        print(f"Operations: {OPERATIONS[i]}")
        print(
            "Minimum number of multiplications: "
            f"{COSTS[i]} (vs {powers_of_two(i)} with powers of two)\n"
        )

    print(
        "Average gain over powers of two approach: "
        f"{np.mean([powers_of_two(i) - COSTS[i] for i in range(1, N + 1)]):.2f} operations\n"
    )
