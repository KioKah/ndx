from src.count_array import CountArray
from src.optimal_decomposition import OptimalDecomposition

from itertools import product as cartesian_product
from typing import Callable, Dict, List, Sequence, Tuple
import warnings
import math
import matplotlib.pyplot as plt


class Result:
    """`Result(count_array: CountArray, starting_value: int = 1, expression_repr: str = "")`

    Result of an Expression

    A Result is an array that counts the number of ways to obtain each value,
    offset by a starting_value - 1.

    For example 2d6 gives the following count_array:
    [1 2 3 4 5 6 5 4 3 2 1] with a starting_value of 2.

    Attributes:
        `__count_array` (List[int]) : Array of counts for each value
        `__starting_value` (int) : Starting value of the count_array
        `__total_count` (int) : Total number of ways to obtain a value
        `__probabilities` (Dict[int, float]) : Values and probabilities of drawing each value
        `__title` (str) : Title of the plot
    """

    def __init__(
        self,
        count_array: CountArray,
        starting_value: int = 1,
        title: str = "",
    ):
        # Checks
        if not isinstance(count_array, CountArray):
            raise TypeError("count_array must be a CountArray")
        if count_array.size > 0:
            if not all(count >= 0 for count in count_array):
                raise ValueError("count_array values must be >= 0")
        # Init
        self.__count_array = count_array
        self.__starting_value = starting_value
        self.__total_count = count_array.total()
        self.__probabilities = None
        self.__title = title
        ## print(">", str(self))

    # Getters
    @property
    def count_array(self) -> CountArray:
        """Getter for `__count_array`"""
        return self.__count_array

    @property
    def starting_value(self) -> int:
        """Getter for `__starting_value`"""
        return self.__starting_value

    @property
    def total_count(self) -> int:
        """Getter for `__total_count`"""
        return self.__total_count

    @property
    def probabilities(self) -> Dict[int, float]:
        """Getter for `__probabilities` but also computes the probabilities if not already done"""
        if self.__probabilities is None:
            self.__probabilities = self.compute_probabilities()
        return self.__probabilities

    @property
    def title(self) -> str:
        """Getter for `__title`"""
        return self.__title

    # Main methods
    def set_gcd_to_one(self):
        """Set the greatest common divisor of the count_array to 1

        Divide each element of the count_array by the greatest common divisor
        of the count_array to simplify the Result
        """
        gcd = self.__count_array.gdc()
        if gcd != 1:
            self.__count_array //= gcd
            self.__total_count //= gcd

    def plot(self, title: str):
        """Plot the count_array"""
        start = self.__starting_value
        size = self.__count_array.size
        end = start + size - 1
        x_axis = range(start, end + 1)

        probabilities = [p * 100 for p in self.probabilities.values()]
        uniform_probability = 100 / size
        max_probability = max(probabilities)

        colors = []
        MAX_COLOR = "#003AB0"
        HIGH_COLOR = "#0054FF"
        BASE_COLOR = "#4180FF"
        for p in probabilities:
            if p == max_probability:
                colors.append(MAX_COLOR)
            elif p > uniform_probability:
                colors.append(HIGH_COLOR)
            else:
                colors.append(BASE_COLOR)

        plt.bar(x_axis, probabilities, color=colors)
        plt.plot(
            [start - 0.5, end + 0.5],
            [uniform_probability, uniform_probability],
            "#FF0000",
            label="Moyenne",
            linewidth=0.5,
        )
        plt.title(f"{title}\nMean {self.mean():.2f} / STDev {self.stdev():.2f}")
        plt.xlabel("Résultat")
        plt.ylabel("Probabilité (%)")
        plt.legend()
        plt.show()

    def compute_probabilities(self) -> Dict[int, float]:
        return {
            i + self.__starting_value: (count / self.__total_count)
            for i, count in enumerate(self.__count_array)
        }

    def mean(self) -> float:
        return sum(
            value * probability for value, probability in self.probabilities.items()
        )

    def variance(self) -> float:
        variance_term = sum(
            value**2 * probability for value, probability in self.probabilities.items()
        )
        variance = variance_term - (self.mean()) ** 2
        return variance if variance >= 0 else 0.0

    def stdev(self) -> float:
        return math.sqrt(self.variance())

    # Static methods
    @staticmethod
    def zero(count: int = 1) -> "Result":
        """
        The "zero" of the Result class.

        Args:
            `count` (int, optional): Value inside the count_array. Defaults to 1.

        Returns: (Result)
            A Result with a count_array with the given count, of size 1.
        """
        if not isinstance(count, int):
            raise TypeError("count must be an int")
        if count < 1:
            raise ValueError("count must be > 0")
        return Result(CountArray.single(count), 0)

    @staticmethod
    def scaled_count_array(result: "Result", factor: int) -> "Result":
        """Gives the input Result with its count_array scaled by a factor.

        Args:
            `result` (Result): Input Result to scale
            `factor` (int): Scaling factor

        Returns: (Result)
            A Result with the count_array of the input Result scaled by the factor.
            Starting value is unchanged.
        """
        if not isinstance(factor, int):
            raise TypeError("factor must be an int")
        if factor < 0:
            raise ValueError("factor must be >= 0")
        if factor == 0:
            return Result.zero()
        if factor == 1:
            return result
        return Result(result.count_array * factor, result.starting_value)

    # Operations
    @staticmethod
    def equalize_totals(results: List["Result"]) -> List["Result"]:
        """Make the total count of each Result in the List equal.

        Args:
            `results` (List[Result]): List of Result to equalize.

        Returns: (List[Result])
            List of Result with equal total counts.
        """
        # Checks
        if not isinstance(results, List):
            raise TypeError("results must be a List")
        if not all(isinstance(result, Result) for result in results):
            raise TypeError("results must be a List of Result")
        # Equalize
        if len(results) == 0:
            return [Result.zero()]
        if len(results) == 1:
            return results
        total_counts = [result.total_count for result in results]
        lcm = math.lcm(*total_counts)
        equalizer = [lcm // total_count for total_count in total_counts]
        return [
            Result.scaled_count_array(result, factor) if factor != 1 else result
            for result, factor in zip(results, equalizer)
        ]

    @staticmethod
    def fuse_outcomes(
        results: List["Result"],
        weights: List[int] | None = None,
        set_gdc_to_one: bool = True,
    ) -> "Result":
        """Fuse multiple Result objects into a single Result object.

        This method takes a list of Result objects and fuses them together into a
        single Result object. The fusion process involves summing pairwise the count arrays
        of the Result objects, optionally applying weights to each count array, and optionally
        setting the greatest common divisor (GCD) of the resulting count array to one.

        Args:
            `results` (List[Result]):
                A list of Result objects to be fused.
            `weights` (List[int] | None, optional):
                A list of weights to be applied to each result.
                If None, all results are weighted equally. Defaults to None.
            `set_gdc_to_one` (bool, optional):
                Whether to set the GCD of the resulting count array to one. Defaults to True.

        Returns: (Result)
            The fused Result object.
        """
        # Checks
        if not isinstance(results, List):
            raise TypeError("results must be a List")
        if not all(isinstance(result, Result) for result in results):
            raise TypeError("results must be a List of Result")
        if isinstance(weights, List):
            if not all(isinstance(weight, int) for weight in weights):
                raise TypeError("weights must be a List of ints")
            if len(results) != len(weights):
                raise ValueError("results and weights must have the same length")
        elif weights is not None:
            raise TypeError("weights must be a List (of ints) or None")
        if not isinstance(set_gdc_to_one, bool):
            raise TypeError("gdc_to_one must be a bool")
        # Fuse
        starting_value = min(result.starting_value for result in results)
        ending_value_plus_one = max(
            result.starting_value + result.count_array.size for result in results
        )
        length = ending_value_plus_one - starting_value
        count_array = CountArray.zeros(length)
        for index, result in enumerate(results):
            weight = weights[index] if weights is not None else 1
            for i, count in enumerate(result.count_array):
                target = i + result.starting_value
                count_array[target - starting_value] += count * weight
        fused_outcomes = Result(count_array, starting_value)
        if set_gdc_to_one:
            fused_outcomes.set_gcd_to_one()
        return fused_outcomes

    @staticmethod
    def combinations(
        starting_values: List[int], ending_values: List[int]
    ) -> List[Tuple[int]]:
        """Generate combinations of values within specified ranges.

        This method takes two lists of integers, starting_values and ending_values,
        and generates all possible combinations of values within the specified ranges.

        Args:
            `starting_values` (List[int]): A list of starting values for each range.
            `ending_values` (List[int]): A list of ending values for each range.

        Returns:
            List[Tuple[int]]: The list of tuples representing the combinations
            of values within the specified ranges.
        """
        list_of_ranges = [
            range(starting_value, ending_value + 1)
            for starting_value, ending_value in zip(starting_values, ending_values)
        ]
        return list(cartesian_product(*list_of_ranges))

    # 'Too many local variables' pylint: disable=R0914
    @staticmethod
    def operation(
        operation: Callable[[Sequence[int]], int],
        results: List["Result"],
        monotonous_increasing: bool = True,
    ) -> "Result":
        """Performs an operation on a list of results.

        Args:
            `operation` (Callable[[Sequence[int]], int]):
                A callable that takes a sequence of integers and returns an integer.
            `results` (List[Result]):
                A list of Result objects.
            `monotonous_increasing` (bool, optional):
                A flag indicating whether the operation function is monotonously increasing or not.
                Allows the starting and ending values to be calculated more efficiently if True.
                Defaults to True.

        Returns:
            Result: The result of the operation.
        """
        # Checks
        if not callable(operation):
            raise TypeError("operation must be a Callable")
        if not isinstance(results, List):
            raise TypeError("results must be a List")
        no_of_results = len(results)
        if no_of_results == 0:
            raise ValueError("results must not be empty")
        if not all(isinstance(result, Result) for result in results):
            raise TypeError("results must be a List of Result")
        if not isinstance(monotonous_increasing, bool):
            raise TypeError("monotonous_increasing must be a bool")
        # Setup
        if no_of_results == 1:
            return results[0]

        count_arrays: List[CountArray] = [result.count_array for result in results]
        starting_values = [result.starting_value for result in results]
        ending_values = [
            result.starting_value + result.count_array.size - 1 for result in results
        ]
        combinations = Result.combinations(starting_values, ending_values)
        # Operation
        operated_combinations = [operation(combination) for combination in combinations]
        if monotonous_increasing:
            return_starting_value = operation(starting_values)
            return_ending_value = operation(ending_values)
        else:
            return_starting_value = min(operated_combinations)
            return_ending_value = max(operated_combinations)
        return_count_array = CountArray.zeros(
            return_ending_value - return_starting_value + 1
        )
        for operated_combination, combination in zip(
            operated_combinations, combinations
        ):
            index = operated_combination - return_starting_value
            return_count_array[index] += math.prod(
                [
                    count_array[combination[i] - starting_values[i]]
                    for i, count_array in enumerate(count_arrays)
                ]
            )
        return_result = Result(return_count_array, return_starting_value)
        return_result.set_gcd_to_one()
        return return_result

    # Operations
    def __add__(self: "Result", other: "Result") -> "Result":
        # Checks
        if not isinstance(other, Result):
            raise TypeError("other must be an Result")
        # Add
        return Result.operation(sum, [self, other])

    @staticmethod
    def sum(results: List["Result"]) -> "Result":
        """Calculates the sum of a list of Result objects.

        Args:
            `results` (List[Result]): A list of Result objects to be summed.

        Returns: (Result)
            The sum of the Result objects.
        """
        # Checks
        if not isinstance(results, List):
            raise TypeError("results must be a List")
        if not all(isinstance(result, Result) for result in results):
            raise TypeError("results must be a List of Result")
        # Sum
        return Result.operation(sum, results)

    def __neg__(self: "Result") -> "Result":
        # Setup
        count_array = self.count_array
        length = count_array.size
        # Neg
        if length < 2:
            return self
        return Result(
            count_array.reversed(),
            -(length + self.starting_value - 1),
        )

    def __sub__(self: "Result", other: "Result") -> "Result":
        # Check
        if not isinstance(other, Result):
            raise TypeError("other must be an Result")
        # Sub
        return self + (-other)

    def __mul__(self: "Result", other: "Result") -> "Result":
        # Checks
        if not isinstance(other, Result):
            raise TypeError("other must be an int or an Result")
        # Mul
        return Result.operation(math.prod, [self, other], monotonous_increasing=False)

    @staticmethod
    def product(results: List["Result"]) -> "Result":
        """Calculates the product of a list of Result objects.

        Args:
            `results` (List[Result]): A list of Result objects to be multiplied.

        Returns: (Result)
            The product of the Result objects.
        """
        # Checks
        if not isinstance(results, List):
            raise TypeError("results must be a List")
        if not all(isinstance(result, Result) for result in results):
            raise TypeError("results must be a List of Result")
        # Product
        return Result.operation(math.prod, results, monotonous_increasing=False)

    def __floordiv__(self: "Result", other: "Result") -> "Result":
        # Checks
        if not isinstance(other, Result):
            raise TypeError("other must be an Result")

        # FloorDiv
        def safe_floordiv(fraction: Tuple[int]):
            if not isinstance(fraction, tuple) or len(fraction) != 2:
                raise ValueError("fraction must be a tuple containing two integers")
            if not all(isinstance(value, int) for value in fraction):
                raise ValueError("fraction must contain only integers")
            numerator, denominator = fraction
            if denominator == 0:
                warnings.warn("Division by zero, returned 0", RuntimeWarning)
                return 0
            return numerator // denominator

        return Result.operation(
            safe_floordiv, [self, other], monotonous_increasing=False
        )

    @staticmethod
    def floor_division(numerator: "Result", denominators: List["Result"]) -> "Result":
        """Calculates the (safe) floor division of a Result object by a list of Result objects.

        Args:
            `numerator` (Result): The Result object to be divided.
            `denominators` (List[Result]): A list of Result objects to divide by in order.

        Returns: (Result)
            The division of the Result object by the Result objects.
        """
        # Checks
        if not isinstance(numerator, Result):
            raise TypeError("numerator must be an Result")
        if not isinstance(denominators, List):
            raise TypeError("denominators must be a List")
        if not all(isinstance(denominator, Result) for denominator in denominators):
            raise TypeError("denominators must be a List of Result")
        # Division
        division_result = numerator
        for denominator in denominators:
            division_result = division_result // denominator
        return division_result

    def ceil_div(self: "Result", other: "Result") -> "Result":
        # Checks
        if not isinstance(other, Result):
            raise TypeError("other must be an Result")

        # FloorDiv
        def safe_ceil_div(fraction: Tuple[int]):
            if not isinstance(fraction, tuple) or len(fraction) != 2:
                raise ValueError("fraction must be a tuple containing two integers")
            if not all(isinstance(value, int) for value in fraction):
                raise ValueError("fraction must contain only integers")
            numerator, denominator = fraction
            if denominator == 0:
                warnings.warn("Division by zero, returned 0", RuntimeWarning)
                return 0
            div, mod = divmod(numerator, denominator)
            return div + (mod > 0)

        return Result.operation(
            safe_ceil_div, [self, other], monotonous_increasing=False
        )

    @staticmethod
    def ceil_division(numerator: "Result", denominators: List["Result"]) -> "Result":
        """Calculates the (safe) ceil division of a Result object by a list of Result objects.

        Args:
            `numerator` (Result): The Result object to be divided.
            `denominators` (List[Result]): A list of Result objects to divide by in order.

        Returns: (Result)
            The division of the Result object by the Result objects.
        """
        # Checks
        if not isinstance(numerator, Result):
            raise TypeError("numerator must be an Result")
        if not isinstance(denominators, List):
            raise TypeError("denominators must be a List")
        if not all(isinstance(denominator, Result) for denominator in denominators):
            raise TypeError("denominators must be a List of Result")
        # Division
        division_result = numerator
        for denominator in denominators:
            division_result = division_result.ceil_div(denominator)
        return division_result

    def __matmul__(self: "Result", other: "Result") -> "Result":
        # TODO : REVAMP !!
        # Check
        if not isinstance(other, Result):
            raise TypeError("other must be an Result")
        # Setup
        s_count_array = self.count_array
        s_starting_value = self.starting_value
        s_length = s_count_array.size

        if s_starting_value + s_length - 1 < 0:
            return -((-self) @ other)

        # MatMul
        added_other_list: List[Result] = [None] * (s_starting_value + s_length)
        added_other_list[0] = Result.zero()
        added_other_list[1] = other
        if s_starting_value >= 0:
            if s_starting_value != 1:
                optimal_decomposition = OptimalDecomposition(s_starting_value)
                for target, (i, j) in optimal_decomposition.ordered_operations.items():
                    added_other_list[target] = added_other_list[i] + added_other_list[j]
            # NOTE : Could be optimized for s_count_array with zeros.
            for target in range(s_starting_value + 1, s_starting_value + s_length):
                # other could be written as results_list[1] below
                added_other_list[target] = added_other_list[target - 1] + other
            targets = [
                i + s_starting_value
                for i, count in enumerate(s_count_array)
                if count > 0
            ]
            targets_results = [added_other_list[target] for target in targets]
            targets_weights = [
                s_count_array[target - s_starting_value] for target in targets
            ]
            equalized_results = Result.equalize_totals(targets_results)
            return Result.fuse_outcomes(
                equalized_results, weights=targets_weights, set_gdc_to_one=True
            )
        # -1 and 0 is in self_count_array
        raise NotImplementedError()
        # s_geq0_count_array = s_count_array[-s_starting_value:]
        # s_lt0_count_array = s_count_array[:-s_starting_value]
        # s_geq0_length = s_geq0_count_array.size
        # s_lt0_length = s_lt0_count_array.size
        # s_max_length = max(s_geq0_length, s_lt0_length)
        # for target in range(1, s_max_length + 1):
        #     added_other_list[target + 1] = added_other_list[target] + other

    # TODO ?
    # def matpow(self: "Result", other: int) -> "Result":
    #     if not isinstance(other, int):
    #         raise TypeError("other must be an int")
    #     if other < 0:
    #         raise ValueError("other must be >= 0")
    #     if other == 0:
    #         return Result.zero()
    #     if other == 1:
    #         return self
    #     return self @ (self.matpow(other - 1))

    @staticmethod
    def advantage(results: List["Result"]) -> "Result":
        """Calculates the advantage result based on a list of Result objects.

        Args:
            results (List[Result]): The results the advantage operation is performed on.

        Returns:
            Result: The Result object representing the calculated advantage.
        """
        # Checks
        if not isinstance(results, List):
            raise TypeError("results must be a List")
        if not all(isinstance(result, Result) for result in results):
            raise TypeError("results must be a List of Result")
        # Product
        return Result.operation(max, results, monotonous_increasing=True)

    @staticmethod
    def disadvantage(results: List["Result"]) -> "Result":
        """Calculates the disadvantage result based on a list of Result objects.

        Args:
            results (List[Result]): The results the disadvantage operation is performed on.

        Returns:
            Result: The Result object representing the calculated disadvantage.
        """
        # Checks
        if not isinstance(results, List):
            raise TypeError("results must be a List")
        if not all(isinstance(result, Result) for result in results):
            raise TypeError("results must be a List of Result")
        # Product
        return Result.operation(min, results, monotonous_increasing=True)

    # Shift
    def __lshift__(self: "Result", other: int) -> "Result":
        # Checks
        if not isinstance(other, int):
            raise TypeError("other must be an int")
        if other == 0:
            return self
        # Return
        return Result(self.count_array, self.starting_value - other)

    def __rshift__(self: "Result", other: int) -> "Result":
        # Checks
        if not isinstance(other, int):
            raise TypeError("other must be an int")
        if other == 0:
            return self
        # Return
        return Result(self.count_array, self.starting_value + other)

    def __eq__(self: "Result", other: "Result") -> bool:
        if not isinstance(other, Result):
            return False
        return (
            self.count_array == other.count_array
            and self.starting_value == other.starting_value
        )

    def __ne__(self: "Result", other: "Result") -> bool:
        if not isinstance(other, Result):
            return True
        return (
            self.count_array != other.count_array
            or self.starting_value != other.starting_value
        )

    def __gt__(self: "Result", other: "Result") -> bool:
        if not isinstance(other, Result):
            raise TypeError("other must be an Result")
        return self.total_count > other.total_count

    # String representation
    def __str__(self) -> str:
        return f"{self.__count_array}_{self.__starting_value}"

    def __repr__(self) -> str:
        return f"Result({repr(self.__count_array)}, {self.__starting_value})"
