"""
    ndx stands for n x-sided dice
    # * 15d6 confirmed by numpy convolutions
"""

from typing import List, Tuple, Dict, Union, Callable, Sequence
from abc import ABC, abstractmethod
from itertools import product as cartesian_product
import math
import matplotlib.pyplot as plt

from optimal_decomposition import OptimalDecomposition

# from colorsys import hsv_to_rgb


class CountArray:
    def __init__(self, data: List[int]):
        # Checks
        if not isinstance(data, List):
            raise TypeError("data must be a List")
        size = len(data)
        if size == 0:
            raise ValueError("data must not be empty")
        if not all(isinstance(value, int) for value in data):
            raise TypeError("data must be a List of ints")
        if not all(value >= 0 for value in data):
            raise ValueError("data values must be >= 0")
        # Init
        self.__data = data
        self.__size = size

    @property
    def data(self) -> List[int]:
        """Getter for `__data`"""
        return self.__data

    @property
    def size(self) -> int:
        """Getter for `__size`"""
        return self.__size

    @staticmethod
    def zeros(length: int) -> List[int]:
        """Create a list of zeros.

        Args:
            `length` (int): Length of the list.

        Returns:
            List[int]: A list of zeros of the given length.
        """
        return CountArray([0] * length)

    @staticmethod
    def ones(length: int) -> List[int]:
        """Create a list of ones.

        Args:
            `length` (int): Length of the list.

        Returns:
            List[int]: A list of ones of the given length.
        """
        return CountArray([1] * length)

    @staticmethod
    def single(value: int) -> List[int]:
        """
        Create a list of a given value.

        Args:
            value (int): The value to fill the list with.

        Returns:
            List[int]: A list containing the given value.
        """
        return CountArray([value])

    def reversed(self) -> "CountArray":
        return CountArray(self.__data[::-1])

    def total(self) -> int:
        """Total count of the count_array

        Returns:
            int: Sum of the values from the count_array
        """
        return sum(self.__data)

    def gdc(self) -> int:
        """Greatest common divisor of the count_array

        Returns:
            int: Greatest common divisor of the count_array
        """
        return math.gcd(*self.__data)

    def __getitem__(self, index: Union[int, slice]) -> Union[int, "CountArray"]:
        if isinstance(index, slice):
            return CountArray(self.__data[index])
        if isinstance(index, int):
            return self.__data[index]
        raise TypeError("index must be an int or a slice")

    def __setitem__(self, index: Union[int, slice], value: Union[int, "CountArray"]):
        # Checks
        if not isinstance(value, (int, CountArray)):
            raise TypeError("value must be an int or a CountArray")
        # SetItem
        if isinstance(index, slice):
            if isinstance(value, CountArray):
                self.__data[index] = value.data
            else:
                self.__data[index] = value
        elif isinstance(index, int):
            self.__data[index] = value
        else:
            raise TypeError("index must be an int or a slice")

    # NOTE : Review 'CountArray operator CountArray' operations
    def __add__(self, other):
        if isinstance(other, CountArray):
            return CountArray(
                [x + y for x, y in zip(self.__data, other.data, strict=True)]
            )
        if isinstance(other, int):
            return CountArray([x + other for x in self.__data])
        raise TypeError("other must be a CountArray or an int")

    def __sub__(self, other):
        if isinstance(other, CountArray):
            return CountArray(
                [x - y for x, y in zip(self.__data, other.data, strict=True)]
            )
        if isinstance(other, int):
            return CountArray([x - other for x in self.__data])
        raise TypeError("other must be a CountArray or an int")

    def __mul__(self, other):
        if isinstance(other, CountArray):
            return CountArray(
                [x * y for x, y in zip(self.__data, other.data, strict=True)]
            )
        if isinstance(other, int):
            return CountArray([x * other for x in self.__data])
        raise TypeError("other must be a CountArray or an int")

    def __floordiv__(self, other):
        if isinstance(other, CountArray):
            return CountArray(
                [x // y for x, y in zip(self.__data, other.data, strict=True)]
            )
        if isinstance(other, int):
            return CountArray([x // other for x in self.__data])
        raise TypeError("other must be a CountArray or an int")

    def __eq__(self, other) -> bool:
        if not isinstance(other, CountArray):
            return False
        return self.__data == other.data

    def __ne__(self, other) -> bool:
        if not isinstance(other, CountArray):
            return True
        return self.__data != other.data

    def __iadd__(self, other: Union["CountArray", int]):
        if isinstance(other, CountArray):
            if self.__size != other.size:
                raise ValueError("CountArrays must have the same size")
            for i in range(self.__size):
                self.__data[i] += other[i]
        elif isinstance(other, int):
            for i in range(self.__size):
                self.__data[i] += other
        else:
            raise TypeError("other must be a CountArray or an int")

    def __isub__(self, other: Union["CountArray", int]):
        if isinstance(other, CountArray):
            for i in range(self.__size):
                if self.__size != other.size:
                    raise ValueError("CountArrays must have the same size")
                self.__data[i] -= other[i]
        elif isinstance(other, int):
            for i in range(self.__size):
                self.__data[i] -= other
        else:
            raise TypeError("other must be a CountArray or an int")

    def __imul__(self, other: Union["CountArray", int]):
        if isinstance(other, CountArray):
            for i in range(self.__size):
                if self.__size != other.size:
                    raise ValueError("CountArrays must have the same size")
                self.__data[i] *= other[i]
        elif isinstance(other, int):
            for i in range(self.__size):
                self.__data[i] *= other
        else:
            raise TypeError("other must be a CountArray or an int")

    def __ifloordiv__(self, other: Union["CountArray", int]):
        if isinstance(other, CountArray):
            for i in range(self.__size):
                if self.__size != other.size:
                    raise ValueError("CountArrays must have the same size")
                self.__data[i] //= other[i]
        elif isinstance(other, int):
            for i in range(self.__size):
                self.__data[i] //= other
        else:
            raise TypeError("other must be a CountArray or an int")

    def __str__(self):
        return f"[{' '.join([str(x) for x in self.__data])}]"

    def __repr__(self):
        return f"CountArray({repr(self.__data)})"


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
        `__expression_repr` (str) : String representation of the expression
    """

    def __init__(
        self,
        count_array: CountArray,
        starting_value: int = 1,
        expression_repr: str = "",
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
        self.__expression_repr = expression_repr
        print(">", str(self))

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
        """Getter for `__probabilities`"""
        if self.__probabilities is None:
            self.__probabilities = self.compute_probabilities()
        return self.__probabilities

    @property
    def expression_repr(self) -> str:
        """Getter for `__expression_repr`"""
        return self.__expression_repr

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
        uniform = 100 / size
        max_probability = max(probabilities)

        colors = []
        MAX_COLOR = "#003AB0"
        HIGH_COLOR = "#0054FF"
        BASE_COLOR = "#4180FF"
        for p in probabilities:
            if p == max_probability:
                colors.append(MAX_COLOR)
            elif p >= uniform:
                colors.append(HIGH_COLOR)
            else:
                colors.append(BASE_COLOR)

        plt.bar(x_axis, probabilities, color=colors)
        plt.plot(
            [start - 0.5, end + 0.5],
            [uniform, uniform],
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
        def safe_floordiv(numerator, denominator):
            return numerator // denominator if denominator != 0 else 0

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


class Expression(ABC):
    @abstractmethod
    def evaluate_result(self) -> Result:
        """Evaluate the result based on the expression.

        Returns:
            Result: The result of the operation defined by the expression.
        """


class ConstantExpression(Expression):
    def __init__(self, value: int):
        # Checks
        if not isinstance(value, int):
            raise TypeError("value must be an int")
        # Init
        self.__value = value

    def evaluate_result(self) -> Result:
        return Result(CountArray.ones(1), self.__value)

    def __str__(self):
        return str(self.__value)

    def __repr__(self):
        return f"ConstantExpression({self.__value})"


class SumOfExpressions(Expression):
    def __init__(self, expressions: List[Expression]):
        # Checks
        if not isinstance(expressions, List):
            raise TypeError("expressions must be a List")
        if not all(isinstance(expr, Expression) for expr in expressions):
            raise TypeError("expressions must be a List of Expression")
        if not len(expressions) > 1:
            raise ValueError("expressions List must have at least 2 Expression")
        # Init
        self.__expressions = expressions

    def evaluate_result(self) -> Result:
        return Result.sum([expr.evaluate_result() for expr in self.__expressions])

    def __str__(self):
        expressions_str = ") + (".join(repr(expr) for expr in self.__expressions)
        return f"({expressions_str})"

    def __repr__(self):
        expressions_repr = ", ".join(repr(expr) for expr in self.__expressions)
        return f"SumOfExpressions([{expressions_repr}])"


class ProductOfExpressions(Expression):
    def __init__(self, expressions: List[Expression]):
        # Checks
        if not isinstance(expressions, List):
            raise TypeError("expressions must be a List")
        if not all(isinstance(expr, Expression) for expr in expressions):
            raise TypeError("expressions must be a List of Expression")
        if not len(expressions) > 1:
            raise ValueError("expressions List must have at least 2 Expression")
        # Init
        self.__expressions = expressions

    def evaluate_result(self) -> Result:
        return Result.product([expr.evaluate_result() for expr in self.__expressions])

    def __str__(self):
        expressions_str = ") * (".join(repr(expr) for expr in self.__expressions)
        return f"({expressions_str})"

    def __repr__(self):
        expressions_repr = ", ".join(repr(expr) for expr in self.__expressions)
        return f"ProductOfExpressions([{expressions_repr}])"


class DivisionOfExpressions(Expression):
    def __init__(self, numerator: Expression, denominators: List[Expression]):
        # Checks
        if not isinstance(numerator, Expression):
            raise TypeError("numerator must be an Expression")
        if not isinstance(denominators, List):
            raise TypeError("denominators must be a List")
        if not all(isinstance(expr, Expression) for expr in denominators):
            raise TypeError("denominators must be a List of Expression")
        if not len(denominators) > 0:
            raise ValueError("denominators List must have at least 1 Expression")
        # Init
        self.__numerator = numerator
        self.__denominators = denominators

    def evaluate_result(self) -> Result:
        return Result.floor_division(
            self.__numerator.evaluate_results(),
            [expr.evaluate_result() for expr in self.__denominators],
        )

    def __str__(self):
        expressions_str = ") // ((".join(
            repr(expr) for expr in self.__numerator + self.__denominators
        )
        closing_parentheses = (len(self.__denominators) - 1) * ")"
        return f"({expressions_str}{closing_parentheses})"

    def __repr__(self):
        denominators_repr = ", ".join(repr(expr) for expr in self.__denominators)
        return f"ProductOfExpressions({repr(self.__numerator)}, [{denominators_repr}])"


class RepeatedExpression(Expression):
    def __init__(
        self, repetition_expression: Expression, repeated_expression: Expression
    ):
        # Checks
        if not isinstance(repetition_expression, Expression):
            raise ValueError("repetition_expression must be an Expression")
        if not isinstance(repeated_expression, Expression):
            raise ValueError("repeated_expression must be an Expression")
        # Init
        self.__repetition_expression = repetition_expression
        self.__repeated_expression = repeated_expression

    def evaluate_result(self) -> Result:
        return (
            self.__repetition_expression.evaluate_result()
            @ self.__repeated_expression.evaluate_result()
        )

    def __str__(self):
        return f"({self.__repetition_expression}) @ ({self.__repeated_expression})"

    def __repr__(self):
        repetition_repr = repr(self.__repetition_expression)
        repeated_repr = repr(self.__repeated_expression)
        return f"RepeatedExpression({repetition_repr}, {repeated_repr})"


class DieExpression(Expression):
    def __init__(self, expression: Expression):
        # Checks
        if not isinstance(expression, Expression):
            raise TypeError("expression must be an Expression")
        # Init
        self.__expression = expression

    def evaluate_result(self) -> Result:
        # Setup
        result: Result = self.__expression.evaluate_result()
        result_count_array = result.count_array
        result_starting_value = result.starting_value
        dice: List[Result] = []
        indices: List[int] = []
        # Create dice
        for index, count in enumerate(result_count_array):
            if count == 0:
                continue
            indices.append(index)
            target = index + result_starting_value
            if target < 0:
                dice.append(Result(CountArray.ones(-target), target))
            elif target == 0:
                dice.append(Result.zero())
            else:
                dice.append(Result(CountArray.ones(target), 1))
        # Fuse dice
        if len(dice) == 0:
            return Result.zero()
        if len(dice) == 1:
            return dice[0]
        # Need to equalize totals
        equalized_dice = Result.equalize_totals(dice)
        return Result.fuse_outcomes(
            [
                (
                    Result.scaled_count_array(eq_die, result_count_array[index])
                    if result_count_array[index] != 1
                    else eq_die
                )
                for eq_die, index in zip(equalized_dice, indices, strict=True)
            ],
            set_gdc_to_one=True,
        )

    def __str__(self):
        return f"d({self.__expression})"

    def __repr__(self):
        return f"DieExpression({repr(self.__expression)})"


class AdvantageExpression(Expression):
    def __init__(self, expressions: List[Expression]):
        # Checks
        if not isinstance(expressions, List):
            raise ValueError("expressions must be a List")
        if not all(isinstance(expr, Expression) for expr in expressions):
            raise ValueError("expressions must be a List of Expression")
        if not len(expressions) > 1:
            raise ValueError("expressions List must have at least 2 Expression")
        # Init
        self.__expressions = expressions

    def evaluate_result(self) -> Result:
        results = [expression.evaluate_result() for expression in self.__expressions]
        return Result.advantage(results)

    def __str__(self):
        expressions_str = ", ".join(str(expr) for expr in self.__expressions)
        return f"A({expressions_str})"

    def __repr__(self):
        expressions_repr = ", ".join(repr(expr) for expr in self.__expressions)
        return f"AdvantageExpression([{expressions_repr}])"


class DisadvantageExpression(Expression):
    def __init__(self, expressions: List[Expression]):
        # Checks
        if not isinstance(expressions, List):
            raise ValueError("expressions must be a List")
        if not all(isinstance(expr, Expression) for expr in expressions):
            raise ValueError("expressions must be a List of Expression")
        if not len(expressions) > 1:
            raise ValueError("expressions List must have at least 2 Expression")
        # Init
        self.__expressions = expressions

    def evaluate_result(self) -> Result:
        results = [expression.evaluate_result() for expression in self.__expressions]
        return Result.disadvantage(results)

    def __str__(self):
        expressions_str = ", ".join(str(expr) for expr in self.__expressions)
        return f"D({expressions_str})"

    def __repr__(self):
        expressions_repr = ", ".join(repr(expr) for expr in self.__expressions)
        return f"DisadvantageExpression([{expressions_repr}])"


class ResultExpression(Expression):
    def __init__(self, result: Result):
        if not isinstance(result, Result):
            raise ValueError("result must be an Result")
        self.__result = result

    def evaluate_result(self) -> Result:
        return self.__result

    def __str__(self):
        return str(self.__result)

    def __repr__(self):
        return f"ResultExpression({repr(self.__result)})"


class CustomExpression(Expression):
    def __init__(self, expressions_dict: Dict[Expression, int]):
        if not isinstance(expressions_dict, Dict):
            raise ValueError("expressions_dict must be a Dict")
        if not all(isinstance(value, Expression) for value in expressions_dict.keys()):
            raise ValueError("expressions_dict keys must be Expressions")
        if not all(isinstance(weight, int) for weight in expressions_dict.values()):
            raise ValueError("expressions_dict values must be ints")
        self.__expressions_dict = expressions_dict

    def evaluate_result(self) -> Result:
        equalized_results = Result.equalize_totals(
            [
                value_expr.evaluate_result()
                for value_expr in self.__expressions_dict.keys()
            ]
        )
        return Result.fuse_outcomes(
            equalized_results,
            weights=list(self.__expressions_dict.values()),
            set_gdc_to_one=True,
        )

    def __str__(self):
        expressions_str = ", ".join(
            f"{str(key)}: {value}" for key, value in self.__expressions_dict.items()
        )
        return f"{{{expressions_str}}}"

    def __repr__(self):
        expressions_repr = ", ".join(
            f"{repr(key)}: {value}" for key, value in self.__expressions_dict.items()
        )
        return f"CustomExpression({{{expressions_repr}}})"


class StringExpression:
    """NdX string expression

    expr = "2d6+1d4+1"
    => d6 + d6 + d4 & shift by 1

    -ndx = n reversed(dx) - n*(x+1)

    expr = "1d6 - 1d4 + 2"
    => d6 + reversed(d4) (==d4) & shift by 2-5 = -3

    expr = "2(d4 + d6)"
    => 2d4 + 2d6

    expr = "2*(d4 + d6)"
    => d4 + d6, dilate x2

    expr = "A(d4, d6)" or "D(d4, d6)"
    => d4 vs d6 keep highest for A, lowest for D

    expr = "1d{1, 2, 4, 5}"
    => 4-sided dice with values 1, 2, 4, 5 (uniform distribution)

    complex expressions:
    expr = "A(2d6, d{1:2,2,3,5,8}) * D(1d6, 4)d(1d2) - 40//8d[1,1]_7"
    => 1 can be omitted for 1dx
    => {n}d{x} means n dices with x faces not n times a die with x faces
    => + and * are supported for <EXPR>operator<EXPR>
    => <EXPR>-<EXPR> is just <EXPR>+(-<EXPR>)
    => // is supported but only for <EXPR>//{number}
    => you can create your own x-sided die with '{side_1, side_2, ..., side_x}'
    => you can even create weighted die with '{side_1: weight_1, side_2: weight_2}',
        weight can be omitted if 1
    => you can have more complex die d<EXPR>
    => you can even have EXPR inside custom/weighted die
    => you can have <EXPR>d<EXPR>
    => order of operations is respected. For first to last evaluated :
       (EXPR are evaluated left to right unless specified otherwise)
        - <EXPR> in A or D
        - <EXPR> ( <EXPR> )
        - A(<EXPR>, <EXPR>, ...) or D(<EXPR>, <EXPR>, ...)
        - <EXPR>d<EXPR>
        - <EXPR>*<EXPR> or <EXPR>//{number}
        - <EXPR>+<EXPR> or <EXPR>-<EXPR>
    """

    def __init__(self, str_expr: str):
        self.str_expr = str_expr
        self.expression: Expression = self.parse()
        # self.result: Result = self.evaluate()

    def parse(self) -> Expression:
        raise NotImplementedError()

    def evaluate(self) -> Result:
        return self.expression.evaluate_result()

    def __str__(self):
        return self.str_expr

    def __repr__(self):
        return f"""StringExpression("{self.str_expr}")"""


if __name__ == "__main__":
    # * 15d6 confirmed by numpy convolutions

    # PPCM/x[i] : int
    # x[i]/PGCD : int
    X = 4
    dx = DieExpression(ConstantExpression(X)).evaluate_result()
    print(f"d{X}: {dx}")
    ddx = DieExpression(ResultExpression(dx)).evaluate_result()
    print(f"dd{X}: {ddx}")
    dddx = DieExpression(ResultExpression(ddx)).evaluate_result()
    print(f"ddd{X}: {dddx}")
    # A dd's starting value is always 1
    assert ddx.starting_value == 1
    assert dddx.starting_value == 1
    print(dx.mean(), dx.stdev())
    print(ddx.mean(), ddx.stdev())
    print(dddx.mean(), dddx.stdev())

    print("\n\n")

    N = 3
    ndx = RepeatedExpression(
        ConstantExpression(N), ResultExpression(dx)
    ).evaluate_result()
    NDDX = RepeatedExpression(ConstantExpression(N), ResultExpression(ddx))
    nddx = NDDX.evaluate_result()

    print(f"{N}d{X}: {ndx}")
    print(f"{N}dd{X}: {nddx}")
    print(ndx.mean(), ndx.stdev())
    print(nddx.mean(), nddx.stdev())

    # nd.plot(
    #     str(
    #         RepeatedExpression(
    #             ConstantExpression(N),
    #             DieExpression(ConstantExpression(X)),
    #         )
    #     )
    # )

    print("\n\n")

    # Advantage test
    a = AdvantageExpression(
        [ResultExpression(dx), ResultExpression(ddx)]
    ).evaluate_result()
    print(f"A(d{X}, dd{X}): {a}")
    print(a.mean(), a.stdev())
    # A.plot(
    #     str(
    #         AdvantageExpression(
    #             [
    #                 DieExpression(ConstantExpression(X)),
    #                 DieExpression(DieExpression(ConstantExpression(X))),
    #             ]
    #         )
    #     )
    # )

    print("\n\n")

    # Disadvantage test
    d = DisadvantageExpression(
        [ResultExpression(dx), ResultExpression(ddx)]
    ).evaluate_result()
    print(f"D(d{X}, dd{X}): {d}")
    print(d.mean(), d.stdev())
    # D.plot(
    #     str(
    #         DisadvantageExpression(
    #             [
    #                 DieExpression(ConstantExpression(X)),
    #                 DieExpression(DieExpression(ConstantExpression(X))),
    #             ]
    #         )
    #     )
    # )

    print("\n\n")
    # Custom test
    c = CustomExpression(
        {
            ResultExpression(dx): 2,
            ResultExpression(ddx): 3,
        }
    ).evaluate_result()
    print(f"Custom: {c}")
    print(c.mean(), c.stdev())
    c.plot(
        str(
            CustomExpression(
                {
                    DieExpression(ConstantExpression(X)): 2,
                    DieExpression(DieExpression(ConstantExpression(X))): 3,
                }
            )
        )
    )
