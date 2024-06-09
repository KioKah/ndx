"""
    Hey
"""

from typing import List, Union, Dict
from abc import ABC, abstractmethod
from functools import reduce
from itertools import product as cartesian_product
import math

from optimal_decomposition import OptimalDecomposition

import matplotlib.pyplot as plt

# from statistics import mean
# from statistics import stdev
# from colorsys import hsv_to_rgb


class CounterArray:
    def __init__(self, data: List[int]):
        # Checks
        if not isinstance(data, List):
            raise TypeError("data must be a List")
        if len(data) == 0:
            raise ValueError("data must not be empty")
        if not all(isinstance(value, int) for value in data):
            raise TypeError("data must be a List of ints")
        if not all(value >= 0 for value in data):
            raise ValueError("data values must be >= 0")
        # Init
        self.data = data
        self.size = len(data)

    @staticmethod
    def zeros(length: int) -> List[int]:
        """Create a list of zeros.

        Args:
            `length` (int): Length of the list.

        Returns:
            List[int]: A list of zeros of the given length.
        """
        return CounterArray([0] * length)

    @staticmethod
    def ones(length: int) -> List[int]:
        """Create a list of ones.

        Args:
            `length` (int): Length of the list.

        Returns:
            List[int]: A list of ones of the given length.
        """
        return CounterArray([1] * length)

    @staticmethod
    def single(value: int) -> List[int]:
        """
        Create a list of a given value.

        Args:
            value (int): The value to fill the list with.

        Returns:
            List[int]: A list containing the given value.
        """
        return CounterArray([value])

    def lcm(self) -> int:
        """Least common multiple of the counter_array

        Returns:
            int: Least common multiple of the counter_array
        """
        return math.lcm(*self.data)

    def gdc(self) -> int:
        """Greatest common divisor of the counter_array

        Returns:
            int: Greatest common divisor of the counter_array
        """
        return math.gcd(*self.data)

    def prod(self) -> int:
        """Product of the counter_array

        Returns:
            int: Product of the counter_array
        """
        return math.prod(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __repr__(self):
        return f"Array({self.data})"

    # TODO : Check if this is the right way to implement these operators
    def __add__(self, other):
        if isinstance(other, CounterArray):
            return CounterArray([x + y for x, y in zip(self.data, other.data)])
        else:
            return CounterArray([x + other for x in self.data])

    def __sub__(self, other):
        if isinstance(other, CounterArray):
            return CounterArray([x - y for x, y in zip(self.data, other.data)])
        else:
            return CounterArray([x - other for x in self.data])

    def __mul__(self, other):
        if isinstance(other, CounterArray):
            return CounterArray([x * y for x, y in zip(self.data, other.data)])
        else:
            return CounterArray([x * other for x in self.data])

    # NOTE: Truediv has no place being here
    def __truediv__(self, other):
        if isinstance(other, CounterArray):
            return CounterArray([x / y for x, y in zip(self.data, other.data)])
        else:
            return CounterArray([x / other for x in self.data])


class Result:
    """`Result(counter_array: List[int], starting_value: int = 1)`

    Result of an Expression

    A Result is an array that counts the number of ways to obtain each value,
    offset by a starting_value - 1.

    For example 2d6 gives the following counter_array:
    [1 2 3 4 5 6 5 4 3 2 1] with a starting_value of 2.

    Attributes:
        `__counter_array` (List[int]) : Array of counts for each value
        `__starting_value` (int) : Starting value of the counter_array
        `__total_count` (int) : Total number of ways to obtain a value
        `__lcm` (int) : Least common multiple of the counter_array
    """

    def __init__(
        self,
        counter_array: CounterArray,
        starting_value: int = 1,
    ):
        # Checks
        if not isinstance(counter_array, CounterArray):
            raise TypeError("counter_array must be a CounterArray")
        if counter_array.size > 0:
            if not all(count >= 0 for count in counter_array):
                raise ValueError("counter_array values must be >= 0")
            # Init
            lcm = counter_array.lcm()
        else:
            lcm = 1
        self.__lcm = lcm
        self.__counter_array = counter_array
        self.__starting_value = starting_value
        self.__total_count = sum(self.__counter_array)

    def set_gcd_to_one(self):
        """Set the greatest common divisor of the counter_array to 1

        Divide each element of the counter_array by the greatest common divisor
        of the counter_array to simplify the Result
        """
        if self.__counter_array.size == 0:
            return
        gcd = self.__counter_array.gdc()
        if gcd != 1:
            self.__counter_array //= gcd
            self.__lcm //= gcd
            self.__total_count //= gcd

    def plot(self):
        """Plot the counter_array"""
        plt.bar(
            range(
                self.__starting_value, self.__starting_value + self.__counter_array.size
            ),
            self.__counter_array,
        )
        plt.show()
        # TODO : Add title, labels, etc.

    # Getters
    @property
    def lcm(self) -> int:
        """Getter for `__lcm`"""
        return self.__lcm

    @property
    def counter_array(self) -> List[int]:
        """Getter for `__counter_array`"""
        return self.__counter_array

    @property
    def starting_value(self) -> int:
        """Getter for `__starting_value`"""
        return self.__starting_value

    @property
    def total_count(self) -> int:
        """Getter for `__total_count`"""
        return self.__total_count

    # Static methods
    @staticmethod
    def zero(count: int = 1) -> "Result":
        """
        The "zero" of the Result class.

        Args:
            `count` (int, optional): Value inside the counter_array. Defaults to 1.

        Returns: (Result)
            A Result with a counter_array with the given count, of size 1.
        """
        if not isinstance(count, int):
            raise TypeError("count must be an int")
        if count < 1:
            raise ValueError("count must be > 0")
        return Result(CounterArray.single(count), 0)

    # @staticmethod
    # def one(count: int = 1) -> "Result":
    #     """The "one" of the Result class.

    #     Args:
    #         `count` (int, optional): Value inside the counter_array. Defaults to 1.

    #     Returns: (Result)
    #         A Result with a starting value of zero
    #         and counter_array with the given count, of size 1.
    #     """
    #     if not isinstance(count, int):
    #         raise TypeError("count must be an int")
    #     if count < 1:
    #         raise ValueError("count must be > 0")
    #     return Result(Array.single(count), 1)

    @staticmethod
    def scaled_counter_array(result: "Result", factor: int) -> "Result":
        """Gives the input Result with its counter_array scaled by a factor.

        Args:
            `result` (Result): Input Result to scale
            `factor` (int): Scaling factor

        Returns: (Result)
            A Result with the counter_array of the input Result scaled by the factor.
            Starting value is unchanged.
        """
        if not isinstance(factor, int):
            raise TypeError("factor must be an int")
        if factor < 0:
            raise ValueError("factor must be >= 0")
        if factor == 0:
            return Result.zero()
        return Result(result.counter_array * factor, result.starting_value)

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
            Result.scaled_counter_array(result, factor)
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
        single Result object. The fusion process involves summing pairwise the counter arrays
        of the Result objects, optionally applying weights to each counter array, and optionally
        setting the greatest common divisor (GCD) of the resulting counter array to one.

        Args:
            `results` (List[Result]):
                A list of Result objects to be fused.
            `weights` (List[int] | None, optional):
                A list of weights to be applied to each result.
                If None, all results are weighted equally. Defaults to None.
            `set_gdc_to_one` (bool, optional):
                Whether to set the GCD of the resulting counter array to one. Defaults to True.

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
        length = (
            max(result.starting_value + result.counter_array.size for result in results)
            - starting_value
            + 1
        )
        counter_array = CounterArray.zeros(length)
        for index, result in enumerate(results):
            weight = weights[index] if weights is not None else 1
            for i, count in enumerate(result.counter_array):
                target = i + result.starting_value
                counter_array[target - starting_value] += count * weight
        fused_outcomes = Result(counter_array, starting_value)
        if set_gdc_to_one:
            fused_outcomes.set_gcd_to_one()
        return fused_outcomes

    def __add__(self: "Result", other: "Result") -> "Result":
        # Checks
        if not isinstance(other, Result):
            raise TypeError("other must be an Result")
        # Setup
        s_counter_array = self.counter_array
        o_counter_array = other.counter_array
        s_length = s_counter_array.size
        o_length = o_counter_array.size
        # Add
        starting_value = self.starting_value + other.starting_value
        if s_length == 1:
            return Result(o_counter_array, starting_value)
        if o_length == 1:
            return Result(s_counter_array, starting_value)
        counter_array = CounterArray.zeros(s_length + o_length - 1)
        for s in range(s_length):
            for o in range(o_length):
                counter_array[s + o] += s_counter_array[s] * o_counter_array[o]
        return Result(counter_array, starting_value)

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
        if len(results) == 0:
            return Result.zero()
        if len(results) == 1:
            return results[0]
        return reduce(lambda r1, r2: r1 + r2, results)

    def __neg__(self: "Result") -> "Result":
        # Setup
        counter_array = self.counter_array
        length = counter_array.size
        # Neg
        if length < 2:
            return self
        return Result(
            counter_array[::-1],
            -(length + self.starting_value - 1),
        )

    def __sub__(self: "Result", other: "Result") -> "Result":
        # Check
        if not isinstance(other, Result):
            raise TypeError("other must be an Result")
        # Sub
        return self + (-other)

    def __matmul__(self: "Result", other: "Result") -> "Result":
        # TODO : REVAMP !!
        # Check
        if not isinstance(other, Result):
            raise TypeError("other must be an Result")
        # Setup
        s_counter_array = self.counter_array
        s_starting_value = self.starting_value
        s_length = s_counter_array.size

        if s_starting_value + s_length - 1 < 0:
            return -((-self) @ other)

        # MatMul
        if s_starting_value >= 0:
            results_list: List[Result] = [None] * (s_starting_value + s_length)
            results_list[0] = Result.zero()
            results_list[1] = other
            if s_starting_value != 1:
                optimal_decomposition = OptimalDecomposition(s_starting_value)
                for target, (i, j) in optimal_decomposition.ordered_operations.items():
                    results_list[target] = results_list[i] + results_list[j]
            # NOTE : Could be optimized for s_counter_array with zeros.
            for target in range(s_starting_value + 1, s_starting_value + s_length):
                # other could be written as results_list[1] below
                results_list[target] = results_list[target - 1] + other
            targets = [
                i + s_starting_value
                for i, count in enumerate(s_counter_array)
                if count > 0
            ]
            targets_results = [results_list[target] for target in targets]
            targets_weights = [
                s_counter_array[target - s_starting_value] for target in targets
            ]
            equalized_results = Result.equalize_totals(targets_results)
            return Result.fuse_outcomes(
                equalized_results, weights=targets_weights, set_gdc_to_one=True
            )
        else:  # -1 and 0 is in self_counter_array
            s_geq0_counter_array = s_counter_array[-s_starting_value:]
            s_lt0_counter_array = s_counter_array[:-s_starting_value]
            s_geq0_length = s_geq0_counter_array.size
            s_lt0_length = s_lt0_counter_array.size
            s_max_length = max(s_geq0_length, s_lt0_length)
            results_list: List[Result] = [None] * (s_max_length + 1)
            results_list[0] = Result.zero()
            results_list[1] = other
            for target in range(1, s_max_length + 1):
                results_list[target + 1] = results_list[target] + other
            return Result.sum(
                [
                    Result.scaled_counter_array(
                        results_list[i], s_geq0_counter_array[i]
                    )
                    for i in range(s_geq0_length)
                ]
            ) - Result.sum(
                [
                    s_lt0_counter_array[i] * results_list[i + 1]
                    for i in range(s_lt0_length)
                ]
            )

    def matpow(self: "Result", other: int) -> "Result":
        # TODO
        if not isinstance(other, int):
            raise TypeError("other must be an int")
        if other < 0:
            raise ValueError("other must be >= 0")
        if other == 0:
            return Result.zero()
        if other == 1:
            return self
        return self @ (self.matpow(other - 1))

    def __mul__(self: "Result", other: Union[int, "Result"]) -> "Result":
        # Checks
        if isinstance(other, int):
            return other * self
        if not isinstance(other, Result):
            raise TypeError("other must be an int or an Result")
        # Setup
        s_counter_array = self.counter_array
        o_counter_array = other.counter_array
        s_length = s_counter_array.size
        o_length = o_counter_array.size
        s_starting_value = self.starting_value
        o_starting_value = other.starting_value
        # Mul
        values = {
            s * o
            for s in range(s_starting_value, s_starting_value + s_length)
            for o in range(o_starting_value, o_starting_value + o_length)
        }
        starting_value = min(values)
        counter_array = CounterArray.zeros(max(values) - starting_value + 1)
        for s in range(s_length):
            for o in range(o_length):
                counter_array[
                    (s + s_starting_value) * (o + o_starting_value) - starting_value
                ] += (s_counter_array[s] * o_counter_array[o])
        return Result(counter_array, starting_value)

    # Shift
    def __lshift__(self: "Result", other: int) -> "Result":
        # Checks
        if not isinstance(other, int):
            raise TypeError("other must be an int")
        if other == 0:
            return self
        # Return
        return Result(self.counter_array, self.starting_value - other)

    def __rshift__(self: "Result", other: int) -> "Result":
        # Checks
        if not isinstance(other, int):
            raise TypeError("other must be an int")
        if other == 0:
            return self
        # Return
        return Result(self.counter_array, self.starting_value + other)

    # r operations
    def __rmul__(self, other: int) -> "Result":
        # Checks
        if not isinstance(other, int):
            raise TypeError("other must be an int")
        # Return
        if other == 0:
            return Result.zero()
        if other < 0:
            return -(ConstantExpression(-other).evaluate_result() * self)
        return ConstantExpression(other).evaluate_result() * self

    def __rmatmul__(self, other: int) -> "Result":
        # Checks
        if not isinstance(other, int):
            raise TypeError("other must be an int")
        # Return
        if other == 0:
            return Result.zero(self.total_count())
        if other < 0:
            return -(ConstantExpression(-other).evaluate_result() @ self)
        return ConstantExpression(other).evaluate_result() @ self

    # String representation
    def __str__(self):
        return f"{self.__counter_array}_{self.__starting_value}"

    def __repr__(self):
        return f"Result({repr(self.__counter_array)}, {self.__starting_value})"


class Expression(ABC):
    @abstractmethod
    def evaluate_result(self) -> Result:
        """Evaluate the result based on the expression.

        Returns:
            Result: The result of the operation defined by the expression.
        """

    def __str__(self):
        return f"{self.__class__.__name__}"


class ConstantExpression(Expression):
    def __init__(self, value: int):
        # Checks
        if not isinstance(value, int):
            raise TypeError("value must be an int")
        # Init
        self.__value = value

    def evaluate_result(self) -> Result:
        return Result(CounterArray.ones(1), self.__value)


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
        result_counter_array = result.counter_array
        result_starting_value = result.starting_value
        dice: List[Result] = []
        indices: List[int] = []
        # Create dice
        for index, count in enumerate(result_counter_array):
            if count == 0:
                continue
            indices.append(index)
            target = index + result_starting_value
            if target < 0:
                dice.append(Result(CounterArray.ones(-target), -target))
            elif target == 0:
                dice.append(Result.zero())
            else:
                dice.append(Result(CounterArray.ones(target), 1))
        # Fuse dice
        if len(dice) == 0:
            return Result.zero()
        if len(dice) == 1:
            return dice[0]
        # Need to equalize totals
        equalized_dice = Result.equalize_totals(dice)
        return Result.fuse_outcomes(
            [
                Result.scaled_counter_array(eq_die, result_counter_array[index])
                for eq_die, index in zip(equalized_dice, indices, strict=True)
            ],
            set_gdc_to_one=True,
        )


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
        # Setup
        no_of_expressions = len(self.__expressions)  # > 1 by construction
        results = [expr.evaluate_result() for expr in self.__expressions]
        equalized_results = Result.equalize_totals(results)

        counter_arrays: List[List[int]] = [
            result.counter_array for result in equalized_results
        ]
        starting_values: List[int] = [
            result.starting_value for result in equalized_results
        ]
        ending_values: List[int] = [
            result.starting_value + result.counter_array.size
            for result in equalized_results
        ]
        # Advantage
        advantage_starting_value = max(starting_values)
        advantage_counter_array = CounterArray.zeros(
            max(ending_values) - advantage_starting_value + 1
        )

        lists = [
            list(range(starting_values[i], ending_values[i] + 1))
            for i in range(no_of_expressions)
        ]
        combinations = cartesian_product(*lists)

        for combination in combinations:
            advantage_index = max(combination) - advantage_starting_value
            advantage_counter_array[advantage_index] += math.prod(
                [
                    counter_array[combination[i] - starting_values[i]]
                    for i, counter_array in enumerate(counter_arrays)
                ]
            )
        advantage_result = Result(advantage_counter_array, advantage_starting_value)
        advantage_result.set_gcd_to_one()
        return advantage_result


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
        # Setup
        no_of_expressions = len(self.__expressions)  # > 1 by construction
        results = [expr.evaluate_result() for expr in self.__expressions]
        equalized_results = Result.equalize_totals(results)

        counter_arrays = [result.counter_array for result in equalized_results]
        starting_values = [result.starting_value for result in equalized_results]
        ending_values = [
            result.starting_value + result.counter_array.size - 1
            for result in equalized_results
        ]
        # Disadvantage
        disadvantage_starting_value = min(starting_values)
        disadvantage_counter_array = CounterArray.zeros(
            min(ending_values) - disadvantage_starting_value + 1
        )

        indices_lists = [
            list(range(starting_values[i], ending_values[i] + 1))
            for i in range(no_of_expressions)
        ]
        combinations = cartesian_product(*indices_lists)

        for combination in combinations:
            disadvantage_index = min(combination) - disadvantage_starting_value
            disadvantage_counter_array[disadvantage_index] += math.prod(
                [
                    counter_array[combination[i] - starting_values[i]]
                    for i, counter_array in enumerate(counter_arrays)
                ]
            )
        disadvantage_result = Result(
            disadvantage_counter_array, disadvantage_starting_value
        )
        disadvantage_result.set_gcd_to_one()
        return disadvantage_result


class ResultExpression(Expression):
    def __init__(self, result: Result):
        if not isinstance(result, Result):
            raise ValueError("result must be an Result")
        self.__result = result

    def evaluate_result(self) -> Result:
        return self.__result


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


if __name__ == "__main__":
    d6 = DieExpression(ConstantExpression(6)).evaluate_result()
    print("d6:", d6)
    dd6 = DieExpression(ResultExpression(d6)).evaluate_result()
    print("dd6", dd6)
    # A dd6's starting value is 1
    assert dd6.starting_value == 1
    # PPCM/x[i] : int
    # x[i]/PGCD : int
