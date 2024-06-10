from ndx.count_array import CountArray
from ndx.result import Result

from abc import ABC, abstractmethod
from typing import List, Dict


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
