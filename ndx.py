"""
    ndx stands for n x-sided dice
    # * 15d6 confirmed by numpy convolutions
"""

from abc import ABC, abstractmethod

from src.expressions import (
    ConstantExpression,
    DieExpression,
    SumOfExpressions,
    SubtractionOfExpressions,
    ProductOfExpressions,
    DivisionOfExpressions,
    ExpressionRepetitionOfExpression,
    AdvantageOfExpressions,
    DisadvantageOfExpressions,
    WeightedExpressions,
    ResultExpression,
)
from src.result import Result

import matplotlib.pyplot as plt


class Plot(ABC):
    def __init__(
        self,
        result: Result,
    ):
        # Checks
        if not isinstance(result, Result):
            raise TypeError("result must be a Result")
        # Init
        self.result = result
        self.plot_result()

    @abstractmethod
    def plot_result(self):
        """Plot the result"""


class MainPlot(Plot):
    MAX_COLOR = "#003AB0"
    HIGH_COLOR = "#0054FF"
    BASE_COLOR = "#4180FF"

    def __init__(
        self,
        result: Result,
        title: str,
        plot_uniform_distribution: bool = True,
        signed: bool = False,
    ):
        self.title = title
        self.plot_uniform_distribution = plot_uniform_distribution
        self.signed = signed
        super().__init__(result)

    def signed_probabilities(self):
        x_axis = range(
            self.result.starting_value,
            self.result.starting_value + self.result.count_array.size,
        )
        values = list(self.result.probabilities.values())

        signed_probabilities = []
        for i, x in enumerate(x_axis):
            if x == 0:
                signed_probabilities.append(0)
            else:
                signed_probabilities.append(100 * values[i] * (2 * (x > 0) - 1))
        return signed_probabilities

    def plot_result(self):
        start = self.result.starting_value
        size = self.result.count_array.size
        end = start + size - 1
        x_axis = range(start, end + 1)

        if self.signed:
            probabilities = self.signed_probabilities()
        else:
            probabilities = [p * 100 for p in self.result.probabilities.values()]
        uniform_probability = 100 / size
        max_probability = max(probabilities)

        colors = []
        for prob in probabilities:
            p = abs(prob)
            if p == max_probability:
                colors.append(self.MAX_COLOR)
            elif p > uniform_probability:
                colors.append(self.HIGH_COLOR)
            else:
                colors.append(self.BASE_COLOR)

        plt.bar(x_axis, probabilities, color=colors)
        if self.plot_uniform_distribution:
            plt.plot(
                [start - 0.5, end + 0.5],
                [uniform_probability, uniform_probability],
                "#FF0000",
                label="Distribution uniforme",
                linewidth=0.5,
            )
        mean = self.result.mean()
        sign = "+" if mean > 0 else ("-" if mean < 0 else "±")
        plt.title(f"{self.title}\nMoyenne : {sign*self.signed}{mean:.2f}")
        # "/ STDev {self.result.stdev():.2f}"
        plt.xlabel("Résultat")
        plt.ylabel(f"Probabilité {'signée'*self.signed} (%)")
        # Make horizontal axis visible
        if self.signed:
            plt.axhline(0, color="black", linewidth=0.5)
        if self.plot_uniform_distribution:
            plt.legend()
        plt.show()


class SubPlot(Plot):
    def __init__(self, result: Result, legend: str, plot_color: str):
        self.legend = legend
        self.plot_color = plot_color
        super().__init__(result)

    def plot_result(self):
        start = self.result.starting_value
        size = self.result.count_array.size
        end = start + size - 1
        x_axis = range(start, end + 1)

        probabilities = [p * 100 for p in self.result.probabilities.values()]

        plt.plot(x_axis, probabilities, color=self.plot_color, label=self.legend)


if __name__ == "__main__":
    pass
    # e_2d6 = ExpressionRepetitionOfExpression(
    #     ConstantExpression(2), DieExpression(ConstantExpression(6))
    # )
    # r_2d6 = e_2d6.evaluate_result()
    # SubPlot(r_2d6, "2d6", "red")

    # e_3d4 = ExpressionRepetitionOfExpression(
    #     ConstantExpression(3), DieExpression(ConstantExpression(4))
    # )
    # r_3d4 = e_3d4.evaluate_result()
    # SubPlot(r_3d4, "3d4", "green")

    # e_adv_2d6_3d4 = AdvantageOfExpressions([e_2d6, e_3d4])
    # r_adv_2d6_3d4 = e_adv_2d6_3d4.evaluate_result()
    # MainPlot(r_adv_2d6_3d4, title="Avantage : 2d6 ou 3d4 (garde le meilleur)")

    # r = ExpressionRepetitionOfExpression(
    #     ConstantExpression(442), DieExpression(ConstantExpression(2))
    # ).evaluate_result()
    # MainPlot(r, title="442d2")

    # r_d42_d42 = ExpressionRepetitionOfExpression(
    #     DieExpression(ConstantExpression(42)), DieExpression(ConstantExpression(42))
    # ).evaluate_result()
    # MainPlot(r_d42_d42, title="d42 lancers de d42")

    # e = SubtractionOfExpressions(
    #     ExpressionRepetitionOfExpression(
    #         ConstantExpression(2),
    #         DieExpression(ConstantExpression(6)),
    #     ),
    #     [DieExpression(ConstantExpression(10))],
    # )
    # r = e.evaluate_result()
    # MainPlot(r, title="2d6 - d10", signed=True, plot_uniform_distribution=False)
