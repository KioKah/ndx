"""
    ndx stands for n x-sided dice
    # * 15d6 confirmed by numpy convolutions
"""

from ndx.expressions import (
    DieExpression,
    ResultExpression,
    RepeatedExpression,
    AdvantageExpression,
    DisadvantageExpression,
    CustomExpression,
    ConstantExpression,
)
from ndx.result import Result

import matplotlib.pyplot as plt

# from colorsys import hsv_to_rgb


def test_dX_ddX_dddX(X: int):
    expr_dx = DieExpression(ConstantExpression(X))
    dx = expr_dx.evaluate_result()
    print(f"d{X}: {dx}")
    print(dx.mean(), dx.stdev())
    dx.plot(str(expr_dx))

    expr_ddx = DieExpression(expr_dx)
    ddx = expr_ddx.evaluate_result()
    print(f"dd{X}: {ddx}")
    print(ddx.mean(), ddx.stdev())
    ddx.plot(str(expr_ddx))

    expr_dddx = DieExpression(expr_ddx)
    dddx = expr_dddx.evaluate_result()
    print(f"ddd{X}: {dddx}")
    print(dddx.mean(), dddx.stdev())
    dddx.plot(str(expr_dddx))


def test_NdX(N: int, X: int):
    expr_ndx = RepeatedExpression(
        ConstantExpression(N), DieExpression(ConstantExpression(X))
    )
    ndx = expr_ndx.evaluate_result()
    print(f"{N}d{X}: {ndx}")
    print(ndx.mean(), ndx.stdev())
    ndx.plot(str(expr_ndx))


def test_advantage_dX_dX(X: int):
    expr_adv = AdvantageExpression(
        [DieExpression(ConstantExpression(X)), DieExpression(ConstantExpression(X))]
    )
    adv = expr_adv.evaluate_result()
    print(f"A(d{X}, d{X}): {adv}")
    print(adv.mean(), adv.stdev())
    adv.plot(str(expr_adv))


def test_custom_w1_dX_w2_ddX(X: int, w1: int, w2: int):
    expr_custom = CustomExpression(
        {
            ResultExpression(DieExpression(ConstantExpression(X))): w1,
            ResultExpression(DieExpression(DieExpression(ConstantExpression(X)))): w2,
        }
    )
    custom = expr_custom.evaluate_result()
    print(f"{{d{X}: {w1}, dd{X}: {w2}}}")
    print(custom.mean(), custom.stdev())
    custom.plot(str(expr_custom))


if __name__ == "__main__":
    test_dX_ddX_dddX(X=6)

    print("\n\n")

    test_NdX(N=15, X=6)

    print("\n\n")

    test_advantage_dX_dX(X=6)

    print("\n\n")

    test_custom_w1_dX_w2_ddX(X=6, w1=2, w2=3)
