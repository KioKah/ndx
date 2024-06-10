"""
    ndx stands for n x-sided dice
    # * 15d6 confirmed by numpy convolutions
"""

from timeit import timeit
from src.expressions import (
    ConstantExpression,
    DieExpression,
    SumOfExpressions,
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

# from colorsys import hsv_to_rgb


def test_dX_ddX_dddX(X: int, print_result: bool = False, plot_result: bool = False):
    expr_dx = DieExpression(ConstantExpression(X))
    dx = expr_dx.evaluate_result()
    if print_result:
        print(f"d{X}: {dx}")
        print(dx.mean(), dx.stdev())
    if plot_result:
        dx.plot(str(expr_dx))

    expr_ddx = DieExpression(expr_dx)
    ddx = expr_ddx.evaluate_result()
    if print_result:
        print(f"dd{X}: {ddx}")
        print(ddx.mean(), ddx.stdev())
    if plot_result:
        ddx.plot(str(expr_ddx))

    expr_dddx = DieExpression(expr_ddx)
    dddx = expr_dddx.evaluate_result()
    if print_result:
        print(f"ddd{X}: {dddx}")
        print(dddx.mean(), dddx.stdev())
    if plot_result:
        dddx.plot(str(expr_dddx))


def test_NdX_plus_MdY(
    N: int,
    X: int,
    M: int,
    Y: int,
    print_result: bool = False,
    plot_result: bool = False,
):
    expr_ndx_plus_mdx = SumOfExpressions(
        [
            ExpressionRepetitionOfExpression(
                ConstantExpression(N), DieExpression(ConstantExpression(X))
            ),
            ExpressionRepetitionOfExpression(
                ConstantExpression(M), DieExpression(ConstantExpression(Y))
            ),
        ]
    )
    ndx_plus_mdx = expr_ndx_plus_mdx.evaluate_result()
    if print_result:
        print(f"{N}d{X} + {M}d{Y}: {ndx_plus_mdx}")
        print(ndx_plus_mdx.mean(), ndx_plus_mdx.stdev())
    if plot_result:
        ndx_plus_mdx.plot(str(expr_ndx_plus_mdx))


def test_dX_times_dY(
    X: int, Y: int, print_result: bool = False, plot_result: bool = False
):
    expr_dx_times_dy = ProductOfExpressions(
        [DieExpression(ConstantExpression(X)), DieExpression(ConstantExpression(Y))]
    )
    dx_times_dy = expr_dx_times_dy.evaluate_result()
    if print_result:
        print(f"d{X} * d{Y}: {dx_times_dy}")
        print(dx_times_dy.mean(), dx_times_dy.stdev())
    if plot_result:
        dx_times_dy.plot(str(expr_dx_times_dy))


def test_dX_divided_by_dY(
    X: int, Y: int, floor: bool, print_result: bool = False, plot_result: bool = False
):
    expr_dx_divided_by_dy = DivisionOfExpressions(
        DieExpression(ConstantExpression(X)),
        [DieExpression(ConstantExpression(Y))],
        floor=floor,
    )
    dx_divided_by_dy = expr_dx_divided_by_dy.evaluate_result()
    if print_result:
        print(f"d{X} // d{Y}: {dx_divided_by_dy}")
        print(dx_divided_by_dy.mean(), dx_divided_by_dy.stdev())
    if plot_result:
        dx_divided_by_dy.plot(str(expr_dx_divided_by_dy))


def test_NdX(N: int, X: int, print_result: bool = False, plot_result: bool = False):
    expr_ndx = ExpressionRepetitionOfExpression(
        ConstantExpression(N), DieExpression(ConstantExpression(X))
    )
    ndx = expr_ndx.evaluate_result()
    if print_result:
        print(f"{N}d{X}: {ndx}")
        print(ndx.mean(), ndx.stdev())
    if plot_result:
        ndx.plot(str(expr_ndx))


def test_advantage_dX_dX(X: int, print_result: bool = False, plot_result: bool = False):
    expr_adv = AdvantageOfExpressions(
        [DieExpression(ConstantExpression(X)), DieExpression(ConstantExpression(X))]
    )
    adv = expr_adv.evaluate_result()
    if print_result:
        print(f"A(d{X}, d{X}): {adv}")
        print(adv.mean(), adv.stdev())
    if plot_result:
        adv.plot(str(expr_adv))


def test_custom_w1_dX_w2_ddX(
    X: int, w1: int, w2: int, print_result: bool = False, plot_result: bool = False
):
    expr_custom = WeightedExpressions(
        {
            DieExpression(ConstantExpression(X)): w1,
            DieExpression(DieExpression(ConstantExpression(X))): w2,
        }
    )
    custom = expr_custom.evaluate_result()
    if print_result:
        print(f"{{d{X}: {w1}, dd{X}: {w2}}}: {custom}")
        print(custom.mean(), custom.stdev())
    if plot_result:
        custom.plot(str(expr_custom))


if __name__ == "__main__":
    # LCM//x[i] : int
    # x[i]//GCD : int
    PRINT_RESULT = True
    PLOT_RESULT = False
    NUMBER = 100

    test_dX_ddX_dddX(X=6, print_result=PRINT_RESULT, plot_result=PLOT_RESULT)
    execution_time = timeit("test_dX_ddX_dddX(6)", globals=globals(), number=NUMBER)

    print(f"\n{execution_time/NUMBER*1000:.3f} ms\n\n")

    test_NdX(N=6, X=6, print_result=PRINT_RESULT, plot_result=PLOT_RESULT)
    execution_time = timeit("test_NdX(6, 6)", globals=globals(), number=NUMBER)

    print(f"\n{execution_time/NUMBER*1000:.3f} ms\n\n")

    test_NdX_plus_MdY(
        N=1, X=8, M=2, Y=4, print_result=PRINT_RESULT, plot_result=PLOT_RESULT
    )
    execution_time = timeit(
        "test_NdX_plus_MdY(1, 8, 2, 4)", globals=globals(), number=NUMBER
    )

    print(f"\n{execution_time/NUMBER*1000:.3f} ms\n\n")

    test_dX_times_dY(X=6, Y=4, print_result=PRINT_RESULT, plot_result=PLOT_RESULT)
    execution_time = timeit("test_dX_times_dY(6, 4)", globals=globals(), number=NUMBER)

    print(f"\n{execution_time/NUMBER*1000:.3f} ms\n\n")

    test_dX_divided_by_dY(
        X=6, Y=2, floor=True, print_result=PRINT_RESULT, plot_result=True
    )
    execution_time = timeit(
        "test_dX_divided_by_dY(6, 2, True)", globals=globals(), number=NUMBER
    )

    print(f"\n{execution_time/NUMBER*1000:.3f} ms\n\n")

    test_advantage_dX_dX(X=6, print_result=PRINT_RESULT, plot_result=PLOT_RESULT)
    execution_time = timeit("test_advantage_dX_dX(6)", globals=globals(), number=NUMBER)

    print(f"\n{execution_time/NUMBER*1000:.3f} ms\n\n")

    test_custom_w1_dX_w2_ddX(
        X=6, w1=2, w2=3, print_result=PRINT_RESULT, plot_result=PLOT_RESULT
    )
    execution_time = timeit(
        "test_custom_w1_dX_w2_ddX(6, 2, 3)", globals=globals(), number=NUMBER
    )

    print(f"\n{execution_time/NUMBER*1000:.3f} ms\n\n")
