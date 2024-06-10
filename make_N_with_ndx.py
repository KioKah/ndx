"""
    Does some obscure stuff with dice rolls :)
    (Plots the probability distribution of all ndx such that n*x = N)
"""

from statistics import stdev
from colorsys import hsv_to_rgb
from matplotlib import rcParams
import matplotlib.pyplot as plt


def options(N):
    D = []  # liste des dés identiques tq leur maximum vaut N
    for n in range(1, N):
        if (N % n) == 0:
            D.append([N // n] * n)

    div = N * (len(D) + 1)

    scale = 0.8
    dpi = scale * 9000 / div  # resolution (%)
    owi = scale * 0.8 * 1920 / dpi  # original width (in inches) (18.00 < 19.80)
    ppb = (owi * dpi) // div  # pixel per bar
    ipb = ppb / dpi  # inches per bar
    new_width = ipb * div / 0.8 / scale

    fig, ax = plt.subplots(1, 1, figsize=(new_width, 900 / dpi), dpi=dpi)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    rcParams.update({"font.size": 1200 / dpi})

    R = []
    for n, dices in enumerate(D):
        r = []

        maxi = sum(dices)  # highest_value
        mini = len(dices)  # lowest_value
        l = len(dices)  # number of dices
        L = maxi - mini + 1  # number of values

        rolls = [0] * maxi
        for i in range(0, dices[0]):
            rolls[i] = 1

        for i in range(1, l):
            temp = rolls
            rolls = [0] * maxi
            for j in range(i, maxi):
                if temp[j - 1] != 0:
                    for k in range(0, dices[i]):
                        rolls[k + j] += temp[j - 1]
                else:
                    j = maxi

        somme = sum(rolls)

        ordo = [r + mini for r in range(L)]
        percents = [100 * rolls[i + mini - 1] / somme for i in range(L)]

        expected = (N + l) / 2

        mean = 100 / L
        std = round(stdev(percents), 2)

        r = (dices, mini, maxi, expected, 100 * std / mean)
        R.append(r)

        rgbcolor = hsv_to_rgb(n / len(D), 1, 1)
        hexcolor = (
            "#"
            + hex_int(rgbcolor[0] * 255)[2:].zfill(2)
            + hex_int(rgbcolor[1] * 255)[2:].zfill(2)
            + hex_int(rgbcolor[2] * 255)[2:].zfill(2)
        )

        lab = str(len(dices)) + " d" + str(dices[0])
        dx = ((n + 0.5) / (1 + len(D)) - 0.5) / 2
        line_width = (len(D) + 1) * ipb * dpi

        plt.plot(
            decal(ordo, dx),
            percents,
            linewidth=line_width,
            label=lab,
            color=hexcolor,
            zorder=1,
        )
        plt.bar(decal(ordo, dx), percents, width=ipb, color=hexcolor, zorder=0)

    for n in range(1, N + 1):
        plt.plot([n - dx, n + dx], [0, 0], linewidth=line_width * 2, color="#000000")

    plt.ylabel("Probabilité (%)")
    plt.xlabel("Résultat")
    plt.xlim([0.5, N + 0.5])
    plt.ylim(bottom=0)
    ax.spines["left"].set_linewidth(line_width / 2)
    ax.spines["bottom"].set_linewidth(line_width / 2)
    plt.legend()
    # plt.savefig(f'{N}.png')
    plt.show()
    return R


def visu(N):
    R = options(N)

    la = max([len_str(r[0][0]) + len_str(len(r[0])) for r in R]) + 2
    lb = 13 + len_str(R[-1][3]) + len_str(R[-1][1]) + len_str(R[0][2])

    for r in R:
        D, mini, maxi, expected, rsd = r
        a = str(len(D)) + " d" + str(D[0])
        b = "Expected " + str(expected) + " (" + str(mini) + "-" + str(maxi) + ")"
        c = "Rsd " + str(round(rsd, 1)) + "%"
        print(pad(a, la), pad(b, lb), c, sep=" / ")
    return


def pad(strng, n):
    while len(strng) < n:
        strng = strng + " "
    return strng


def len_str(x):
    return len(str(x))


def hex_int(i: float):
    return hex(int(i))


def decal(l, x):
    return [l[i] + x for i in range(len(l))]
