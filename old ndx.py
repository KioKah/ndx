import numpy as np
import math
import matplotlib.pyplot as plt
from statistics import mean
from statistics import stdev
from colorsys import hsv_to_rgb
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict


class ndxOld:
    def __init__(self, dices):
        self.dices = dices
        self.maxi = sum(dices)  # highest_value
        self.mini = len(dices)  # lowest_value
        self.l = len(dices)  # length
        self.L = self.maxi - self.mini + 1

        self.rolls = [0] * self.maxi
        for i in range(0, self.dices[0]):
            self.rolls[i] = 1

        for i in range(1, self.l):
            temp = self.rolls
            self.rolls = [0] * self.maxi
            for j in range(i, self.maxi):
                if temp[j - 1] != 0:
                    for k in range(0, self.dices[i]):
                        self.rolls[k + j] += temp[j - 1]
                else:
                    j = self.maxi

        self.somme = sum(self.rolls)

        self.ordo = [
            r + self.mini for r in range(self.L)
        ]  # résultats possibles (de mini à maxi)
        self.percents = [
            100 * self.rolls[i + self.mini - 1] / self.somme for i in range(self.L)
        ]  # probabilités de chacun des résultats possibles (de mini à maxi)

        self.moy = 100 / self.L
        self.high = max(self.percents)

        self.somme = (f"{self.somme:_}").replace("_", " ")  # "4586" → "4_586" → "4 586"

        self.plus = self.percents[:]
        for i in range(self.maxi - self.mini + 1):
            if self.percents[i] < self.moy:
                self.plus[i] = 0

        self.std = round(stdev(self.percents), 2)

    def show(self):
        plt.bar(self.ordo, self.percents, color="#4180FF")
        plt.bar(self.ordo, self.plus, color="#0054FF")
        plt.bar(
            math.floor((self.maxi + len(self.dices)) / 2),
            self.percents[math.floor((self.maxi + len(self.dices)) / 2) - self.l],
            color="#003AB0",
        )
        plt.bar(
            math.ceil((self.maxi + len(self.dices)) / 2),
            self.percents[math.ceil((self.maxi + len(self.dices)) / 2) - self.l],
            color="#003AB0",
        )


axes = plt.gca()

dices = [20, 20]

maxi = sum(dices)  # highest_value
mini = len(dices)  # lowest_value
l = len(dices)  # length
L = maxi - mini + 1

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

ordo = [r + mini for r in range(L)]  # résultats possibles (de mini à maxi)
percents = [
    100 * rolls[i + mini - 1] / somme for i in range(L)
]  # probabilités de chacun des résultats possibles (de mini à maxi)

moy = 100 / L
high = max(percents)

somme = (f"{somme:_}").replace("_", " ")  # "4586" → "4_586" → "4 586"

plus = percents[:]
for i in range(maxi - mini + 1):
    if percents[i] < moy:
        plus[i] = 0

std = round(stdev(percents), 2)

###


plt.bar(ordo, percents, color="#4180FF")
plt.bar(ordo, plus, color="#0054FF")
plt.bar(
    math.floor((maxi + len(dices)) / 2),
    percents[math.floor((maxi + len(dices)) / 2) - l],
    color="#003AB0",
)
plt.bar(
    math.ceil((maxi + len(dices)) / 2),
    percents[math.ceil((maxi + len(dices)) / 2) - l],
    color="#003AB0",
)
plt.plot(
    [mini - 0.5, maxi + 0.5], [moy, moy], "#FF0000", label="Moyenne", linewidth=0.5
)
plt.title(
    "Dés : "
    + str(dices).replace(", ", "][")
    + "\n"
    + somme
    + " possibilités / STD "
    + str(std)
    + " / RSD "
    + str(100 * std / moy)
)
plt.ylabel("Probabilité (%)")
plt.xlabel("Résultat")
plt.legend()
plt.show()


###


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
    matplotlib.rcParams.update({"font.size": 1200 / dpi})

    R = []
    for n in range(len(D)):
        r = []
        dices = D[n]

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
            + hexi(rgbcolor[0] * 255)[2:].zfill(2)
            + hexi(rgbcolor[1] * 255)[2:].zfill(2)
            + hexi(rgbcolor[2] * 255)[2:].zfill(2)
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

    la = max([lens(r[0][0]) + lens(len(r[0])) for r in R]) + 2
    lb = 13 + lens(R[-1][3]) + lens(R[-1][1]) + lens(R[0][2])

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


def lens(x):
    return len(str(x))


def hexi(i):
    return hex(int(i))


def decal(l, x):
    return [l[i] + x for i in range(len(l))]
