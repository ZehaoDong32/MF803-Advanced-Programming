import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def bootstrapping(swap):
    forward = {}
    D = 1
    floating2 = 0
    sum_D = 0
    year = 0
    for key, c in swap.items():
        def f(x):
            i = 0.5
            fixed = c / 2 * sum_D
            floating = floating2
            while i <= key - year:
                fixed += c / 2 * np.exp(-x * i) * D
                floating += (np.exp(x / 2) - 1) * np.exp(-x * i) * D
                i += 0.5
            return fixed - floating

        f_last = fsolve(f, 0.03)[0]
        forward[key] = f_last
        j = 0.5
        while j <= key - year:
            D = D * np.exp(-f_last * j)
            sum_D += D
            floating2 += (np.exp(f_last / 2) - 1) * D
            j += 0.5
        year = key
    return forward


def plot(a, b, label1, label2):
    lists1 = sorted(a.items())
    lists2 = sorted(b.items())
    x1, y1 = zip(*lists1)
    x2, y2 = zip(*lists2)
    plt1, = plt.plot(x1, y1, label=label1)
    plt2, = plt.plot(x2, y2, label=label2)
    plt.legend(handles=[plt1, plt2])
    plt.show()


def discount_factor(forward):
    D = {}
    n = 1
    D[0.5] = np.exp(-forward[1] / 2)
    for key in forward:
        while n <= 30:
            if n <= key:
                D[n] = D[n - 0.5] * np.exp(-forward[key] / 2)
                n += 0.5
            else:
                break
    return D


def breakeven_swap(forward, D):
    swap = {}
    n = 0.5
    d = 0
    forward_before = 0
    for key in forward:
        while n <= 30:
            if n <= key:
                d += D[n]
                swap[n] = (forward_before + (np.exp(forward[key]) - 1) / 2 * D[n]) * 2 / d
                forward_before += (np.exp(forward[key] / 2) - 1) * D[n]
                n += 0.5

            else:
                break
    return swap


swap = {1: 2.8438 * 0.01, 2: 3.060 * 0.01, 3: 3.126 * 0.01, 4: 3.144 * 0.01, 5: 3.150 * 0.01, 7: 3.169 * 0.01,
        10: 3.210 * 0.01, 30: 3.237 * 0.01}

# (a)(b)(c)
forward = bootstrapping(swap)
print ('The constant forward rate for the entire curve:', forward)

plot(forward, swap, 'forward', 'swap')

# (d)
D = discount_factor(forward)
print('The breakeven swap rate of a 15Y swap is:', breakeven_swap(forward, D)[15])

# (e)
print('The discount factors are:', D)

zero_rate = {}
for key in swap:
    zero_rate[key] = -np.log(D[key]) / key
print('The zero rates are:', zero_rate)

plot(zero_rate, swap, 'zero_rate', 'swap')

# (f)
forward2 = {}
for key in forward:
    forward2[key] = forward[key] + 0.01
breakeven = breakeven_swap(forward2, D)
breakeven_swap_rate = {}
for key in breakeven:
    for key2 in forward2:
        if key == key2:
            breakeven_swap_rate[key] = breakeven[key]
print('The breakeven swap rates are:', breakeven_swap_rate)
swap2 = {}
for key in swap:
    swap2[key] = swap[key] + 0.01
plot(breakeven_swap_rate, swap2, 'breakeven_swap', 'swap2')

# (g)
swap3 = {}
swap3[1] = swap[1]
swap3[2] = swap[2]
swap3[3] = swap[3]
swap3[4] = swap[4] + 0.05 * 0.01
swap3[5] = swap[5] + 0.1 * 0.01
swap3[7] = swap[7] + 0.15 * 0.01
swap3[10] = swap[10] + 0.25 * 0.01
swap3[30] = swap[30] + 0.5 * 0.01
print('The new swap rates are:', swap3)

# (h)
forward3 = bootstrapping(swap3)
print('The new forward rates are:', forward3)
plot(forward, forward3, 'forward', 'forward3')

# (i)
swap4 = {}
swap4[1] = swap[1] - 0.5 * 0.01
swap4[2] = swap[2] - 0.25 * 0.01
swap4[3] = swap[3] - 0.15 * 0.01
swap4[4] = swap[4] - 0.10 * 0.01
swap4[5] = swap[5] - 0.05 * 0.01
swap4[7] = swap[7]
swap4[10] = swap[10]
swap4[30] = swap[30]
print('The new swap rates are:', swap4)

# (j)
forward4 = bootstrapping(swap4)
print('The new forward rates are:', forward4)
plot(forward, forward4, 'forward', 'forward4')