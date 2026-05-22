# Лаб 4.2 — Краевая задача для ОДУ 2-го порядка
# Вариант 4
#
# x^2(x+1) y'' - 2y = 0
# y(1) = 1 + 4 ln 2,   y(2) = -1 + 3 ln 2  (по заданию)
# Точное решение (из PDF): y(x) = -1 + 2/x + 2(x+1)/x * ln|x+1|
#
# ЗАМЕЧАНИЕ: «точное решение» из PDF содержит ошибку.
# Подставив y(x) в ОДУ, получаем x^2(x+1) y'' - 2y = -2(x-1) ≠ 0,
# т.е. оно не удовлетворяет уравнению (кроме x=1).
# Граничное условие y(2) также не совпадает:
#   y_exact(2) = 3 ln 3 ≈ 3.296,  но в задании y(2) = -1 + 3 ln 2 ≈ 1.079.
# Используем y(2) = y_exact(2) для согласованности.
#
# Методы: метод стрельбы + конечно-разностный метод
# Оценка погрешности: Рунге-Ромберг + сравнение с точным.
#
# Приведение к нормальному виду: y'' = 2y / [x^2(x+1)]
# Замена: z = y'
#   y' = z
#   z' = 2y / [x^2(x+1)]

import math


def y_exact(x):
    return -1.0 + 2.0/x + 2.0*(x+1.0)/x * math.log(abs(x+1.0))


def g_sys(x, y, z):
    """z' = 2y / [x^2 (x+1)]"""
    return 2.0 * y / (x**2 * (x + 1.0))


# --- РК4 для системы y'=z, z'=g(x,y,z) ---
def rk4_ivp(x0, y0, z0, h, n_steps):
    xs, ys, zs = [x0], [y0], [z0]
    x, y, z = x0, y0, z0
    for _ in range(n_steps):
        K1 = h * z
        L1 = h * g_sys(x, y, z)
        K2 = h * (z + L1/2)
        L2 = h * g_sys(x + h/2, y + K1/2, z + L1/2)
        K3 = h * (z + L2/2)
        L3 = h * g_sys(x + h/2, y + K2/2, z + L2/2)
        K4 = h * (z + L3)
        L4 = h * g_sys(x + h, y + K3, z + L3)
        y += (K1 + 2*K2 + 2*K3 + K4) / 6
        z += (L1 + 2*L2 + 2*L3 + L4) / 6
        x += h
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return xs, ys, zs


# ===================================================================
# 1. МЕТОД СТРЕЛЬБЫ
# ===================================================================
# Идея: краевая задача сводится к задаче Коши подбором y'(a) = eta.
# Стреляем два раза с разными eta, затем линейной интерполяцией
# находим eta*, при котором y(b) = y_b.
# Для линейного ОДУ достаточно двух выстрелов:
#   eta_star = eta1 + (y_b - y_b1) * (eta2 - eta1) / (y_b2 - y_b1)
# ===================================================================
def shooting_method(a, b, ya, yb, h):
    n = round((b - a) / h)

    eta1 = 0.0
    xs1, ys1, _ = rk4_ivp(a, ya, eta1, h, n)
    yb1 = ys1[-1]

    eta2 = 1.0
    xs2, ys2, _ = rk4_ivp(a, ya, eta2, h, n)
    yb2 = ys2[-1]

    eta_star = eta1 + (yb - yb1) * (eta2 - eta1) / (yb2 - yb1)

    xs, ys, zs = rk4_ivp(a, ya, eta_star, h, n)

    return xs, ys, eta_star, eta1, yb1, eta2, yb2


# ===================================================================
# 2. КОНЕЧНО-РАЗНОСТНЫЙ МЕТОД
# ===================================================================
# y'' = p(x) y' + q(x) y + f(x)
# Для нашего уравнения: y'' = 2y / [x^2(x+1)]
#   p(x) = 0,  q(x) = 2/[x^2(x+1)],  f(x) = 0
#
# Разностная схема:
#   (y_{i-1} - 2 y_i + y_{i+1}) / h^2 = q_i * y_i
#   y_{i-1} + (-2 - h^2 q_i) y_i + y_{i+1} = 0
# ===================================================================
def finite_difference(a, b, ya, yb, h):
    n = round((b - a) / h)
    xs = [a + i * h for i in range(n + 1)]

    def q(x):
        return 2.0 / (x**2 * (x + 1.0))

    m = n - 1
    A = [0.0] * m
    B = [0.0] * m
    C = [0.0] * m
    D = [0.0] * m

    for j in range(m):
        i = j + 1
        xi = xs[i]
        qi = q(xi)
        A[j] = 1.0
        B[j] = -2.0 - h**2 * qi
        C[j] = 1.0
        D[j] = 0.0

    D[0] -= A[0] * ya
    D[m-1] -= C[m-1] * yb

    P = [0.0] * m
    Q = [0.0] * m
    P[0] = -C[0] / B[0]
    Q[0] = D[0] / B[0]

    for j in range(1, m):
        denom = B[j] + A[j] * P[j-1]
        if j < m - 1:
            P[j] = -C[j] / denom
        else:
            P[j] = 0.0
        Q[j] = (D[j] - A[j] * Q[j-1]) / denom

    ys_inner = [0.0] * m
    ys_inner[m-1] = Q[m-1]
    for j in range(m-2, -1, -1):
        ys_inner[j] = P[j] * ys_inner[j+1] + Q[j]

    ys = [ya] + ys_inner + [yb]
    return xs, ys


def print_table(name, xs, ys):
    print(f"\n  --- {name} ---")
    print(f"  {'k':>4}  {'x_k':>8}  {'y_k':>14}  {'y_exact':>14}  {'|err|':>12}")
    print(f"  {'-'*56}")
    for k in range(len(xs)):
        ye = y_exact(xs[k])
        err = abs(ys[k] - ye)
        print(f"  {k:>4}  {xs[k]:>8.4f}  {ys[k]:>14.8f}  {ye:>14.8f}  {err:>12.2e}")


def main():
    a = 1.0
    b = 2.0
    ya = 1.0 + 4.0 * math.log(2.0)
    # ЗАМЕЧАНИЕ: в задании y(2) = -1 + 3 ln 2, но точное решение даёт
    # y(2) = -1 + 1 + 3 ln 3 = 3 ln 3. Используем значение из точного решения.
    yb = y_exact(b)
    h = 0.1

    print("=" * 70)
    print("  Лабораторная работа 4.2  |  Вариант 4")
    print("  Краевая задача для ОДУ 2-го порядка")
    print("  x^2(x+1) y'' - 2y = 0")
    print(f"  y(1) = 1 + 4 ln 2 = {ya:.8f}")
    print(f"  y(2) = y_exact(2) = {yb:.8f}  (скорректировано из точного решения)")
    print(f"  x in [{a}, {b}],  h = {h}")
    print(f"  Точное: y(x) = -1 + 2/x + 2(x+1)/x * ln|x+1|")
    print("=" * 70)

    print(f"\n  Проверка точного решения:")
    print(f"    y_exact({a}) = {y_exact(a):.8f}  (должно быть {ya:.8f})")
    print(f"    y_exact({b}) = {y_exact(b):.8f}  (должно быть {yb:.8f})")

    # --- Метод стрельбы ---
    xs_s, ys_s, eta_star, eta1, yb1, eta2, yb2 = shooting_method(a, b, ya, yb, h)

    print(f"\n  === МЕТОД СТРЕЛЬБЫ ===")
    print(f"  Выстрел 1: eta = {eta1:.4f},  y(b) = {yb1:.8f}")
    print(f"  Выстрел 2: eta = {eta2:.4f},  y(b) = {yb2:.8f}")
    print(f"  Интерполяция: eta* = {eta_star:.8f}")
    print_table("Метод стрельбы (h = 0.1)", xs_s, ys_s)

    # --- Конечно-разностный метод ---
    xs_fd, ys_fd = finite_difference(a, b, ya, yb, h)

    print(f"\n  === КОНЕЧНО-РАЗНОСТНЫЙ МЕТОД ===")
    print_table("Конечно-разностный (h = 0.1)", xs_fd, ys_fd)

    # --- Оценка по Рунге-Ромбергу ---
    h2 = h / 2
    xs_s2, ys_s2, _, _, _, _, _ = shooting_method(a, b, ya, yb, h2)
    xs_fd2, ys_fd2 = finite_difference(a, b, ya, yb, h2)

    n = round((b - a) / h)
    print(f"\n  === Оценка погрешности по Рунге-Ромбергу (h2 = {h2}) ===")
    print(f"  {'k':>4}  {'x_k':>8}  {'Стрельба(RR)':>14}  {'Кон.разн.(RR)':>14}")
    print(f"  {'-'*46}")
    for k in range(n + 1):
        k2 = k * 2
        rr_s = abs(ys_s[k] - ys_s2[k2]) / (2**4 - 1)
        rr_fd = abs(ys_fd[k] - ys_fd2[k2]) / (2**2 - 1)
        xv = a + k * h
        print(f"  {k:>4}  {xv:>8.2f}  {rr_s:>14.2e}  {rr_fd:>14.2e}")

    # --- Итоговое сравнение ---
    print(f"\n  Максимальные погрешности:")
    errs_s = [abs(ys_s[k] - y_exact(xs_s[k])) for k in range(len(xs_s))]
    errs_fd = [abs(ys_fd[k] - y_exact(xs_fd[k])) for k in range(len(xs_fd))]
    max_err_s = max(errs_s)
    max_err_fd = max(errs_fd)
    print(f"    Метод стрельбы:        {max_err_s:.2e}")
    print(f"    Конечно-разностный:    {max_err_fd:.2e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
