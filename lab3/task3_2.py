# Лаб 3.2 — Кубический сплайн
# Вариант 4
# Данные: x = [1.0, 1.9, 2.8, 3.7, 4.6], f = [2.4142, 1.0818, 0.50953, 0.11836, -0.24008]
# X* = 2.66666667
# Условие: S''(x_0) = 0, S''(x_n) = 0 (натуральный сплайн)

import math


def main():
    # Исходные данные (вариант 4)
    X = [1.0, 1.9, 2.8, 3.7, 4.6]
    F = [2.4142, 1.0818, 0.50953, 0.11836, -0.24008]
    x_star = 2.66666667
    n = len(X) - 1  # n = 4

    print("=" * 65)
    print("  Лабораторная работа 3.2  |  Вариант 4")
    print("  Кубический сплайн (натуральный: S''(x_0) = S''(x_n) = 0)")
    print(f"  X* = {x_star}")
    print("=" * 65)

    print(f"\n  Исходные данные:")
    print(f"  {'i':>4}  {'x_i':>10}  {'f_i':>10}")
    print(f"  {'-'*28}")
    for i in range(n + 1):
        print(f"  {i:>4}  {X[i]:>10.4f}  {F[i]:>10.5f}")

    # ── Шаги h_i = x_i - x_{i-1}, i = 1..n ────────────────────────
    h = [0.0]  # h[0] не используется
    for i in range(1, n + 1):
        h.append(X[i] - X[i - 1])

    print(f"\n  Шаги: h = {[round(h[i], 4) for i in range(1, n+1)]}")

    # ── Система уравнений (3.13) для c_2, ..., c_n ──────────────────
    # c_1 = 0  (S''(x_0) = 0 → c_1 = 0)
    # Система: трёхдиагональная для c_2, c_3, ..., c_n
    #
    # 2(h_1 + h_2)·c_2 + h_2·c_3 = 3[(f_2 - f_1)/h_2 - (f_1 - f_0)/h_1]
    # h_{i-1}·c_{i-1} + 2(h_{i-1} + h_i)·c_i + h_i·c_{i+1} = 3[Δf_i/h_i - Δf_{i-1}/h_{i-1}],  i=3,...,n-1
    # h_{n-1}·c_{n-1} + 2(h_{n-1} + h_n)·c_n = 3[(f_n - f_{n-1})/h_n - (f_{n-1} - f_{n-2})/h_{n-1}]

    # Размер системы = n - 1 (c_2, c_3, ..., c_n)
    m = n - 1  # число неизвестных: c_2, c_3, ..., c_n

    # Строим систему
    # Индексация: переменная j = 0..m-1 соответствует c_{j+2}
    A_diag = [0.0] * m    # главная диагональ
    A_lower = [0.0] * m   # нижняя диагональ
    A_upper = [0.0] * m   # верхняя диагональ
    rhs = [0.0] * m       # правая часть

    for j in range(m):
        i = j + 2  # индекс c_i
        A_diag[j] = 2 * (h[i - 1] + h[i])
        if j > 0:
            A_lower[j] = h[i - 1]
        if j < m - 1:
            A_upper[j] = h[i]
        rhs[j] = 3 * ((F[i] - F[i - 1]) / h[i] - (F[i - 1] - F[i - 2]) / h[i - 1])

    print(f"\n  Система уравнений (трёхдиагональная) для c_2, ..., c_{n}:")
    for j in range(m):
        i = j + 2
        terms = []
        if j > 0:
            terms.append(f"{A_lower[j]:.4f}·c_{i-1}")
        terms.append(f"{A_diag[j]:.4f}·c_{i}")
        if j < m - 1:
            terms.append(f"{A_upper[j]:.4f}·c_{i+1}")
        print(f"    {' + '.join(terms)} = {rhs[j]:.5f}")

    # ── Решение методом прогонки ────────────────────────────────────
    # Прямой ход
    P = [0.0] * m
    Q = [0.0] * m
    P[0] = -A_upper[0] / A_diag[0]
    Q[0] = rhs[0] / A_diag[0]
    for j in range(1, m):
        denom = A_diag[j] + A_lower[j] * P[j - 1]
        P[j] = -A_upper[j] / denom if j < m - 1 else 0.0
        Q[j] = (rhs[j] - A_lower[j] * Q[j - 1]) / denom

    # Обратный ход
    c_sol = [0.0] * m
    c_sol[m - 1] = Q[m - 1]
    for j in range(m - 2, -1, -1):
        c_sol[j] = P[j] * c_sol[j + 1] + Q[j]

    # Полный массив c: c_1 = 0, c_2..c_n из решения
    c = [0.0] * (n + 1)  # c[1]..c[n], c[0] не используется
    c[1] = 0.0  # натуральный сплайн
    for j in range(m):
        c[j + 2] = c_sol[j]

    print(f"\n  Коэффициенты c_i:")
    for i in range(1, n + 1):
        print(f"    c_{i} = {c[i]:.8f}")

    # ── Восстановление a, b, d по формулам (3.14) ───────────────────
    a = [0.0] * (n + 1)
    b = [0.0] * (n + 1)
    d = [0.0] * (n + 1)

    for i in range(1, n + 1):
        a[i] = F[i - 1]

    for i in range(1, n):
        b[i] = (F[i] - F[i - 1]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    # Для i = n (последний интервал): формулы (3.14) с c_{n+1} = 0
    # Но по условию натурального сплайна S''(x_n) = 0, т.е. c_n + 3·d_n·h_n = 0 → d_n = -c_n/(3·h_n)
    b[n] = (F[n] - F[n - 1]) / h[n] - (2.0 / 3.0) * h[n] * c[n]
    d[n] = -c[n] / (3 * h[n])

    print(f"\n  Коэффициенты сплайна S(x) = a_i + b_i·(x-x_{{i-1}}) + c_i·(x-x_{{i-1}})² + d_i·(x-x_{{i-1}})³:")
    print(f"  {'i':>4}  {'[x_{i-1}, x_i]':>14}  {'a_i':>12}  {'b_i':>12}  {'c_i':>12}  {'d_i':>12}")
    print(f"  {'-'*72}")
    for i in range(1, n + 1):
        interval = f"[{X[i-1]:.1f}, {X[i]:.1f}]"
        print(f"  {i:>4}  {interval:>14}  {a[i]:>12.5f}  {b[i]:>12.5f}  {c[i]:>12.5f}  {d[i]:>12.5f}")

    # ── Вычисление S(X*) ──────────────────────────────────────────
    # Найти интервал, в который попадает X*
    seg = -1
    for i in range(1, n + 1):
        if X[i - 1] <= x_star <= X[i]:
            seg = i
            break

    if seg == -1:
        print(f"\n  ОШИБКА: X* = {x_star} не принадлежит ни одному отрезку!")
        return

    dx = x_star - X[seg - 1]
    S_val = a[seg] + b[seg] * dx + c[seg] * dx**2 + d[seg] * dx**3

    print(f"\n  Точка X* = {x_star} принадлежит отрезку [{X[seg-1]}, {X[seg]}], i = {seg}")
    print(f"  S(X*) = {a[seg]:.5f} + {b[seg]:.5f}·({dx:.8f}) + {c[seg]:.5f}·({dx:.8f})² + {d[seg]:.5f}·({dx:.8f})³")
    print(f"  S(X*) = {S_val:.8f}")

    # ── Проверка: значения в узлах ────────────────────────────────
    print(f"\n  Проверка сплайна в узлах:")
    print(f"  {'i':>4}  {'x_i':>10}  {'f_i':>12}  {'S(x_i)':>12}  {'|f - S|':>12}")
    print(f"  {'-'*52}")
    for i in range(n + 1):
        if i == 0:
            s_seg = 1
            dx_check = X[0] - X[0]
        else:
            s_seg = i
            dx_check = X[i] - X[i - 1]
        s_val = a[s_seg] + b[s_seg] * dx_check + c[s_seg] * dx_check**2 + d[s_seg] * dx_check**3
        print(f"  {i:>4}  {X[i]:>10.4f}  {F[i]:>12.5f}  {s_val:>12.8f}  {abs(F[i] - s_val):>12.2e}")


if __name__ == "__main__":
    main()
