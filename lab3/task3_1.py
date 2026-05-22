# Лаб 3.1 — Интерполяционные многочлены Лагранжа и Ньютона
# Вариант 4: y = ctg(x)
# a) X_i = π/8, 2π/8, 3π/8, 4π/8;   b) X_i = π/8, 5π/16, 3π/8, π/2
# X* = π/3

import math

def f(x):
    """y = ctg(x) = cos(x)/sin(x)"""
    return math.cos(x) / math.sin(x)


# ─────────────── Многочлен Лагранжа (формула 3.5) ───────────────
def lagrange(X, F, x):
    """
    L_n(x) = Σ f_i · ω_{n+1}(x) / ((x - x_i) · ω'_{n+1}(x_i))
    где ω_{n+1}(x) = Π(x - x_j),  ω'_{n+1}(x_i) = Π_{j≠i}(x_i - x_j)
    """
    n = len(X)
    result = 0.0
    for i in range(n):
        # базисный многочлен l_i(x) = Π_{j≠i} (x - x_j)/(x_i - x_j)
        li = 1.0
        for j in range(n):
            if j != i:
                li *= (x - X[j]) / (X[i] - X[j])
        result += F[i] * li
    return result


# ──────────── Разделённые разности (для многочлена Ньютона) ──────────
def divided_differences(X, F):
    """
    Таблица разделённых разностей (формула 3.7).
    dd[k] = f(x_0, x_1, ..., x_k)  (диагональные элементы).
    """
    n = len(X)
    # dd[i][j] = f(x_i, x_{i+1}, ..., x_{i+j})
    dd = [[0.0] * n for _ in range(n)]
    for i in range(n):
        dd[i][0] = F[i]
    for j in range(1, n):
        for i in range(n - j):
            dd[i][j] = (dd[i+1][j-1] - dd[i][j-1]) / (X[i+j] - X[i])
    return dd


def newton(X, F, x):
    """
    Многочлен Ньютона (формула 3.8):
    P_n(x) = f(x_0) + (x-x_0)·f(x_0,x_1) + (x-x_0)(x-x_1)·f(x_0,x_1,x_2) + ...
    """
    dd = divided_differences(X, F)
    n = len(X)
    result = dd[0][0]
    product = 1.0
    for k in range(1, n):
        product *= (x - X[k-1])
        result += dd[0][k] * product
    return result


# ────────────────────────── Вывод результатов ──────────────────────
def solve_variant(label, X, x_star):
    F = [f(xi) for xi in X]
    n = len(X)

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"\n  Узлы интерполяции (n = {n-1}):")
    print(f"  {'i':>4}  {'x_i':>14}  {'f(x_i) = ctg(x_i)':>20}")
    print(f"  {'-'*42}")
    for i in range(n):
        print(f"  {i:>4}  {X[i]:>14.8f}  {F[i]:>20.8f}")

    # --- Таблица разделённых разностей ---
    dd = divided_differences(X, F)
    print(f"\n  Таблица разделённых разностей:")
    header = f"  {'i':>4}  {'x_i':>10}  {'f_i':>12}"
    for k in range(1, n):
        header += f"  {'порядок '+str(k):>14}"
    print(header)
    print(f"  {'-'*(14 + 14*n)}")
    for i in range(n):
        row = f"  {i:>4}  {X[i]:>10.6f}  {dd[i][0]:>12.6f}"
        for j in range(1, n - i):
            row += f"  {dd[i][j]:>14.6f}"
        print(row)

    # --- Многочлен Лагранжа ---
    # Промежуточные величины для Лагранжа (как в примере 3.1 методички)
    print(f"\n  --- Многочлен Лагранжа ---")
    print(f"  {'i':>4}  {'x_i':>12}  {'f_i':>12}  {'ω`(x_i)':>14}  {'f_i/ω`(x_i)':>16}  {'X*-x_i':>12}")
    print(f"  {'-'*72}")

    for i in range(n):
        omega_prime = 1.0
        for j in range(n):
            if j != i:
                omega_prime *= (X[i] - X[j])
        ratio = F[i] / omega_prime
        diff = x_star - X[i]
        print(f"  {i:>4}  {X[i]:>12.8f}  {F[i]:>12.6f}  {omega_prime:>14.6f}  {ratio:>16.6f}  {diff:>12.6f}")

    L_val = lagrange(X, F, x_star)
    print(f"\n  L_{n-1}(X*) = L_{n-1}({x_star:.8f}) = {L_val:.8f}")

    # --- Многочлен Ньютона ---
    print(f"\n  --- Многочлен Ньютона ---")
    N_val = newton(X, F, x_star)
    print(f"  P_{n-1}(X*) = P_{n-1}({x_star:.8f}) = {N_val:.8f}")

    # --- Точное значение и погрешность ---
    exact = f(x_star)
    err_L = abs(exact - L_val)
    err_N = abs(exact - N_val)

    print(f"\n  Точное значение: f(X*) = ctg({x_star:.8f}) = {exact:.8f}")
    print(f"  Абсолютная погрешность Лагранжа:  Δ(L_{n-1}) = |f(X*) - L_{n-1}(X*)| = {err_L:.8f}")
    print(f"  Абсолютная погрешность Ньютона:   Δ(P_{n-1}) = |f(X*) - P_{n-1}(X*)| = {err_N:.8f}")

    # --- Априорная оценка погрешности (формула 3.9) ---
    # |ε_n(x)| ≤ M_{n+1} / (n+1)! · |ω_{n+1}(x)|
    omega_val = 1.0
    for xi in X:
        omega_val *= (x_star - xi)
    omega_val = abs(omega_val)
    factorial = math.factorial(n)
    print(f"\n  |ω_{n}(X*)| = {omega_val:.8e}")
    print(f"  (n+1)! = {n}! = {factorial}")
    apriori_factor = omega_val / factorial
    print(f"  Фактор ω/(n+1)! = {apriori_factor:.8e}")


# ─────────────────────────────── main ─────────────────────────────
def main():
    pi = math.pi
    x_star = pi / 3

    print("=" * 65)
    print("  Лабораторная работа 3.1  |  Вариант 4")
    print(f"  Функция: y = ctg(x)")
    print(f"  Точка интерполяции: X* = π/3 = {x_star:.8f}")
    print("=" * 65)

    # Набор a)
    X_a = [pi/8, 2*pi/8, 3*pi/8, 4*pi/8]
    solve_variant("Набор а)  X_i = π/8, 2π/8, 3π/8, 4π/8", X_a, x_star)

    # Набор b)
    X_b = [pi/8, 5*pi/16, 3*pi/8, pi/2]
    solve_variant("Набор б)  X_i = π/8, 5π/16, 3π/8, π/2", X_b, x_star)

    # Сравнение
    F_a = [f(xi) for xi in X_a]
    F_b = [f(xi) for xi in X_b]
    exact = f(x_star)

    err_a = abs(exact - lagrange(X_a, F_a, x_star))
    err_b = abs(exact - lagrange(X_b, F_b, x_star))

    print(f"\n{'='*65}")
    print(f"  Сравнение наборов:")
    print(f"    Набор а): погрешность = {err_a:.8e}")
    print(f"    Набор б): погрешность = {err_b:.8e}")
    if err_a < err_b:
        print(f"    Набор а) точнее (точка X* ближе к центру набора а)")
    else:
        print(f"    Набор б) точнее (точка X* ближе к центру набора б)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
