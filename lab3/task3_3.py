# Лаб 3.3 — Метод наименьших квадратов (МНК)
# Вариант 4
# Данные: x = [1.0, 1.9, 2.8, 3.7, 4.6, 5.5]
#         y = [2.4142, 1.0818, 0.50953, 0.11836, -0.24008, -0.66818]
# Найти приближающие многочлены 1-й и 2-й степени.
# Построить графики.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def solve_linear_system_2(A, b):
    """Решение 2×2 системы методом Крамера."""
    det = A[0][0]*A[1][1] - A[0][1]*A[1][0]
    x0 = (b[0]*A[1][1] - b[1]*A[0][1]) / det
    x1 = (A[0][0]*b[1] - A[1][0]*b[0]) / det
    return [x0, x1]


def solve_linear_system_3(A, b):
    """Решение 3×3 системы методом Гаусса."""
    import copy
    n = 3
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for k in range(n):
        # Выбор главного элемента
        max_row = k
        for i in range(k+1, n):
            if abs(M[i][k]) > abs(M[max_row][k]):
                max_row = i
        M[k], M[max_row] = M[max_row], M[k]

        for i in range(k+1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n+1):
                M[i][j] -= factor * M[k][j]

    x = [0.0] * n
    for i in range(n-1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j]*x[j] for j in range(i+1, n))) / M[i][i]
    return x


def main():
    X = [1.0, 1.9, 2.8, 3.7, 4.6, 5.5]
    Y = [2.4142, 1.0818, 0.50953, 0.11836, -0.24008, -0.66818]
    N = len(X) - 1  # N = 5

    print("=" * 65)
    print("  Лабораторная работа 3.3  |  Вариант 4")
    print("  Метод наименьших квадратов")
    print("=" * 65)

    print(f"\n  Исходные данные (N = {N}):")
    print(f"  {'i':>4}  {'x_i':>10}  {'y_i':>12}")
    print(f"  {'-'*28}")
    for i in range(N + 1):
        print(f"  {i:>4}  {X[i]:>10.1f}  {Y[i]:>12.5f}")

    # ── Вычисление сумм ────────────────────────────────────────────
    sum_x = sum(X)
    sum_x2 = sum(x**2 for x in X)
    sum_x3 = sum(x**3 for x in X)
    sum_x4 = sum(x**4 for x in X)
    sum_y = sum(Y)
    sum_xy = sum(X[i]*Y[i] for i in range(N+1))
    sum_x2y = sum(X[i]**2 * Y[i] for i in range(N+1))

    print(f"\n  Суммы:")
    print(f"    Σx_i     = {sum_x:.4f}")
    print(f"    Σx_i²    = {sum_x2:.4f}")
    print(f"    Σx_i³    = {sum_x3:.4f}")
    print(f"    Σx_i⁴    = {sum_x4:.4f}")
    print(f"    Σy_i     = {sum_y:.5f}")
    print(f"    Σx_i·y_i = {sum_xy:.5f}")
    print(f"    Σx_i²·y_i= {sum_x2y:.5f}")

    # ═══════════════════════════════════════════════════════════════
    # Приближающий многочлен 1-й степени: F_1(x) = a_0 + a_1·x
    # Нормальная система (формула (a) методички):
    #   (N+1)·a_0 + Σx·a_1     = Σy
    #   Σx·a_0    + Σx²·a_1    = Σxy
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  a) Приближающий многочлен 1-й степени: F₁(x) = a₀ + a₁·x")
    print(f"{'='*65}")

    A1 = [
        [N + 1, sum_x],
        [sum_x, sum_x2],
    ]
    b1 = [sum_y, sum_xy]

    print(f"\n  Нормальная система МНК:")
    print(f"    {A1[0][0]:.1f}·a₀ + {A1[0][1]:.2f}·a₁ = {b1[0]:.5f}")
    print(f"    {A1[1][0]:.2f}·a₀ + {A1[1][1]:.2f}·a₁ = {b1[1]:.5f}")

    sol1 = solve_linear_system_2(A1, b1)
    a0_1, a1_1 = sol1

    print(f"\n  Решение:  a₀ = {a0_1:.6f},  a₁ = {a1_1:.6f}")
    print(f"  F₁(x) = {a0_1:.6f} + {a1_1:.6f}·x")

    # Значения F_1 в узлах и сумма квадратов ошибок
    print(f"\n  {'i':>4}  {'x_i':>10}  {'y_i':>12}  {'F₁(x_i)':>12}  {'y_i - F₁':>12}")
    print(f"  {'-'*52}")
    sse1 = 0.0
    for i in range(N + 1):
        f_val = a0_1 + a1_1 * X[i]
        err = Y[i] - f_val
        sse1 += err**2
        print(f"  {i:>4}  {X[i]:>10.1f}  {Y[i]:>12.5f}  {f_val:>12.5f}  {err:>12.5f}")
    print(f"\n  Сумма квадратов ошибок: Φ₁ = {sse1:.6f}")

    # ═══════════════════════════════════════════════════════════════
    # Приближающий многочлен 2-й степени: F_2(x) = a_0 + a_1·x + a_2·x²
    # Нормальная система (формула (c) методички):
    #   (N+1)·a_0 + Σx·a_1  + Σx²·a_2   = Σy
    #   Σx·a_0    + Σx²·a_1 + Σx³·a_2   = Σxy
    #   Σx²·a_0   + Σx³·a_1 + Σx⁴·a_2   = Σx²y
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  б) Приближающий многочлен 2-й степени: F₂(x) = a₀ + a₁·x + a₂·x²")
    print(f"{'='*65}")

    A2 = [
        [N + 1, sum_x, sum_x2],
        [sum_x, sum_x2, sum_x3],
        [sum_x2, sum_x3, sum_x4],
    ]
    b2 = [sum_y, sum_xy, sum_x2y]

    print(f"\n  Нормальная система МНК:")
    print(f"    {A2[0][0]:.1f}·a₀ + {A2[0][1]:.2f}·a₁ + {A2[0][2]:.2f}·a₂ = {b2[0]:.5f}")
    print(f"    {A2[1][0]:.2f}·a₀ + {A2[1][1]:.2f}·a₁ + {A2[1][2]:.2f}·a₂ = {b2[1]:.5f}")
    print(f"    {A2[2][0]:.2f}·a₀ + {A2[2][1]:.2f}·a₁ + {A2[2][2]:.2f}·a₂ = {b2[2]:.5f}")

    sol2 = solve_linear_system_3(A2, b2)
    a0_2, a1_2, a2_2 = sol2

    print(f"\n  Решение:  a₀ = {a0_2:.6f},  a₁ = {a1_2:.6f},  a₂ = {a2_2:.6f}")
    print(f"  F₂(x) = {a0_2:.6f} + {a1_2:.6f}·x + {a2_2:.6f}·x²")

    # Значения F_2 в узлах и сумма квадратов ошибок
    print(f"\n  {'i':>4}  {'x_i':>10}  {'y_i':>12}  {'F₂(x_i)':>12}  {'y_i - F₂':>12}")
    print(f"  {'-'*52}")
    sse2 = 0.0
    for i in range(N + 1):
        f_val = a0_2 + a1_2 * X[i] + a2_2 * X[i]**2
        err = Y[i] - f_val
        sse2 += err**2
        print(f"  {i:>4}  {X[i]:>10.1f}  {Y[i]:>12.5f}  {f_val:>12.5f}  {err:>12.5f}")
    print(f"\n  Сумма квадратов ошибок: Φ₂ = {sse2:.6f}")

    # ── Сравнение ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Сравнение:")
    print(f"    Φ₁ = {sse1:.6f}")
    print(f"    Φ₂ = {sse2:.6f}")
    print(f"    Многочлен 2-й степени даёт меньшую сумму квадратов ошибок")
    print(f"{'='*65}")

    # ── Графики ────────────────────────────────────────────────────
    xs_plot = [X[0] + (X[-1] - X[0]) * i / 200 for i in range(201)]
    ys_f1 = [a0_1 + a1_1 * x for x in xs_plot]
    ys_f2 = [a0_2 + a1_2 * x + a2_2 * x**2 for x in xs_plot]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X, Y, "ko", markersize=7, label="Табличные данные")
    ax.plot(xs_plot, ys_f1, "b-", linewidth=2, label=f"$F_1(x) = {a0_1:.4f} + {a1_1:.4f}x$")
    ax.plot(xs_plot, ys_f2, "r--", linewidth=2, label=f"$F_2(x) = {a0_2:.4f} + {a1_2:.4f}x + {a2_2:.4f}x^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("МНК: приближающие многочлены 1-й и 2-й степени (вариант 4)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("graph_3_3_lsq.png", dpi=120)
    plt.close()
    print(f"\n  [График] Сохранён: graph_3_3_lsq.png")


if __name__ == "__main__":
    main()
