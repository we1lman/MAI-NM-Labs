# Лаб 3.5 — Численное интегрирование
# Вариант 4
# y = (3x + 4) / (2x + 7)
# X_0 = -2, X_k = 2, h_1 = 1.0, h_2 = 0.5
# Методы: прямоугольников, трапеций, Симпсона
# Уточнение: метод Рунге-Ромберга

import math


def y(x):
    """Подынтегральная функция: (3x + 4) / (2x + 7)"""
    return (3 * x + 4) / (2 * x + 7)


def exact_integral(a, b):
    """
    Точное значение интеграла:
    ∫ (3x+4)/(2x+7) dx = (3/2)x + (-13/4)·ln|2x+7| + C

    Проверка: d/dx [(3/2)x - (13/4)·ln(2x+7)]
            = 3/2 - (13/4)·(2/(2x+7))
            = 3/2 - 13/(2(2x+7))
            = [3(2x+7) - 13] / [2(2x+7)]
            = (6x + 21 - 13) / (2(2x+7))
            = (6x + 8) / (2(2x+7))
            = (3x + 4) / (2x + 7)  ✓
    """
    def F(x):
        return 1.5 * x - 3.25 * math.log(abs(2 * x + 7))
    return F(b) - F(a)


def method_rectangles(a, b, h):
    """
    Метод прямоугольников (средних, формула 3.23):
    F ≈ Σ h · f((x_{i-1} + x_i) / 2)
    """
    N = round((b - a) / h)
    result = 0.0
    for i in range(N):
        x_mid = a + (i + 0.5) * h
        result += h * y(x_mid)
    return result


def method_trapezoid(a, b, h):
    """
    Метод трапеций (формула 3.25):
    F ≈ h · [y_0/2 + y_1 + y_2 + ... + y_{N-1} + y_N/2]
    """
    N = round((b - a) / h)
    result = y(a) / 2 + y(b) / 2
    for i in range(1, N):
        result += y(a + i * h)
    result *= h
    return result


def method_simpson(a, b, h):
    """
    Метод Симпсона (формула 3.28):
    F ≈ (h/3) · [y_0 + 4·y_{1/2} + 2·y_1 + 4·y_{3/2} + ... + 4·y_{N-1/2} + y_N]
    Число интервалов 2N; на вход h — расстояние между основными узлами (не полуузлами).
    """
    N = round((b - a) / h)
    result = y(a) + y(b)
    for i in range(1, N):
        result += 2 * y(a + i * h)
    for i in range(N):
        x_half = a + (i + 0.5) * h
        result += 4 * y(x_half)
    result *= h / 6
    return result


def runge_romberg(F_h, F_kh, k, p):
    """
    Метод Рунге-Ромберга (формула 3.30):
    F ≈ F_h + (F_h - F_{kh}) / (k^p - 1)
    k = h_1 / h_2 = 2 (отношение шагов)
    p — порядок точности метода
    """
    return F_h + (F_h - F_kh) / (k**p - 1)


def main():
    a = -2.0
    b = 2.0
    h1 = 1.0
    h2 = 0.5

    print("=" * 70)
    print("  Лабораторная работа 3.5  |  Вариант 4")
    print("  Численное интегрирование")
    print(f"  y = (3x + 4) / (2x + 7)")
    print(f"  Пределы: [{a}, {b}],  h₁ = {h1},  h₂ = {h2}")
    print("=" * 70)

    F_exact = exact_integral(a, b)
    print(f"\n  Точное значение интеграла: F = {F_exact:.8f}")

    # ── Вычисления с шагом h1 ──────────────────────────────────────
    print(f"\n  --- Шаг h₁ = {h1} ---")
    N1 = round((b - a) / h1)

    # Таблица значений
    print(f"\n  {'i':>4}  {'x_i':>8}  {'y_i':>12}")
    print(f"  {'-'*28}")
    for i in range(N1 + 1):
        xi = a + i * h1
        print(f"  {i:>4}  {xi:>8.2f}  {y(xi):>12.6f}")

    F_rect_h1 = method_rectangles(a, b, h1)
    F_trap_h1 = method_trapezoid(a, b, h1)
    F_simp_h1 = method_simpson(a, b, h1)

    print(f"\n  Метод прямоугольников: {F_rect_h1:.8f}")
    print(f"  Метод трапеций:        {F_trap_h1:.8f}")
    print(f"  Метод Симпсона:        {F_simp_h1:.8f}")

    # ── Вычисления с шагом h2 ──────────────────────────────────────
    print(f"\n  --- Шаг h₂ = {h2} ---")
    N2 = round((b - a) / h2)

    print(f"\n  {'i':>4}  {'x_i':>8}  {'y_i':>12}")
    print(f"  {'-'*28}")
    for i in range(N2 + 1):
        xi = a + i * h2
        print(f"  {i:>4}  {xi:>8.2f}  {y(xi):>12.6f}")

    F_rect_h2 = method_rectangles(a, b, h2)
    F_trap_h2 = method_trapezoid(a, b, h2)
    F_simp_h2 = method_simpson(a, b, h2)

    print(f"\n  Метод прямоугольников: {F_rect_h2:.8f}")
    print(f"  Метод трапеций:        {F_trap_h2:.8f}")
    print(f"  Метод Симпсона:        {F_simp_h2:.8f}")

    # ── Уточнение по Рунге-Ромбергу (формула 3.30) ─────────────────
    k = h1 / h2  # k = 2

    # Прямоугольники: порядок p = 2
    F_rect_rr = runge_romberg(F_rect_h2, F_rect_h1, k, 2)
    # Трапеции: порядок p = 2
    F_trap_rr = runge_romberg(F_trap_h2, F_trap_h1, k, 2)
    # Симпсон: порядок p = 4
    F_simp_rr = runge_romberg(F_simp_h2, F_simp_h1, k, 4)

    print(f"\n  --- Уточнение методом Рунге-Ромберга ---")
    print(f"  k = h₁/h₂ = {k:.0f}")
    print(f"\n  Прямоугольники (p=2): {F_rect_rr:.8f}")
    print(f"  Трапеции (p=2):       {F_trap_rr:.8f}")
    print(f"  Симпсон (p=4):        {F_simp_rr:.8f}")

    # ── Итоговая таблица ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Итоговая сводная таблица:")
    print(f"  {'-'*68}")
    print(f"  {'':>25}  {'Точное':>12}  {'h₁='+str(h1):>10}  {'h₂='+str(h2):>10}  {'Рунге-Р.':>10}")
    print(f"  {'-'*68}")
    print(f"  {'Прямоугольники':>25}  {'':>12}  {F_rect_h1:>10.6f}  {F_rect_h2:>10.6f}  {F_rect_rr:>10.6f}")
    print(f"  {'Трапеции':>25}  {'':>12}  {F_trap_h1:>10.6f}  {F_trap_h2:>10.6f}  {F_trap_rr:>10.6f}")
    print(f"  {'Симпсон':>25}  {'':>12}  {F_simp_h1:>10.6f}  {F_simp_h2:>10.6f}  {F_simp_rr:>10.6f}")
    print(f"  {'Точное значение':>25}  {F_exact:>12.6f}")
    print(f"  {'-'*68}")

    print(f"\n  Абсолютные погрешности:")
    print(f"  {'-'*68}")
    print(f"  {'':>25}  {'h₁':>10}  {'h₂':>10}  {'Рунге-Р.':>10}")
    print(f"  {'-'*68}")
    print(f"  {'Прямоугольники':>25}  {abs(F_exact - F_rect_h1):>10.6f}  {abs(F_exact - F_rect_h2):>10.6f}  {abs(F_exact - F_rect_rr):>10.6f}")
    print(f"  {'Трапеции':>25}  {abs(F_exact - F_trap_h1):>10.6f}  {abs(F_exact - F_trap_h2):>10.6f}  {abs(F_exact - F_trap_rr):>10.6f}")
    print(f"  {'Симпсон':>25}  {abs(F_exact - F_simp_h1):>10.6f}  {abs(F_exact - F_simp_h2):>10.6f}  {abs(F_exact - F_simp_rr):>10.6f}")
    print(f"  {'-'*68}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
