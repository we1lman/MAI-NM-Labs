import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ───────────────────────────── Функция и её производные ─────────────────────
def f(x):
    return x**3 + x**2 - x - 0.5


def df(x):
    
    """Первая производная f'(x)"""
    return 3*x**2 + 2*x - 1


def d2f(x):
    """Вторая производная f''(x)"""
    return 6*x + 2


# ─────────────────────── Локализация корня графически ───────────────────────
def plot_function(a=-1.5, b=1.5):
    xs = [a + (b - a) * i / 500 for i in range(501)]
    ys = [f(x) for x in xs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "b-", linewidth=2, label=r"$f(x)=x^3+x^2-x-0{,}5$")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("График функции f(x) = x³ + x² − x − 0.5")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("graph_2_1_function.png", dpi=120)
    plt.close()
    print("  [График] Сохранён: graph_2_1_function.png")


# ──────────────────────── Метод простой итерации ────────────────────────────
def simple_iteration(x0: float, eps: float, lam: float = 0.25, max_iter: int = 1000):
    """
    phi(x) = x - lam * f(x)
    phi'(x) = 1 - lam * f'(x)
    При lam = 0.25 на интервале [0.5, 1.0]:
      |phi'| = |1 - 0.25 * f'| < 1  (условие сходимости выполнено).
    """
    history = []          # список (k, x_k, |x_{k+1} - x_k|)
    x = x0
    for k in range(max_iter):
        x_new = x - lam * f(x)
        diff = abs(x_new - x)
        history.append((k, x, diff))
        if diff < eps:
            history.append((k + 1, x_new, 0.0))
            return x_new, history
        x = x_new
    raise RuntimeError("Метод простой итерации не сошёлся за максимальное число итераций")


# ────────────────────────── Метод Ньютона ───────────────────────────────────
def newton(x0: float, eps: float, max_iter: int = 1000):
    """
    x_{k+1} = x_k - f(x_k) / f'(x_k)
    Условие останова: |x_{k+1} - x_k| < eps
    """
    history = []
    x = x0
    for k in range(max_iter):
        fx  = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-14:
            raise ZeroDivisionError("f'(x) ≈ 0, деление на ноль в методе Ньютона")
        x_new = x - fx / dfx
        diff  = abs(x_new - x)
        history.append((k, x, abs(fx), abs(dfx), -fx/dfx))
        if diff < eps:
            history.append((k + 1, x_new, abs(f(x_new)), abs(df(x_new)), 0.0))
            return x_new, history
        x = x_new
    raise RuntimeError("Метод Ньютона не сошёлся за максимальное число итераций")


# ──────────────────── График погрешности от итераций ────────────────────────
def plot_errors(hist_si, hist_nt, true_root):
    iters_si = [row[0] for row in hist_si]
    errs_si  = [abs(row[1] - true_root) for row in hist_si]

    iters_nt = [row[0] for row in hist_nt]
    errs_nt  = [abs(row[1] - true_root) for row in hist_nt]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(iters_si, errs_si, "b-o", markersize=4, label="Простая итерация")
    ax.semilogy(iters_nt, errs_nt, "r-s", markersize=4, label="Ньютон")
    ax.set_xlabel("Номер итерации k")
    ax.set_ylabel("Погрешность |x_k − x*|  (лог. масштаб)")
    ax.set_title("Зависимость погрешности от числа итераций")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("graph_2_1_errors.png", dpi=120)
    plt.close()
    print("  [График] Сохранён: graph_2_1_errors.png")


# ─────────────────────────────── Вывод таблиц ───────────────────────────────
def print_table_si(history):
    print("\n  Метод простой итерации (phi(x) = x − 0.25·f(x))")
    print(f"  {'k':>4}  {'x^(k)':>14}  {'|x^(k+1) - x^(k)|':>20}")
    print("  " + "-" * 44)
    for row in history:
        k, xk, diff = row
        print(f"  {k:>4}  {xk:>14.8f}  {diff:>20.2e}")


def print_table_newton(history):
    print("\n  Метод Ньютона")
    print(f"  {'k':>4}  {'x^(k)':>14}  {'f(x^(k))':>14}  {'f`(x^(k))':>14}  {'-f/f`':>12}")
    print("  " + "-" * 64)
    for row in history:
        k, xk, fxk, dfxk, step = row
        print(f"  {k:>4}  {xk:>14.8f}  {fxk:>14.8f}  {dfxk:>14.8f}  {step:>12.8f}")


# ─────────────────────────────────── main ───────────────────────────────────
def main():
    eps = 1e-6

    print("=" * 65)
    print("  Лабораторная работа 2.1  |  Вариант 4")
    print("  Уравнение: x³ + x² − x − 0.5 = 0")
    print(f"  Точность: ε = {eps}")
    print("=" * 65)

    # Строим график функции для локализации корня
    plot_function()

    # ── Локализация ──────────────────────────────────────────────────────────
    # Проверяем знаки f на отрезке [0, 1]:
    #   f(0) = -0.5  < 0
    #   f(1) =  1+1-1-0.5 = 0.5  > 0
    # => на [0, 1] содержится ровно один положительный корень.
    # f'(x) = 3x^2 + 2x - 1 > 0 при x > 1/3 — функция строго возрастает на [1/3, 1].
    # f''(x) = 6x+2 > 0 при x > 0.
    # По условию Ньютона (ф-ла 2.3): f(x0)*f''(x0) > 0 => выбираем x0 = 1.0 (правая граница).
    print("\n  Локализация корня:")
    print(f"    f(0.0) = {f(0.0):.4f}  (< 0)")
    print(f"    f(1.0) = {f(1.0):.4f}  (> 0)")
    print("    Корень находится на [0.0, 1.0].")
    print("    Начальное приближение x^(0) = 1.0 (выбрано по условию Ньютона: f(x0)·f''(x0) > 0)")

    x0 = 1.0
    lam = 0.25   # λ = 1 / max f'(x) на [0,1] ≈ 1/4 гарантирует |φ'| < 1

    # ── Простая итерация ─────────────────────────────────────────────────────
    root_si, hist_si = simple_iteration(x0, eps, lam)
    print_table_si(hist_si)
    print(f"\n  Результат (простая итерация): x* ≈ {root_si:.8f}")
    print(f"  f(x*) = {f(root_si):.2e}   (невязка)")
    print(f"  Итераций: {len(hist_si) - 1}")

    # ── Метод Ньютона ─────────────────────────────────────────────────────────
    root_nt, hist_nt = newton(x0, eps)
    print_table_newton(hist_nt)
    print(f"\n  Результат (метод Ньютона):    x* ≈ {root_nt:.8f}")
    print(f"  f(x*) = {f(root_nt):.2e}   (невязка)")
    print(f"  Итераций: {len(hist_nt) - 1}")

    # Используем более точный корень Ньютона как эталонный для оценки погрешности
    true_root = root_nt
    plot_errors(hist_si, hist_nt, true_root)

    print("\n" + "=" * 65)
    print(f"  Итоговый положительный корень: x* ≈ {root_nt:.8f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
