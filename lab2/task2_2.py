import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────── Параметр и уравнения системы ──────────────────────────
A = 1  # значение параметра a для варианта 4


def f1(x1, x2):
    return x1 - math.cos(x2) - 1.0


def f2(x1, x2):
    """log10(x1+1) — десятичный логарифм (lg в методичке)"""
    return x2 - math.log10(x1 + 1.0) - 1.0


# Частные производные
def df1_dx1(x1, x2): return 1.0
def df1_dx2(x1, x2): return math.sin(x2)
def df2_dx1(x1, x2): return -1.0 / ((x1 + 1.0) * math.log(10))
def df2_dx2(x1, x2): return 1.0


# ─────────────────── Локализация: построение кривых ─────────────────────────
def plot_curves():
    """
    Строим кривые f1=0 и f2=0 на плоскости (x1, x2).
    Из f1=0: x1 = 1 + cos(x2)  — параметризация по x2.
    Из f2=0: x2 = 1 + log10(x1+1) — параметризация по x1.
    """
    # Кривая f1=0: x1 = 1 + cos(x2), x2 in [0, pi]
    t1 = [i * math.pi / 200 for i in range(201)]
    c1_x1 = [1.0 + math.cos(t) for t in t1]
    c1_x2 = list(t1)

    # Кривая f2=0: x2 = 1 + log10(x1+1), x1 in [0, 3]
    t2 = [i * 3.0 / 200 for i in range(201)]
    c2_x1 = list(t2)
    c2_x2 = [1.0 + math.log10(x + 1.0) for x in t2]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(c1_x1, c1_x2, "b-", linewidth=2, label=r"$f_1=0$: $x_1 = 1+\cos x_2$")
    ax.plot(c2_x1, c2_x2, "r-", linewidth=2, label=r"$f_2=0$: $x_2 = 1+\lg(x_1+1)$")
    ax.axhline(0, color="k", linewidth=0.6)
    ax.axvline(0, color="k", linewidth=0.6)
    ax.set_xlim(-0.2, 3.0)
    ax.set_ylim(-0.2, 3.0)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Локализация решения системы (вариант 4, a=1)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("graph_2_2_curves.png", dpi=120)
    plt.close()
    print("  [График] Сохранён: graph_2_2_curves.png")


# ──────────────────────────── Метод Ньютона ─────────────────────────────────
def newton_system(x1_0: float, x2_0: float, eps: float, max_iter: int = 1000):
    """
    Итерационный процесс по формуле (2.17):
        x^(k+1) = x^(k) - J^{-1}(x^(k)) * f(x^(k))
    Для системы 2x2 используется формула Крамера (2.20).
    Условие останова: max(|dx1|, |dx2|) < eps  (норма sup, ф-ла 2.18)
    """
    history = []
    x1, x2 = x1_0, x2_0

    for k in range(max_iter):
        F1 = f1(x1, x2)
        F2 = f2(x1, x2)

        # Матрица Якоби J
        j11 = df1_dx1(x1, x2); j12 = df1_dx2(x1, x2)
        j21 = df2_dx1(x1, x2); j22 = df2_dx2(x1, x2)

        det_J = j11 * j22 - j12 * j21
        if abs(det_J) < 1e-14:
            raise ZeroDivisionError(f"det J ≈ 0 на итерации {k}")

        # det A1 (заменяем первый столбец на -F)
        det_A1 = (-F1) * j22 - j12 * (-F2)
        # det A2 (заменяем второй столбец на -F)
        det_A2 = j11 * (-F2) - (-F1) * j21

        dx1 = det_A1 / det_J
        dx2 = det_A2 / det_J

        x1_new = x1 + dx1
        x2_new = x2 + dx2

        err = max(abs(dx1), abs(dx2))
        history.append({
            "k": k, "x1": x1, "x2": x2,
            "f1": F1, "f2": F2,
            "j11": j11, "j12": j12, "j21": j21, "j22": j22,
            "det_J": det_J, "dx1": dx1, "dx2": dx2, "err": err
        })

        if err < eps:
            history.append({
                "k": k + 1, "x1": x1_new, "x2": x2_new,
                "f1": f1(x1_new, x2_new), "f2": f2(x1_new, x2_new),
                "j11": 0, "j12": 0, "j21": 0, "j22": 0,
                "det_J": 0, "dx1": 0, "dx2": 0, "err": 0.0
            })
            return x1_new, x2_new, history

        x1, x2 = x1_new, x2_new

    raise RuntimeError("Метод Ньютона (система) не сошёлся")


# ─────────────────── Метод простой итерации (система) ───────────────────────
def simple_iteration_system(x1_0: float, x2_0: float, eps: float, max_iter: int = 1000):
    """
    Из уравнений системы выражаем итерационную схему:
        phi1(x1, x2) = 1 + cos(x2)        <- из f1=0
        phi2(x1, x2) = 1 + log10(x1 + 1)  <- из f2=0

    Матрица производных phi' в окрестности корня:
        d phi1/dx1 = 0,            d phi1/dx2 = -sin(x2)
        d phi2/dx1 = 1/((x1+1)*ln10), d phi2/dx2 = 0

    Норма (максимальная строчная сумма модулей):
      строка 1: |0| + |sin(x2)| ≈ sin(1.3) ≈ 0.96
      строка 2: 1/((x1+1)*ln10) ≈ 1/(2.8*2.303) ≈ 0.155
    q = max(0.96, 0.155) ≈ 0.96 < 1 — условие сходимости выполнено, но q близко к 1,
    поэтому сходимость медленнее, чем у Ньютона.

    Условие останова: q/(1-q) * max(|dx1|,|dx2|) < eps  (ф-ла 2.26)
    """
    q = 0.96   # оценка нормы матрицы производных в области решения
    history = []
    x1, x2 = x1_0, x2_0

    for k in range(max_iter):
        x1_new = 1.0 + math.cos(x2)
        x2_new = 1.0 + math.log10(x1 + 1.0)

        diff = max(abs(x1_new - x1), abs(x2_new - x2))
        err_bound = q / (1.0 - q) * diff

        history.append({
            "k": k, "x1": x1, "x2": x2,
            "phi1": x1_new, "phi2": x2_new,
            "diff": diff, "err_bound": err_bound
        })

        if err_bound < eps:
            history.append({
                "k": k + 1, "x1": x1_new, "x2": x2_new,
                "phi1": x1_new, "phi2": x2_new,
                "diff": 0.0, "err_bound": 0.0
            })
            return x1_new, x2_new, history

        x1, x2 = x1_new, x2_new

    raise RuntimeError("Метод простой итерации (система) не сошёлся")


# ──────────────────── График погрешности от итераций ────────────────────────
def plot_errors_system(hist_si, hist_nt, true_x1, true_x2):
    def err(row_x1, row_x2):
        return max(abs(row_x1 - true_x1), abs(row_x2 - true_x2))

    iters_si = [r["k"] for r in hist_si]
    errs_si  = [err(r["x1"], r["x2"]) for r in hist_si]

    iters_nt = [r["k"] for r in hist_nt]
    errs_nt  = [err(r["x1"], r["x2"]) for r in hist_nt]

    # Защита от нулей в логарифмическом масштабе
    errs_si = [max(e, 1e-16) for e in errs_si]
    errs_nt = [max(e, 1e-16) for e in errs_nt]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(iters_si, errs_si, "b-o", markersize=4, label="Простая итерация")
    ax.semilogy(iters_nt, errs_nt, "r-s", markersize=4, label="Ньютон")
    ax.set_xlabel("Номер итерации k")
    ax.set_ylabel("Погрешность max|x_i^(k) − x_i*|  (лог. масштаб)")
    ax.set_title("Зависимость погрешности от числа итераций (система)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("graph_2_2_errors.png", dpi=120)
    plt.close()
    print("  [График] Сохранён: graph_2_2_errors.png")


# ─────────────────────────── Вывод таблиц ───────────────────────────────────
def print_table_newton_sys(history):
    print("\n  Метод Ньютона (система)")
    header = (f"  {'k':>4}  {'x1^(k)':>12}  {'x2^(k)':>12}  "
              f"{'f1':>12}  {'f2':>12}  {'det J':>10}  {'dx1':>12}  {'dx2':>12}  {'err':>12}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in history:
        print(f"  {r['k']:>4}  {r['x1']:>12.7f}  {r['x2']:>12.7f}  "
              f"{r['f1']:>12.2e}  {r['f2']:>12.2e}  {r['det_J']:>10.5f}  "
              f"{r['dx1']:>12.2e}  {r['dx2']:>12.2e}  {r['err']:>12.2e}")


def print_table_si_sys(history):
    print("\n  Метод простой итерации (система)")
    header = (f"  {'k':>4}  {'x1^(k)':>12}  {'x2^(k)':>12}  "
              f"{'phi1':>12}  {'phi2':>12}  {'||diff||':>12}  {'оценка':>12}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in history:
        print(f"  {r['k']:>4}  {r['x1']:>12.7f}  {r['x2']:>12.7f}  "
              f"{r['phi1']:>12.7f}  {r['phi2']:>12.7f}  "
              f"{r['diff']:>12.2e}  {r['err_bound']:>12.2e}")


# ─────────────────────────────────── main ───────────────────────────────────
def main():
    eps = 1e-6

    print("=" * 70)
    print("  Лабораторная работа 2.2  |  Вариант 4, a = 1")
    print("  Система: x1 - cos(x2) - 1 = 0")
    print("           x2 - lg(x1+1)  - 1 = 0")
    print(f"  Точность: ε = {eps}")
    print("=" * 70)

    plot_curves()

    # ── Локализация ──────────────────────────────────────────────────────────
    # Из графика кривых:
    #   Кривая f1=0: x1 = 1 + cos(x2) при x2 ∈ [0, π]
    #     При x2=0:   x1 = 2;   при x2≈1.3: x1 ≈ 1.27
    #   Кривая f2=0: x2 = 1 + log10(x1+1)
    #     При x1=1.27: x2 = 1 + log10(2.27) ≈ 1.356
    # Начальное приближение: x1^(0) ≈ 1.5, x2^(0) ≈ 1.3
    x1_0, x2_0 = 1.5, 1.3

    print("\n  Локализация решения (по графику):")
    print(f"    Начальное приближение: x1^(0) = {x1_0}, x2^(0) = {x2_0}")

    # ── Метод Ньютона ─────────────────────────────────────────────────────────
    x1_nt, x2_nt, hist_nt = newton_system(x1_0, x2_0, eps)
    print_table_newton_sys(hist_nt)
    print(f"\n  Результат (Ньютон):          x1* ≈ {x1_nt:.8f},  x2* ≈ {x2_nt:.8f}")
    print(f"  Невязка: f1 = {f1(x1_nt, x2_nt):.2e},  f2 = {f2(x1_nt, x2_nt):.2e}")
    print(f"  Итераций: {len(hist_nt) - 1}")

    # ── Метод простой итерации ─────────────────────────────────────────────
    x1_si, x2_si, hist_si = simple_iteration_system(x1_0, x2_0, eps)
    print_table_si_sys(hist_si)
    print(f"\n  Результат (простая итерация): x1* ≈ {x1_si:.8f},  x2* ≈ {x2_si:.8f}")
    print(f"  Невязка: f1 = {f1(x1_si, x2_si):.2e},  f2 = {f2(x1_si, x2_si):.2e}")
    print(f"  Итераций: {len(hist_si) - 1}")

    # Эталонный корень — результат Ньютона
    plot_errors_system(hist_si, hist_nt, x1_nt, x2_nt)

    print("\n" + "=" * 70)
    print(f"  Итоговое положительное решение: x1* ≈ {x1_nt:.8f},  x2* ≈ {x2_nt:.8f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
