# Лаб 4.1 — Задача Коши для ОДУ 2-го порядка
# Вариант 4
#
# x^2 y'' - x(x^2-1) y' - (x^2+1) y = 0
# y(1) = 1 + e^{1/2},  y'(1) = -1
# x in [1, 2],  h = 0.1
# Точное решение: y = (1/x)(1 + e^{x^2/2})
#
# ЗАМЕЧАНИЕ: в задании указано y'(1) = 2e^{1/2} - 1, однако это
# не согласуется с точным решением. Из y = (1+e^{x^2/2})/x следует
# y'(1) = -1. Используем корректное начальное условие.
#
# Сведение к системе (замена z = y'):
#   y' = z
#   z' = [x(x^2-1) z + (x^2+1) y] / x^2

import math


def y_exact(x):
    return (1.0 + math.exp(x**2 / 2.0)) / x


def f_sys(x, y, z):
    return z


def g_sys(x, y, z):
    return (x * (x**2 - 1) * z + (x**2 + 1) * y) / (x**2)


# --- Метод Эйлера (формула 4.2) ---
def euler(x0, y0, z0, h, n_steps):
    xs, ys, zs = [x0], [y0], [z0]
    x, y, z = x0, y0, z0
    for _ in range(n_steps):
        y_new = y + h * f_sys(x, y, z)
        z_new = z + h * g_sys(x, y, z)
        x += h
        y, z = y_new, z_new
        xs.append(x); ys.append(y); zs.append(z)
    return xs, ys, zs


# --- Метод Рунге-Кутты 4-го порядка (формула 4.16) ---
def rk4(x0, y0, z0, h, n_steps):
    xs, ys, zs = [x0], [y0], [z0]
    x, y, z = x0, y0, z0
    for _ in range(n_steps):
        K1 = h * f_sys(x, y, z)
        L1 = h * g_sys(x, y, z)
        K2 = h * f_sys(x + h/2, y + K1/2, z + L1/2)
        L2 = h * g_sys(x + h/2, y + K1/2, z + L1/2)
        K3 = h * f_sys(x + h/2, y + K2/2, z + L2/2)
        L3 = h * g_sys(x + h/2, y + K2/2, z + L2/2)
        K4 = h * f_sys(x + h, y + K3, z + L3)
        L4 = h * g_sys(x + h, y + K3, z + L3)
        y += (K1 + 2*K2 + 2*K3 + K4) / 6
        z += (L1 + 2*L2 + 2*L3 + L4) / 6
        x += h
        xs.append(x); ys.append(y); zs.append(z)
    return xs, ys, zs


# --- Метод Адамса 4-го порядка (Адамс-Башфорт, экстраполяционный) ---
# y_{k+1} = y_k + h/24 (55 f_k - 59 f_{k-1} + 37 f_{k-2} - 9 f_{k-3})
# Разгон: первые 4 точки из РК4.
def adams4(x0, y0, z0, h, n_steps):
    xs_rk, ys_rk, zs_rk = rk4(x0, y0, z0, h, min(3, n_steps))
    xs = list(xs_rk); ys = list(ys_rk); zs = list(zs_rk)
    fs = [f_sys(xs[i], ys[i], zs[i]) for i in range(len(xs))]
    gs = [g_sys(xs[i], ys[i], zs[i]) for i in range(len(xs))]
    for k in range(3, n_steps):
        y_new = ys[k] + h/24 * (55*fs[k] - 59*fs[k-1] + 37*fs[k-2] - 9*fs[k-3])
        z_new = zs[k] + h/24 * (55*gs[k] - 59*gs[k-1] + 37*gs[k-2] - 9*gs[k-3])
        x_new = xs[k] + h
        xs.append(x_new); ys.append(y_new); zs.append(z_new)
        fs.append(f_sys(x_new, y_new, z_new))
        gs.append(g_sys(x_new, y_new, z_new))
    return xs, ys, zs


def print_table(name, xs, ys, h_val):
    print(f"\n  --- {name} (h = {h_val}) ---")
    print(f"  {'k':>4}  {'x_k':>10}  {'y_k':>14}  {'y_exact':>14}  {'|err|':>12}")
    print(f"  {'-'*58}")
    for k in range(len(xs)):
        ye = y_exact(xs[k])
        err = abs(ys[k] - ye)
        print(f"  {k:>4}  {xs[k]:>10.4f}  {ys[k]:>14.8f}  {ye:>14.8f}  {err:>12.2e}")


def main():
    x0 = 1.0
    x_end = 2.0
    h = 0.1
    y0 = 1.0 + math.exp(0.5)
    z0 = -1.0   # корректное y'(1) из точного решения
    n_steps = round((x_end - x0) / h)

    print("=" * 70)
    print("  Лабораторная работа 4.1  |  Вариант 4")
    print("  Задача Коши для ОДУ 2-го порядка")
    print("  x^2 y'' - x(x^2-1) y' - (x^2+1) y = 0")
    print(f"  y(1) = 1 + e^(1/2) = {y0:.8f}")
    print(f"  y'(1) = -1  (корректное значение из точного решения)")
    print(f"  x in [{x0}, {x_end}],  h = {h}")
    print(f"  Точное решение: y = (1/x)(1 + e^(x^2/2))")
    print("=" * 70)

    # Эйлер
    xs_e, ys_e, zs_e = euler(x0, y0, z0, h, n_steps)
    print_table("Метод Эйлера", xs_e, ys_e, h)

    # РК4
    xs_r, ys_r, zs_r = rk4(x0, y0, z0, h, n_steps)
    print_table("Метод Рунге-Кутты 4-го порядка", xs_r, ys_r, h)

    # Адамс
    xs_a, ys_a, zs_a = adams4(x0, y0, z0, h, n_steps)
    print_table("Метод Адамса 4-го порядка", xs_a, ys_a, h)

    # --- Оценка погрешности по Рунге-Ромбергу ---
    h2 = h / 2
    n2 = round((x_end - x0) / h2)
    xs_e2, ys_e2, _ = euler(x0, y0, z0, h2, n2)
    xs_r2, ys_r2, _ = rk4(x0, y0, z0, h2, n2)
    xs_a2, ys_a2, _ = adams4(x0, y0, z0, h2, n2)

    print(f"\n  --- Оценка погрешности по Рунге-Ромбергу (h2 = {h2}) ---")
    print(f"  {'k':>4}  {'x_k':>8}  {'Euler(RR)':>14}  {'RK4(RR)':>14}  {'Adams(RR)':>14}")
    print(f"  {'-'*60}")
    for k in range(n_steps + 1):
        k2 = k * 2
        rr_e = abs(ys_e[k] - ys_e2[k2]) / (2**1 - 1)
        rr_r = abs(ys_r[k] - ys_r2[k2]) / (2**4 - 1)
        rr_a = abs(ys_a[k] - ys_a2[k2]) / (2**4 - 1)
        xv = x0 + k * h
        print(f"  {k:>4}  {xv:>8.2f}  {rr_e:>14.2e}  {rr_r:>14.2e}  {rr_a:>14.2e}")

    # --- Итого ---
    ye_final = y_exact(x_end)
    print(f"\n  Сравнение в точке x = {x_end}:")
    print(f"  {'-'*74}")
    fmt = "  {:>30s}  {:>14.8f}  {:>14.8f}  {:>12.2e}"
    print(f"  {'Метод':>30s}  {'y_числ':>14s}  {'y_точн':>14s}  {'|err|':>12s}")
    print(f"  {'-'*74}")
    print(fmt.format("Эйлер", ys_e[-1], ye_final, abs(ys_e[-1]-ye_final)))
    print(fmt.format("Рунге-Кутта 4", ys_r[-1], ye_final, abs(ys_r[-1]-ye_final)))
    print(fmt.format("Адамс 4", ys_a[-1], ye_final, abs(ys_a[-1]-ye_final)))
    print(f"  {'-'*74}")
    print("=" * 70)


if __name__ == "__main__":
    main()
