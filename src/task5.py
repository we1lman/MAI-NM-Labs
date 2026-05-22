# Лаб 1.5 — QR-разложение и QR-алгоритм
# Вариант 4

import math
import copy

n = 3
A = [
    [-4.0, -6.0, -3.0],
    [-1.0,  5.0, -5.0],
    [ 6.0,  2.0,  5.0],
]
eps = 1e-6

print("=== QR-разложение и QR-алгоритм (вариант 4) ===\n")
print("Исходная матрица:")
for row in A:
    print(f"  {row}")


# -------------------------------------------------------
# QR-разложение методом отражений Хаусхолдера (§1.2.4)
#
# Матрица Хаусхолдера (форм. 1.29):
#   H = E − (2 / v^T·v) · v·v^T,  v ≠ 0
#
# Вектор v (форм. 1.30) для обнуления поддиагональных
# элементов k-го столбца:
#   v_k = b_k + sign(b_k)·||b||_2
#   v_i = b_i,  i = k+1..n
# где b = (a_kk, a_{k+1,k}, ..., a_nk)^T — k-й подстолбец
# -------------------------------------------------------
def qr_householder(M):
    sz = len(M)
    R = copy.deepcopy(M)
    Q = [[1.0 if i == j else 0.0 for j in range(sz)] for i in range(sz)]

    for k in range(sz - 1):
        # b = k-й подстолбец (строки k..sz-1)
        x      = [R[i][k] if i >= k else 0.0 for i in range(sz)]
        norm_x = math.sqrt(sum(x[i]**2 for i in range(k, sz)))
        if norm_x < 1e-15:
            continue

        # вектор v по формуле (1.30)
        v    = [0.0]*sz
        v[k] = x[k] + math.copysign(norm_x, x[k])
        for i in range(k+1, sz):
            v[i] = x[i]

        norm_v = math.sqrt(sum(vi**2 for vi in v))
        if norm_v < 1e-15:
            continue
        v = [vi / norm_v for vi in v]   # нормируем для численной устойчивости

        # R := H·R = (E − 2vv^T)·R
        for j in range(sz):
            dot = sum(v[i]*R[i][j] for i in range(sz))
            for i in range(sz):
                R[i][j] -= 2*v[i]*dot

        # Q := Q·H^T = Q·H  (H симметрична)
        for i in range(sz):
            dot = sum(Q[i][j]*v[j] for j in range(sz))
            for j in range(sz):
                Q[i][j] -= 2*dot*v[j]

    return Q, R


# =======================================================
# Часть 1: QR-разложение  A = Q·R
# =======================================================
print("\n--- QR-разложение (метод Хаусхолдера) ---")
Q, R = qr_householder(A)

print("\nQ:")
for row in Q:
    print(f"  {[round(x, 6) for x in row]}")

print("\nR:")
for row in R:
    print(f"  {[round(x, 6) for x in row]}")

print("\nQ·R  (должно быть = A):")
for i in range(n):
    row = [round(sum(Q[i][k]*R[k][j] for k in range(n)), 4) for j in range(n)]
    print(f"  {row}")

print("\nQ^T·Q  (должно быть = E):")
for i in range(n):
    row = [round(sum(Q[k][i]*Q[k][j] for k in range(n)), 6) for j in range(n)]
    print(f"  {row}")


# =======================================================
# Часть 2: QR-алгоритм (§1.2.4)
#
# Базовая схема без сдвигов:
#   A^(0) = A
#   A^(k) = Q^(k)·R^(k)         (QR-разложение)
#   A^(k+1) = R^(k)·Q^(k)       (перемножение в обратном порядке)
#
# Критерий сходимости для вещественного λ_m (столбец m):
#   ( Σ_{i=m+1}^{n} (a_im^(k))^2 )^{1/2}  ≤  ε
#
# Для комплексно-сопряжённой пары (блок 2×2, столбцы m, m+1):
#   |λ^(k) - λ^(k-1)|  ≤  ε
#
# Последовательность A^(k) сходится к верхней квазитреугольной
# матрице с вещественными λ на диагонали или 2×2-блоками
# для комплексных пар.
# =======================================================
print("\n--- QR-алгоритм ---")

Ak = copy.deepcopy(A)

print(f"\n{'итер':>4}  {'норма поддиаг':>14}  диагональ")
print("-" * 58)

prev_block_eig = None   # собственные значения 2×2-блока на предыдущей итерации

for iteration in range(2000):
    # норма всех поддиагональных элементов
    sd = math.sqrt(sum(Ak[i][j]**2 for i in range(1, n) for j in range(i)))

    if iteration <= 10 or iteration % 50 == 0:
        diag = [round(Ak[i][i], 6) for i in range(n)]
        print(f"{iteration:4d}  {sd:14.2e}  {diag}")

    # ---- Критерий сходимости по каждому столбцу ----
    # Для столбца m: (Σ_{i=m+1}^n a_im^2)^{1/2} ≤ ε  → λ_m ≈ a_mm
    col_ok = []
    for m in range(n):
        col_norm = math.sqrt(sum(Ak[i][m]**2 for i in range(m+1, n)))
        col_ok.append(col_norm <= eps)

    # ---- Критерий для 2×2-блока (комплексная пара) ----
    # Вычислим собственные значения нижнего 2×2-блока
    block_converged = False
    if n >= 2:
        aa = Ak[n-2][n-2]; ab = Ak[n-2][n-1]
        ba = Ak[n-1][n-2]; bb = Ak[n-1][n-1]
        tr   = aa + bb
        disc = (aa - bb)**2 + 4*ab*ba
        if disc < 0:
            lam = complex(tr/2, math.sqrt(-disc)/2)
        else:
            sq  = math.sqrt(disc)
            lam = complex((tr + sq)/2, 0)

        if prev_block_eig is not None:
            if abs(lam - prev_block_eig) <= eps:
                block_converged = True
        prev_block_eig = lam

    # Общий критерий останова:
    # либо все поддиагональные элементы малы,
    # либо 1-й столбец сошёлся И 2×2-блок сошёлся
    if all(col_ok):
        diag = [round(Ak[i][i], 6) for i in range(n)]
        print(f"{iteration:4d}  {sd:14.2e}  {diag}")
        print(f"\nСошлось за {iteration} итераций (все поддиагональные элементы малы)")
        break

    if col_ok[0] and block_converged:
        diag = [round(Ak[i][i], 6) for i in range(n)]
        print(f"{iteration:4d}  {sd:14.2e}  {diag}")
        print(f"\nСошлось за {iteration} итераций "
              f"(λ₁ вещественная + комплексная пара сошлась)")
        break

    # Один шаг QR-алгоритма: A^(k+1) = R^(k) · Q^(k)
    Qk, Rk = qr_householder(Ak)
    Ak = [[sum(Rk[i][p]*Qk[p][j] for p in range(n)) for j in range(n)]
          for i in range(n)]

else:
    print(f"\nПредупреждение: достигнуто максимальное число итераций")

# -------------------------------------------------------
# Финальная матрица и извлечение собственных значений
# -------------------------------------------------------
print("\nФинальная матрица A^(k):")
for row in Ak:
    print(f"  {[round(x, 6) for x in row]}")

print("\nСобственные значения:")
i = 0
eigs = []
while i < n:
    # вещественное λ: поддиагональные элементы i-го столбца малы
    if i == n-1 or abs(Ak[i+1][i]) < eps:
        print(f"  λ{i+1} = {Ak[i][i]:.6f}")
        eigs.append(Ak[i][i])
        i += 1
    else:
        # 2×2-блок — комплексно-сопряжённая пара
        aa = Ak[i][i];   ab = Ak[i][i+1]
        ba = Ak[i+1][i]; bb = Ak[i+1][i+1]
        tr   = aa + bb
        disc = tr**2 - 4*(aa*bb - ab*ba)
        if disc >= 0:
            sq = math.sqrt(disc)
            print(f"  λ{i+1} = {(tr+sq)/2:.6f}")
            print(f"  λ{i+2} = {(tr-sq)/2:.6f}")
            eigs += [(tr+sq)/2, (tr-sq)/2]
        else:
            sq = math.sqrt(-disc)
            print(f"  λ{i+1} = {tr/2:.6f} + {sq/2:.6f}i")
            print(f"  λ{i+2} = {tr/2:.6f} - {sq/2:.6f}i")
            eigs += [complex(tr/2, sq/2), complex(tr/2, -sq/2)]
        i += 2

# Проверка по следу
tr_A   = sum(A[i][i] for i in range(n))
tr_eig = sum(x.real if isinstance(x, complex) else x for x in eigs)
print(f"\nПроверка: tr(A) = {tr_A},  Σλ = {tr_eig:.6f},  "
      f"погрешность = {abs(tr_A - tr_eig):.2e}")
