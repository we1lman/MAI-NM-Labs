import math
import copy


# -------------------------------------------------------
# QR-разложение методом отражений Хаусхолдера (§1.2.4)
#
# Матрица Хаусхолдера (форм. 1.29):
#   H = E − (2 / v^T·v) · v·v^T
#
# Вектор v (форм. 1.30):
#   v_k = b_k + sign(b_k)·||b||_2,   v_i = b_i  (i > k)
#   где b — k-й поддиагональный подстолбец
# -------------------------------------------------------
def qr_householder(M):
    sz = len(M)
    R  = copy.deepcopy(M)
    Q  = [[1.0 if i == j else 0.0 for j in range(sz)] for i in range(sz)]

    for k in range(sz - 1):
        x      = [R[i][k] if i >= k else 0.0 for i in range(sz)]
        norm_x = math.sqrt(sum(x[i]**2 for i in range(k, sz)))
        if norm_x < 1e-15:
            continue

        v    = [0.0]*sz
        v[k] = x[k] + math.copysign(norm_x, x[k])
        for i in range(k+1, sz):
            v[i] = x[i]

        norm_v = math.sqrt(sum(vi**2 for vi in v))
        if norm_v < 1e-15:
            continue
        v = [vi / norm_v for vi in v]

        for j in range(sz):
            dot = sum(v[i]*R[i][j] for i in range(sz))
            for i in range(sz):
                R[i][j] -= 2*v[i]*dot

        for i in range(sz):
            dot = sum(Q[i][j]*v[j] for j in range(sz))
            for j in range(sz):
                Q[i][j] -= 2*dot*v[j]

    return Q, R


# -------------------------------------------------------
# QR-алгоритм (§1.2.4, базовая схема без сдвигов):
#
#   A^(0) = A
#   A^(k) = Q^(k)·R^(k)       (QR-разложение)
#   A^(k+1) = R^(k)·Q^(k)     (перемножение в обратном порядке)
#
# Критерий для вещественного λ_m (столбец m):
#   ( Σ_{i=m+1}^{n} (a_im^(k))^2 )^{1/2}  ≤  ε
#
# Критерий для комплексной пары (2×2-блок, столбцы m, m+1):
#   |λ^(k) − λ^(k-1)|  ≤  ε
# -------------------------------------------------------
def run_qr_algorithm(A_in, eps, max_iter=3000):
    nn = len(A_in)
    Ak = copy.deepcopy(A_in)

    print(f"\n{'итер':>5}  {'норма поддиаг':>14}  диагональ")
    print("-" * 70)

    prev_block_eig = [None] * (nn - 1)   # λ^(k-1) для каждого 2×2-блока
    block_done     = [False] * (nn - 1)

    for iteration in range(max_iter):
        sd = math.sqrt(sum(Ak[i][j]**2 for i in range(1, nn) for j in range(i)))

        if iteration <= 10 or iteration % 50 == 0:
            diag = [round(Ak[i][i], 4) for i in range(nn)]
            print(f"{iteration:5d}  {sd:14.2e}  {diag}")

        # --- критерий по отдельным столбцам ---
        col_ok = []
        for m in range(nn):
            col_norm = math.sqrt(sum(Ak[i][m]**2 for i in range(m+1, nn)))
            col_ok.append(col_norm <= eps)

        # --- критерий для 2×2-блоков ---
        for m in range(nn - 1):
            if block_done[m]:
                continue
            aa, ab = Ak[m][m], Ak[m][m+1]
            ba, bb = Ak[m+1][m], Ak[m+1][m+1]
            tr   = aa + bb
            disc = (aa - bb)**2 + 4*ab*ba
            lam  = complex(tr/2, math.sqrt(-disc)/2) if disc < 0 \
                   else complex((tr + math.sqrt(abs(disc)))/2, 0)
            if prev_block_eig[m] is not None and abs(lam - prev_block_eig[m]) <= eps:
                block_done[m] = True
            prev_block_eig[m] = lam

        # --- глобальный останов ---
        if all(col_ok):
            diag = [round(Ak[i][i], 4) for i in range(nn)]
            print(f"{iteration:5d}  {sd:14.2e}  {diag}")
            print(f"\nСошлось за {iteration} итераций (все поддиагональные элементы малы)")
            break

        # попарный обход: каждый «слот» либо вещественный (col_ok), либо комплексная пара (block_done)
        i, all_done = 0, True
        while i < nn:
            if col_ok[i]:
                i += 1
            elif i < nn - 1 and block_done[i]:
                i += 2
            else:
                all_done = False
                break
        if all_done:
            diag = [round(Ak[i][i], 4) for i in range(nn)]
            print(f"{iteration:5d}  {sd:14.2e}  {diag}")
            print(f"\nСошлось за {iteration} итераций")
            break

        Qk, Rk = qr_householder(Ak)
        Ak = [[sum(Rk[i][p]*Qk[p][j] for p in range(nn)) for j in range(nn)]
              for i in range(nn)]
    else:
        print("\nПредупреждение: достигнуто максимальное число итераций")

    return Ak


# -------------------------------------------------------
# Извлечение собственных значений из квазитреугольной матрицы:
#   вещественный λ_m  ← a_mm^(k),  если подстолбец m мал
#   комплексная пара  ← квадратное уравнение из 2×2-блока
# -------------------------------------------------------
def extract_eigenvalues(Ak, eps):
    nn = len(Ak)
    # Алгоритм завершается по критерию сходимости 2×2-блока (|λ^(k)-λ^(k-1)| ≤ ε),
    # при этом поддиагональные элементы могут быть несколько больше самого ε.
    # Используем отдельный, чуть более свободный порог для определения
    # «практически нулевых» поддиагональных элементов:
    ext_eps = max(eps * 100, 1e-4)
    i, eigs = 0, []
    while i < nn:
        if i == nn - 1 or abs(Ak[i+1][i]) < ext_eps:
            print(f"  λ{i+1} = {Ak[i][i]:.6f}")
            eigs.append(Ak[i][i])
            i += 1
        else:
            aa, ab = Ak[i][i], Ak[i][i+1]
            ba, bb = Ak[i+1][i], Ak[i+1][i+1]
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
    return eigs


eps = 1e-6

# =======================================================
# МАТРИЦА 3×3 — вариант 4
# =======================================================
print("=== QR-разложение и QR-алгоритм (вариант 4) ===\n")

n = 3
A = [
    [-4.0, -6.0, -3.0],
    [-1.0,  5.0, -5.0],
    [ 6.0,  2.0,  5.0],
]

print("Матрица 3×3 (по варианту):")
for row in A:
    print(f"  {row}")

# --- QR-разложение ---
print("\n--- QR-разложение методом Хаусхолдера ---")
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

# --- QR-алгоритм 3×3 ---
print("\n--- QR-алгоритм (3×3) ---")
Ak3 = run_qr_algorithm(A, eps)

print("\nФинальная матрица A^(k):")
for row in Ak3:
    print(f"  {[round(x, 6) for x in row]}")

print("\nСобственные значения (3×3):")
eigs3 = extract_eigenvalues(Ak3, eps)

tr_A = sum(A[i][i] for i in range(n))
tr_e = sum(x.real if isinstance(x, complex) else x for x in eigs3)
print(f"\nПроверка: tr(A) = {tr_A},  Σλ = {tr_e:.6f},  погрешность = {abs(tr_A - tr_e):.2e}")


# =======================================================
# МАТРИЦА 5×5 — дополнительный пример с комплексными парами
#
# Структура (с возмущениями):
#   блок [[3,-2],[2,3]]  →  λ = 3 ± 2i
#   блок [[1,-3],[3,1]]  →  λ = 1 ± 3i
#   вещественное значение λ = 5
# =======================================================
print("\n\n" + "="*62)
print("=== QR-алгоритм: матрица 5×5 (2 компл. пары + 1 вещественное) ===\n")

n5 = 5
A5 = [
    [ 3.0, -2.0,  1.0,  0.0,  1.0],
    [ 2.0,  3.0,  0.0,  1.0, -1.0],
    [-1.0,  0.0,  1.0, -3.0,  2.0],
    [ 0.0,  1.0,  3.0,  1.0,  0.0],
    [ 1.0, -1.0,  2.0,  0.0,  5.0],
]

print("Исходная матрица A5 (5×5):")
for row in A5:
    print(f"  {row}")

print("\n--- QR-алгоритм (5×5) ---")
Ak5 = run_qr_algorithm(A5, eps)

print("\nФинальная матрица A5^(k):")
for row in Ak5:
    print(f"  {[round(x, 6) for x in row]}")

print("\nСобственные значения (5×5):")
eigs5 = extract_eigenvalues(Ak5, eps)

tr_A5 = sum(A5[i][i] for i in range(n5))
tr_e5 = sum(x.real if isinstance(x, complex) else x for x in eigs5)
print(f"\nПроверка: tr(A5) = {tr_A5},  Σλ = {tr_e5:.6f},  погрешность = {abs(tr_A5 - tr_e5):.2e}")
