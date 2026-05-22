import math
import copy

n = 3
A_orig = [
    [ 8.0,  2.0, -1.0],
    [ 2.0, -5.0, -8.0],
    [-1.0, -8.0, -5.0],
]
eps = 1e-8

print("=== Метод вращений Якоби (вариант 4) ===\n")
print("Исходная матрица A (симметричная):")
for row in A_orig:
    print(f"  {row}")

# -------------------------------------------------------
# Алгоритм §1.2.2:
#
# 1. Выбрать максимальный по модулю внедиагональный элемент
#    a_ij^(k),  |a_ij^(k)| = max_{l<m} |a_lm^(k)|
#
# 2. Вычислить угол вращения φ^(k):
#    φ^(k) = (1/2) · arctg(2·a_ij / (a_ii − a_jj))
#    если a_ii = a_jj → φ = π/4
#
# 3. Построить матрицу вращения U^(k): единичная матрица,
#    в которой u_ii = u_jj = cos φ, u_ij = −sin φ, u_ji = sin φ
#
# 4. A^(k+1) = U^(k)^T · A^(k) · U^(k)
#
# Критерий останова:
#    t(A^(k)) = ( Σ_{l<m} (a_lm^(k))^2 )^{1/2} < ε
#
# Матрица собственных векторов накапливается:
#    V = U^(0) · U^(1) · ... · U^(k)
# -------------------------------------------------------

A = copy.deepcopy(A_orig)
# V — матрица накопленных вращений (начинаем с единичной)
V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

print(f"\n{'итер':>4}  {'t(A)':>12}  диагональ")
print("-" * 56)

for iteration in range(1000):
    # норма внедиагональных элементов
    t = math.sqrt(sum(A[i][j]**2 for i in range(n) for j in range(n) if i != j))

    print(f"{iteration:4d}  {t:12.2e}  {[round(A[i][i], 8) for i in range(n)]}")

    if t < eps:
        print(f"\nСошлось за {iteration} итераций")
        break

    # шаг 1: максимальный по модулю внедиагональный элемент a_pq
    p, q = 0, 1
    for i in range(n):
        for j in range(i+1, n):
            if abs(A[i][j]) > abs(A[p][q]):
                p, q = i, j

    # шаг 2: угол вращения φ^(k)
    if abs(A[p][p] - A[q][q]) < 1e-15:
        phi = math.pi / 4
    else:
        phi = 0.5 * math.atan(2*A[p][q] / (A[p][p] - A[q][q]))

    c = math.cos(phi)
    s = math.sin(phi)

    # шаг 3–4: A^(k+1) = U^T · A^(k) · U
    A_new = copy.deepcopy(A)

    for i in range(n):
        if i != p and i != q:
            A_new[i][p] = c*A[i][p] + s*A[i][q]
            A_new[p][i] = A_new[i][p]
            A_new[i][q] = -s*A[i][p] + c*A[i][q]
            A_new[q][i] = A_new[i][q]

    A_new[p][p] =  c**2 * A[p][p] + 2*c*s * A[p][q] + s**2 * A[q][q]
    A_new[q][q] =  s**2 * A[p][p] - 2*c*s * A[p][q] + c**2 * A[q][q]
    A_new[p][q] = 0.0
    A_new[q][p] = 0.0

    A = A_new

    # обновление матрицы собственных векторов V := V · U^(k)
    V_new = copy.deepcopy(V)
    for i in range(n):
        V_new[i][p] =  c*V[i][p] + s*V[i][q]
        V_new[i][q] = -s*V[i][p] + c*V[i][q]
    V = V_new

# -------------------------------------------------------
# Результаты: λ_j ≈ a_jj^(k),  x^j — j-й столбец V
# -------------------------------------------------------
print("\nСобственные значения (диагональ финальной A^(k)):")
for i in range(n):
    print(f"  λ{i+1} = {A[i][i]:.10f}")

print("\nСобственные векторы (столбцы матрицы V = U^(0)·U^(1)·...·U^(k)):")
for j in range(n):
    vec = [V[i][j] for i in range(n)]
    print(f"  v{j+1} = {[round(v, 8) for v in vec]}")

# -------------------------------------------------------
# Проверка: ||A·v_j − λ_j·v_j||
# -------------------------------------------------------
print("\nПроверка ||A·v − λ·v||:")
for j in range(n):
    v  = [V[i][j] for i in range(n)]
    Av = [sum(A_orig[i][k]*v[k] for k in range(n)) for i in range(n)]
    lv = [A[j][j]*v[i] for i in range(n)]
    norm = math.sqrt(sum((Av[i] - lv[i])**2 for i in range(n)))
    print(f"  v{j+1}: {norm:.2e}")

# -------------------------------------------------------
# Проверка ортогональности собственных векторов
# -------------------------------------------------------
print("\nОртогональность собственных векторов <v_i, v_j>:")
for i in range(n):
    for j in range(i+1, n):
        dot = sum(V[k][i]*V[k][j] for k in range(n))
        print(f"  <v{i+1}, v{j+1}> = {dot:.2e}")
