n = 4
A = [
    [26, -9, -8,  8],
    [ 9, -21, -2,  8],
    [-3,   2, -18,  8],
    [ 1,  -6,  -1, 11],
]
b = [20, -164, 140, -81]
eps = 1e-6

print("=== Метод простых итераций и Зейделя (вариант 4) ===\n")

# -------------------------------------------------------
# Проверка диагонального преобладания (достаточное условие
# сходимости метода Якоби, §1.1.4):
#   |a_ii| > Σ_{j≠i} |a_ij|  для всех i
# -------------------------------------------------------
print("Диагональное преобладание:")
for i in range(n):
    diag = abs(A[i][i])
    off  = sum(abs(A[i][j]) for j in range(n) if j != i)
    print(f"  строка {i+1}: |{A[i][i]}| = {diag}  vs  {off}  "
          f"{'OK' if diag > off else 'FAIL'}")

# -------------------------------------------------------
# Приведение к эквивалентному виду  x = Bx + g  (форм. 1.18):
#   β_i  = b_i / a_ii
#   α_ij = -a_ij / a_ii  (j ≠ i),   α_ii = 0
# -------------------------------------------------------
B = [[0.0]*n for _ in range(n)]
g = [0.0]*n
for i in range(n):
    for j in range(n):
        if i != j:
            B[i][j] = -A[i][j] / A[i][i]
    g[i] = b[i] / A[i][i]

# ||B||_c (строчная бесконечная норма)
norm_B = max(sum(abs(B[i][j]) for j in range(n)) for i in range(n))
print(f"\n||α||_c = {norm_B:.6f}  "
      f"({'< 1, достаточное условие сходимости выполнено' if norm_B < 1 else '>= 1'})")

# Строго верхнетреугольная часть матрицы α (матрица C для формулы Зейделя)
# C[i][j] = B[i][j]  для j > i,  иначе 0
C = [[B[i][j] if j > i else 0.0 for j in range(n)] for i in range(n)]
norm_C = max(sum(abs(C[i][j]) for j in range(n)) for i in range(n))
print(f"||C||_c  = {norm_C:.6f}  "
      f"(C — верхнетреугольная часть α, нужна для оценки Зейделя)")

# -------------------------------------------------------
# МЕТОД ПРОСТЫХ ИТЕРАЦИЙ (метод Якоби, §1.1.4, форм. 1.19)
#
# x^(0) = β = g
# x^(k) = β + α · x^(k-1)
#
# Оценка погрешности (1.20):
#   ε^(k) = ||α|| / (1 − ||α||) · ||x^(k) − x^(k-1)||
#
# Останов: ε^(k) < ε
# -------------------------------------------------------
print("\n--- Метод простых итераций ---")
x = g[:]
iter_count = 0

for k in range(1, 10001):
    x_new = [g[i] + sum(B[i][j]*x[j] for j in range(n)) for i in range(n)]
    diff  = max(abs(x_new[i] - x[i]) for i in range(n))
    error = norm_B / (1 - norm_B) * diff   # формула (1.20)

    if k <= 5 or error < eps:
        print(f"  итерация {k:4d}: ||Δx|| = {diff:.2e},  ε^(k) = {error:.2e}")

    x = x_new
    if error < eps:
        iter_count = k
        print(f"\nСошлось за {k} итераций")
        break

print("Решение:")
for i in range(n):
    print(f"  x{i+1} = {x[i]:.6f}")

print("Невязка A·x − b:",
      [round(sum(A[i][j]*x[j] for j in range(n)) - b[i], 8) for i in range(n)])

# -------------------------------------------------------
# МЕТОД ЗЕЙДЕЛЯ (§1.1.4)
#
# x^(0) = β = g
# На k-й итерации при вычислении x_i^(k+1) используются
# уже обновлённые x_1^(k+1),...,x_{i-1}^(k+1) (в отличие от Якоби).
#
# Оценка погрешности (Замечание §1.1.4):
#   ε^(k) = ||C|| / (1 − ||α||) · ||x^(k) − x^(k-1)||
#
# где C — строго верхнетреугольная часть матрицы α.
# -------------------------------------------------------
print("\n--- Метод Зейделя ---")
x = g[:]
seidel_count = 0

for k in range(1, 10001):
    x_old = x[:]
    for i in range(n):
        # для j < i используются уже обновлённые x[j]^(k+1)
        # для j > i используются старые x[j]^(k)  (x не обновлён для j>i)
        s    = sum(A[i][j]*x[j] for j in range(n) if j != i)
        x[i] = (b[i] - s) / A[i][i]

    diff  = max(abs(x[i] - x_old[i]) for i in range(n))
    error = norm_C / (1 - norm_B) * diff   # Зейдель: ||C||/(1−||α||)·||Δx||

    if k <= 5 or error < eps:
        print(f"  итерация {k:4d}: ||Δx|| = {diff:.2e},  ε^(k) = {error:.2e}")

    if error < eps:
        seidel_count = k
        print(f"\nСошлось за {k} итераций")
        break

print("Решение:")
for i in range(n):
    print(f"  x{i+1} = {x[i]:.6f}")

print("Невязка A·x − b:",
      [round(sum(A[i][j]*x[j] for j in range(n)) - b[i], 8) for i in range(n)])

# -------------------------------------------------------
# Сравнение
# -------------------------------------------------------
print(f"\nИтого: простые итерации — {iter_count},  Зейдель — {seidel_count}")
if iter_count and seidel_count:
    print(f"Зейдель быстрее в {iter_count / seidel_count:.1f} раз")
