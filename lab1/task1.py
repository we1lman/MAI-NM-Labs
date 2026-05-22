# Лаб 1.1 — LU-разложение с выбором главного элемента
# Вариант 4

import copy

n = 4
A = [
    [-1, -7, -3, -2],
    [-8,  1, -9,  0],
    [ 8,  2, -5, -3],
    [-5,  3,  5, -9],
]
b = [-12, -60, -91, -43]

# -------------------------------------------------------
# LU-разложение: PA = LU
#   P — матрица перестановок (выбор главного элемента)
#   L — нижняя треугольная с l_ii = 1
#   U — верхняя треугольная
#
# k-й шаг: μ_i^(k) = a_ik^(k-1) / a_kk^(k-1)
#           a_ij^(k) = a_ij^(k-1) − μ_i^(k) · a_kj^(k-1)
# -------------------------------------------------------
U = copy.deepcopy(A)
L = [[0.0]*n for _ in range(n)]
perm = list(range(n))   # perm[i] = индекс исходной строки на позиции i
swaps = 0               # число перестановок строк

for k in range(n):
    # выбор строки с максимальным |a_ik| (выбор главного элемента)
    max_row = k
    for i in range(k+1, n):
        if abs(U[i][k]) > abs(U[max_row][k]):
            max_row = i

    if max_row != k:
        U[k], U[max_row] = U[max_row], U[k]
        # меняем только уже вычисленные столбцы L (j = 0..k-1)
        for j in range(k):
            L[k][j], L[max_row][j] = L[max_row][j], L[k][j]
        perm[k], perm[max_row] = perm[max_row], perm[k]
        swaps += 1

    L[k][k] = 1.0
    for i in range(k+1, n):
        # множитель Гаусса: μ_i^(k) = a_ik^(k-1) / a_kk^(k-1)
        L[i][k] = U[i][k] / U[k][k]
        for j in range(k, n):
            U[i][j] -= L[i][k] * U[k][j]

print("=== LU-разложение (вариант 4) ===\n")

print("L:")
for row in L:
    print([round(x, 6) for x in row])

print("\nU:")
for row in U:
    print([round(x, 6) for x in row])

print(f"\nПерестановки ху: perm = {perm},  p = {swaps} (число обменов строк)")
print("(PA = LU, где P — матрица перестановок строк)")

# Проверка: L*U == PA
print("\nL*U:")
for i in range(n):
    row = [round(sum(L[i][k]*U[k][j] for k in range(n)), 4) for j in range(n)]
    print(row)

print("\nPA (строки A в порядке perm):")
for i in range(n):
    print(A[perm[i]])

# -------------------------------------------------------
# Решение СЛАУ методом LU: PA·x = Pb
#   1) Ly = Pb  (прямой ход, нижняя треугольная)
#   2) Ux = y   (обратный ход, верхняя треугольная)
# -------------------------------------------------------
pb = [b[perm[i]] for i in range(n)]

y = [0.0]*n
for i in range(n):
    y[i] = pb[i] - sum(L[i][j]*y[j] for j in range(i))

x = [0.0]*n
for i in range(n-1, -1, -1):
    x[i] = (y[i] - sum(U[i][j]*x[j] for j in range(i+1, n))) / U[i][i]

print("\nРешение СЛАУ:")
for i in range(n):
    print(f"  x{i+1} = {x[i]:.6f}")

print("\nПроверка A·x = b:")
for i in range(n):
    val = sum(A[i][j]*x[j] for j in range(n))
    print(f"  ур.{i+1}: {val:.6f}  (должно быть {b[i]})")

# -------------------------------------------------------
# Определитель: det A = (−1)^p · u_11 · u_22^(1) · ... · u_nn^(n-1)
# -------------------------------------------------------
det = 1.0
for i in range(n):
    det *= U[i][i]
if swaps % 2 == 1:
    det = -det

diag_str = " · ".join(str(round(U[i][i], 4)) for i in range(n))
print(f"\nОпределитель: det A = (−1)^{swaps} · {diag_str} = {det:.4f}")

# -------------------------------------------------------
# Обратная матрица: A·X = E  (n независимых СЛАУ, один LU)
# j-й столбец X = A^{-1} находится из LU·x_j = P·e_j
# -------------------------------------------------------
A_inv = [[0.0]*n for _ in range(n)]

for col in range(n):
    e  = [1.0 if i == col else 0.0 for i in range(n)]
    pe = [e[perm[i]] for i in range(n)]

    yy = [0.0]*n
    for i in range(n):
        yy[i] = pe[i] - sum(L[i][j]*yy[j] for j in range(i))

    xx = [0.0]*n
    for i in range(n-1, -1, -1):
        xx[i] = (yy[i] - sum(U[i][j]*xx[j] for j in range(i+1, n))) / U[i][i]

    for i in range(n):
        A_inv[i][col] = xx[i]

print("\nОбратная матрица A^{-1}:")
for row in A_inv:
    print([round(v, 6) for v in row])

print("\nПроверка A · A^{-1} = E:")
for i in range(n):
    row = [round(sum(A[i][k]*A_inv[k][j] for k in range(n)), 6) for j in range(n)]
    print(row)
