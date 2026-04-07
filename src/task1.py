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

# --- LU-разложение ---
U = copy.deepcopy(A)
L = [[0.0]*n for _ in range(n)]
perm = list(range(n))
swaps = 0

for k in range(n):
    # выбор главного элемента по столбцу
    max_row = k
    for i in range(k+1, n):
        if abs(U[i][k]) > abs(U[max_row][k]):
            max_row = i

    if max_row != k:
        U[k], U[max_row] = U[max_row], U[k]
        L[k], L[max_row] = L[max_row], L[k]
        perm[k], perm[max_row] = perm[max_row], perm[k]
        swaps += 1

    L[k][k] = 1.0
    for i in range(k+1, n):
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

print(f"\nПерестановки: {perm}, обменов: {swaps}")

# проверка L*U = PA
print("\nL*U:")
for i in range(n):
    row = []
    for j in range(n):
        s = sum(L[i][k] * U[k][j] for k in range(n))
        row.append(round(s, 4))
    print(row)

# --- Решение СЛАУ ---
# Ly = Pb
pb = [b[perm[i]] for i in range(n)]
y = [0.0]*n
for i in range(n):
    y[i] = pb[i] - sum(L[i][j]*y[j] for j in range(i))

# Ux = y
x = [0.0]*n
for i in range(n-1, -1, -1):
    x[i] = (y[i] - sum(U[i][j]*x[j] for j in range(i+1, n))) / U[i][i]

print("\nРешение:")
for i in range(n):
    print(f"  x{i+1} = {x[i]:.6f}")

# проверка
print("\nПроверка A*x:")
for i in range(n):
    val = sum(A[i][j]*x[j] for j in range(n))
    print(f"  уравнение {i+1}: {val:.6f} (должно быть {b[i]})")

# --- Определитель ---
det = 1.0
for i in range(n):
    det *= U[i][i]
if swaps % 2 == 1:
    det = -det
print(f"\nОпределитель: {det:.4f}")

# --- Обратная матрица ---
A_inv = [[0.0]*n for _ in range(n)]

for col in range(n):
    # решаем A * x_col = e_col
    e = [0.0]*n
    e[col] = 1.0
    pe = [e[perm[i]] for i in range(n)]

    yy = [0.0]*n
    for i in range(n):
        yy[i] = pe[i] - sum(L[i][j]*yy[j] for j in range(i))

    xx = [0.0]*n
    for i in range(n-1, -1, -1):
        xx[i] = (yy[i] - sum(U[i][j]*xx[j] for j in range(i+1, n))) / U[i][i]

    for i in range(n):
        A_inv[i][col] = xx[i]

print("\nОбратная матрица:")
for row in A_inv:
    print([round(x, 6) for x in row])

# проверка A * A^-1 = E
print("\nA * A^(-1):")
for i in range(n):
    row = []
    for j in range(n):
        s = sum(A[i][k] * A_inv[k][j] for k in range(n))
        row.append(round(s, 6))
    print(row)
