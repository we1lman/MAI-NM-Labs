# Лаб 1.4 — Метод вращений Якоби
# Вариант 4

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
print("Исходная матрица:")
for row in A_orig:
    print(f"  {row}")

A = copy.deepcopy(A_orig)

# матрица собственных векторов (начинаем с единичной)
V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

print(f"\n{'итер':>4}  {'t(A)':>12}  диагональ")
print("-" * 55)

for iteration in range(1000):
    # норма внедиагональных элементов
    t = math.sqrt(sum(A[i][j]**2 for i in range(n) for j in range(n) if i != j))

    print(f"{iteration:4d}  {t:12.2e}  {[round(A[i][i], 8) for i in range(n)]}")

    if t < eps:
        print(f"\nСошлось за {iteration} итераций")
        break

    # ищем максимальный внедиагональный элемент
    p, q = 0, 1
    for i in range(n):
        for j in range(i+1, n):
            if abs(A[i][j]) > abs(A[p][q]):
                p, q = i, j

    # угол вращения
    if abs(A[p][p] - A[q][q]) < 1e-15:
        theta = math.pi / 4
    else:
        theta = 0.5 * math.atan(2*A[p][q] / (A[p][p] - A[q][q]))

    c = math.cos(theta)
    s = math.sin(theta)

    # применяем вращение
    A_new = copy.deepcopy(A)

    for i in range(n):
        if i != p and i != q:
            A_new[i][p] = c*A[i][p] + s*A[i][q]
            A_new[p][i] = A_new[i][p]
            A_new[i][q] = -s*A[i][p] + c*A[i][q]
            A_new[q][i] = A_new[i][q]

    A_new[p][p] = c**2 * A[p][p] + 2*c*s*A[p][q] + s**2 * A[q][q]
    A_new[q][q] = s**2 * A[p][p] - 2*c*s*A[p][q] + c**2 * A[q][q]
    A_new[p][q] = 0.0
    A_new[q][p] = 0.0

    A = A_new

    # обновляем собственные вектора V = V * T
    V_new = copy.deepcopy(V)
    for i in range(n):
        V_new[i][p] = c*V[i][p] + s*V[i][q]
        V_new[i][q] = -s*V[i][p] + c*V[i][q]
    V = V_new

# результаты
print("\nСобственные значения:")
for i in range(n):
    print(f"  λ{i+1} = {A[i][i]:.10f}")

print("\nСобственные векторы (столбцы):")
for j in range(n):
    vec = [V[i][j] for i in range(n)]
    print(f"  v{j+1} = {[round(x, 8) for x in vec]}")

# проверка A*v = λ*v
print("\nПроверка ||A*v - λ*v||:")
for j in range(n):
    v = [V[i][j] for i in range(n)]
    Av = [sum(A_orig[i][k]*v[k] for k in range(n)) for i in range(n)]
    lv = [A[j][j]*v[i] for i in range(n)]
    norm = math.sqrt(sum((Av[i] - lv[i])**2 for i in range(n)))
    print(f"  v{j+1}: {norm:.2e}")

# ортогональность
print("\nОртогональность:")
for i in range(n):
    for j in range(i+1, n):
        dot = sum(V[k][i]*V[k][j] for k in range(n))
        print(f"  <v{i+1}, v{j+1}> = {dot:.2e}")
