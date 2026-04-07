# Лаб 1.3 — Метод простых итераций и метод Зейделя
# Вариант 4

n = 4
A = [
    [26, -9, -8,  8],
    [ 9, -21, -2, 8],
    [-3,  2, -18,  8],
    [ 1, -6,  -1, 11],
]
b = [20, -164, 140, -81]
eps = 1e-6

print("=== Итерации и Зейдель (вариант 4) ===\n")

# проверка диагонального преобладания
print("Диагональное преобладание:")
for i in range(n):
    diag = abs(A[i][i])
    off = sum(abs(A[i][j]) for j in range(n) if j != i)
    print(f"  строка {i+1}: |{A[i][i]}|={diag} vs {off}  {'OK' if diag > off else 'FAIL'}")

# приводим к виду x = Bx + g
B = [[0.0]*n for _ in range(n)]
g = [0.0]*n
for i in range(n):
    for j in range(n):
        if i != j:
            B[i][j] = -A[i][j] / A[i][i]
    g[i] = b[i] / A[i][i]

# норма B
norm_B = max(sum(abs(B[i][j]) for j in range(n)) for i in range(n))
print(f"\n||B|| = {norm_B:.6f} ({'< 1, сходится' if norm_B < 1 else '>= 1, проблема'})")

# --- Метод простых итераций ---
print("\n--- Метод простых итераций ---")
x = g[:]
for k in range(1, 10001):
    x_new = [g[i] + sum(B[i][j]*x[j] for j in range(n)) for i in range(n)]
    diff = max(abs(x_new[i] - x[i]) for i in range(n))
    error = norm_B / (1 - norm_B) * diff if norm_B < 1 else diff

    if k <= 5 or error < eps:
        print(f"  итерация {k:4d}: diff={diff:.2e}, оценка={error:.2e}")

    if error < eps:
        x = x_new
        print(f"\nСошлось за {k} итераций")
        break
    x = x_new

iter_count = k

print("Решение:")
for i in range(n):
    print(f"  x{i+1} = {x[i]:.6f}")

print("Невязка:", [round(sum(A[i][j]*x[j] for j in range(n)) - b[i], 8) for i in range(n)])

# --- Метод Зейделя ---
print("\n--- Метод Зейделя ---")
x = [b[i] / A[i][i] for i in range(n)]

for k in range(1, 10001):
    x_old = x[:]
    for i in range(n):
        s = sum(A[i][j]*x[j] for j in range(n) if j != i)
        x[i] = (b[i] - s) / A[i][i]

    diff = max(abs(x[i] - x_old[i]) for i in range(n))

    if k <= 5 or diff < eps:
        print(f"  итерация {k:4d}: diff={diff:.2e}")

    if diff < eps:
        print(f"\nСошлось за {k} итераций")
        break

seidel_count = k

print("Решение:")
for i in range(n):
    print(f"  x{i+1} = {x[i]:.6f}")

print("Невязка:", [round(sum(A[i][j]*x[j] for j in range(n)) - b[i], 8) for i in range(n)])

# сравнение
print(f"\nИтого: простые итерации — {iter_count}, Зейдель — {seidel_count}")
print(f"Зейдель быстрее в {iter_count/seidel_count:.1f} раз")
