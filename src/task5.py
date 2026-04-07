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
eps = 1e-10

print("=== QR-разложение и QR-алгоритм (вариант 4) ===\n")
print("Исходная матрица:")
for row in A:
    print(f"  {row}")


def qr_householder(M):
    """QR-разложение методом отражений Хаусхолдера"""
    sz = len(M)
    R = copy.deepcopy(M)
    Q = [[1.0 if i == j else 0.0 for j in range(sz)] for i in range(sz)]

    for k in range(sz - 1):
        # столбец подматрицы
        x = [R[i][k] if i >= k else 0.0 for i in range(sz)]
        norm_x = math.sqrt(sum(x[i]**2 for i in range(k, sz)))
        if norm_x < 1e-15:
            continue

        # вектор Хаусхолдера
        v = [0.0]*sz
        v[k] = x[k] + math.copysign(norm_x, x[k])
        for i in range(k+1, sz):
            v[i] = x[i]

        norm_v = math.sqrt(sum(vi**2 for vi in v))
        if norm_v < 1e-15:
            continue
        v = [vi / norm_v for vi in v]

        # R = (I - 2vv^T) * R
        for j in range(sz):
            dot = sum(v[i]*R[i][j] for i in range(sz))
            for i in range(sz):
                R[i][j] -= 2*v[i]*dot

        # Q = Q * (I - 2vv^T)
        for i in range(sz):
            dot = sum(Q[i][j]*v[j] for j in range(sz))
            for j in range(sz):
                Q[i][j] -= 2*dot*v[j]

    return Q, R


# ======= Часть 1: QR-разложение =======
print("\n--- QR-разложение ---")
Q, R = qr_householder(A)

print("\nQ:")
for row in Q:
    print(f"  {[round(x, 6) for x in row]}")

print("\nR:")
for row in R:
    print(f"  {[round(x, 6) for x in row]}")

# проверка Q*R
print("\nQ*R (должно быть = A):")
for i in range(n):
    row = [round(sum(Q[i][k]*R[k][j] for k in range(n)), 4) for j in range(n)]
    print(f"  {row}")

# Q^T * Q = E
print("\nQ^T*Q (должно быть = E):")
for i in range(n):
    row = [round(sum(Q[k][i]*Q[k][j] for k in range(n)), 6) for j in range(n)]
    print(f"  {row}")


# ======= Часть 2: QR-алгоритм =======
print("\n--- QR-алгоритм ---")

Ak = copy.deepcopy(A)

print(f"\n{'итер':>4}  {'норма поддиаг':>14}  диагональ")
print("-" * 60)

for iteration in range(1000):
    # норма поддиагональных элементов
    sd = math.sqrt(sum(Ak[i][j]**2 for i in range(n) for j in range(i)))

    if iteration <= 10 or iteration % 50 == 0:
        diag = [round(Ak[i][i], 6) for i in range(n)]
        print(f"{iteration:4d}  {sd:14.2e}  {diag}")

    # квазитреугольность: все элементы ниже первой поддиагонали ~0
    quasi_ok = all(abs(Ak[i][j]) < eps for i in range(2, n) for j in range(i-1))
    if quasi_ok:
        diag = [round(Ak[i][i], 6) for i in range(n)]
        print(f"{iteration:4d}  {sd:14.2e}  {diag}")
        print(f"\nСошлось за {iteration} итераций")
        break

    # сдвиг Уилкинсона (по нижнему 2x2 блоку)
    a11, a12 = Ak[n-2][n-2], Ak[n-2][n-1]
    a21, a22 = Ak[n-1][n-2], Ak[n-1][n-1]
    trace = a11 + a22
    disc = (a11 - a22)**2 + 4*a12*a21

    if disc >= 0:
        sq = math.sqrt(disc)
        mu1 = (trace + sq) / 2
        mu2 = (trace - sq) / 2
        mu = mu1 if abs(mu1 - a22) < abs(mu2 - a22) else mu2
    else:
        mu = trace / 2  # вещественная часть для комплексных

    # A - mu*I
    Ak_s = copy.deepcopy(Ak)
    for i in range(n):
        Ak_s[i][i] -= mu

    Qk, Rk = qr_householder(Ak_s)

    # A_new = R*Q + mu*I
    Ak = [[sum(Rk[i][k]*Qk[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    for i in range(n):
        Ak[i][i] += mu

# финальная матрица
print("\nФинальная матрица:")
for row in Ak:
    print(f"  {[round(x, 6) for x in row]}")

# извлекаем собственные значения
print("\nСобственные значения:")
i = 0
eigs = []
while i < n:
    if i == n-1 or abs(Ak[i+1][i]) < eps:
        print(f"  λ{i+1} = {Ak[i][i]:.10f}")
        eigs.append(Ak[i][i])
        i += 1
    else:
        # 2x2 блок
        aa, bb = Ak[i][i], Ak[i][i+1]
        cc, dd = Ak[i+1][i], Ak[i+1][i+1]
        tr = aa + dd
        det = aa*dd - bb*cc
        disc = tr**2 - 4*det

        if disc >= 0:
            sq = math.sqrt(disc)
            print(f"  λ{i+1} = {(tr+sq)/2:.10f}")
            print(f"  λ{i+2} = {(tr-sq)/2:.10f}")
            eigs += [(tr+sq)/2, (tr-sq)/2]
        else:
            sq = math.sqrt(-disc)
            print(f"  λ{i+1} = {tr/2:.10f} + {sq/2:.10f}i")
            print(f"  λ{i+2} = {tr/2:.10f} - {sq/2:.10f}i")
            eigs += [complex(tr/2, sq/2), complex(tr/2, -sq/2)]
        i += 2

# проверка по следу
tr_A = sum(A[i][i] for i in range(n))
tr_eig = sum(x.real if isinstance(x, complex) else x for x in eigs)
print(f"\nПроверка: tr(A) = {tr_A}, Σλ = {tr_eig:.10f}, разница = {abs(tr_A - tr_eig):.2e}")
