import numpy as np


def load_mtx(filename):
    with open(filename) as f:
        lines = f.readlines()
        return np.array([np.fromstring(i, sep=' ', dtype=np.float64) for i in lines[:-1]]), np.fromstring(lines[-1], sep=' ', dtype=np.float64)

def get_det(matrix: np.ndarray, eps=1e-12):
    matrix = matrix.copy().astype(np.float64)

    n, m = matrix.shape

    if n != m:
        raise ValueError("Определитель можно считать только для квадратной матрицы")

    sign = 1.0

    for col in range(n):
        pivot_row = col

        while pivot_row < n and abs(matrix[pivot_row, col]) < eps:
            pivot_row += 1

        if pivot_row == n:
            return 0.0

        if pivot_row != col:
            matrix[[col, pivot_row]] = matrix[[pivot_row, col]]
            sign *= -1

        pivot = matrix[col, col]

        for row in range(col + 1, n):
            factor = matrix[row, col] / pivot
            matrix[row, col:] -= factor * matrix[col, col:]

    det = sign * np.prod(np.diag(matrix))
    return det


def cramer(A: np.ndarray, b: np.ndarray, eps=1e-3):
    A = A.copy()
    b = b.copy()

    n, m = A.shape

    if n != m:
        raise ValueError("Матрица A должна быть квадратной")

    if b.shape[0] != n:
        raise ValueError("Размерность b не совпадает с A")

    det_A = get_det(A.copy(), eps)

    if abs(det_A) < eps:
        raise ValueError("Определитель равен 0, система не имеет единственного решения")

    x = np.zeros(n)

    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = get_det(Ai, eps) / det_A

    return x