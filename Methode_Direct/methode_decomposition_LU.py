from typing import List, Tuple

from system_validation import validate_square_matrix, validate_system


def lu_crout_pp(A: List[List[float]]) -> Tuple[List[List[float]], List[List[float]], List[int]]:
	"""Crout LU decomposition with partial pivoting.

	Returns L, U, and permutation vector p such that P*A = L*U.
	U has unit diagonal.
	"""
	n, A_work = validate_square_matrix(A)
	L = [[0.0] * n for _ in range(n)]
	U = [[0.0] * n for _ in range(n)]
	p = list(range(n))

	for k in range(n):
		# Partial pivoting on column k using current residuals
		def residual(i: int) -> float:
			s = sum(L[i][j] * U[j][k] for j in range(k))
			return A_work[i][k] - s

		pivot_row = max(range(k, n), key=lambda i: abs(residual(i)))
		if abs(residual(pivot_row)) == 0:
			raise ValueError("Matrix is singular or nearly singular")
		if pivot_row != k:
			A_work[k], A_work[pivot_row] = A_work[pivot_row], A_work[k]
			p[k], p[pivot_row] = p[pivot_row], p[k]
			if k > 0:
				L[k][:k], L[pivot_row][:k] = L[pivot_row][:k], L[k][:k]

		# Compute L column k
		for i in range(k, n):
			s = sum(L[i][j] * U[j][k] for j in range(k))
			L[i][k] = A_work[i][k] - s

		if abs(L[k][k]) == 0:
			raise ValueError("Matrix is singular or nearly singular")

		# Compute U row k (unit diagonal)
		U[k][k] = 1.0
		for j in range(k + 1, n):
			s = sum(L[k][m] * U[m][j] for m in range(k))
			U[k][j] = (A_work[k][j] - s) / L[k][k]

	return L, U, p


def forward_substitution(L: List[List[float]], b: List[float]) -> List[float]:
	n = len(L)
	y = [0.0] * n
	for i in range(n):
		s = sum(L[i][j] * y[j] for j in range(i))
		if abs(L[i][i]) == 0:
			raise ValueError("Matrix is singular or nearly singular")
		y[i] = (b[i] - s) / L[i][i]
	return y


def back_substitution(U: List[List[float]], y: List[float]) -> List[float]:
	n = len(U)
	x = [0.0] * n
	for i in range(n - 1, -1, -1):
		s = sum(U[i][j] * x[j] for j in range(i + 1, n))
		if abs(U[i][i]) == 0:
			raise ValueError("Matrix is singular or nearly singular")
		x[i] = (y[i] - s) / U[i][i]
	return x


def solve_lu_crout_pp(A: List[List[float]], b: List[float]) -> List[float]:
	"""Solve Ax = b using Crout LU decomposition with partial pivoting."""
	n, A_work, b_work = validate_system(A, b)
	L, U, p = lu_crout_pp(A_work)
	b_perm = [b_work[i] for i in p]
	y = forward_substitution(L, b_perm)
	x = back_substitution(U, y)
	return x

