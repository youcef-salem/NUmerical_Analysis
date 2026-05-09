from typing import List

from system_validation import validate_system


def gaussian_elimination_pp(A: List[List[float]], b: List[float]) -> List[float]:
	"""Solve Ax = b using Gaussian elimination with partial pivoting."""
	n, A_work, b_work = validate_system(A, b)

	for k in range(n - 1):
		# Partial pivoting: find max in column k
		pivot_row = max(range(k, n), key=lambda i: abs(A_work[i][k]))
		if abs(A_work[pivot_row][k]) == 0:
			raise ValueError("Matrix is singular or nearly singular")
		if pivot_row != k:
			A_work[k], A_work[pivot_row] = A_work[pivot_row], A_work[k]
			b_work[k], b_work[pivot_row] = b_work[pivot_row], b_work[k]

		for i in range(k + 1, n):
			factor = A_work[i][k] / A_work[k][k]
			A_work[i][k] = 0.0
			for j in range(k + 1, n):
				A_work[i][j] -= factor * A_work[k][j]
			b_work[i] -= factor * b_work[k]

	if abs(A_work[n - 1][n - 1]) == 0:
		raise ValueError("Matrix is singular or nearly singular")

	# Back substitution
	x = [0.0] * n
	for i in range(n - 1, -1, -1):
		s = sum(A_work[i][j] * x[j] for j in range(i + 1, n))
		x[i] = (b_work[i] - s) / A_work[i][i]
	return x

