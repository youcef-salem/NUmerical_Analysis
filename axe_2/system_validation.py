from typing import List, Tuple


def validate_square_matrix(A: List[List[float]]) -> Tuple[int, List[List[float]]]:
	if not A or not A[0]:
		raise ValueError("A must be a non-empty square matrix")
	n = len(A)
	if any(len(row) != n for row in A):
		raise ValueError("A must be square")
	A_copy = [row[:] for row in A]
	return n, A_copy


def validate_system(A: List[List[float]], b: List[float]) -> Tuple[int, List[List[float]], List[float]]:
	n, A_copy = validate_square_matrix(A)
	if len(b) != n:
		raise ValueError("b must have the same length as A")
	b_copy = b[:]
	return n, A_copy, b_copy
