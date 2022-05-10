import scipy
import scipy.linalg


def tmp(A):
    L = scipy.linalg.cholesky(A, lower=True)
    U = scipy.linalg.cholesky(A, lower=False)
    return L, U


A = scipy.array([[6, 3, 4, 8], [3, 6, 5, 1], [4, 5, 10, 7], [8, 1, 7, 25]])
L, U = tmp(A)

print(A)
print(L)
print(U)


