import numpy as np


'''## PART A
N = 4
HW = [[-1, 0, 0, 1] , [0, -1, 0, 1]]
X = np.array(HW) # build array from mean subtracted from points

cov = (1 / (N -1)) * X @ X.T # complete covariance calulation
#cov = np.round(cov, 2) # format to 2 decimal points

print('\n covariance matrix is: ')
print(cov)

eigenvalues, eigenvectors = np.linalg.eig(cov) # find eigenvalues and eigenvectors

print('\n eigenvalues are: ')
print(eigenvalues)

print('\n eigenvectors are: ')
print(eigenvectors)

## PART B
w1 = eigenvectors[:, 0]

var = w1.T @ cov @ w1
print('\n variance is: ')
print(var)

matrix = np.array([[1, 2, 3, 4], [-1, 1, 4, 4]])
row_norms = np.linalg.norm(matrix, axis=1)
print(f"Matrix:\n{matrix}")
print(f"L2 Norm of each row: {row_norms}")'''

#quiz
X = np.array([[0, 4, -1], [0, 0, 0]])
print(X)

m = np.mean(X, axis = 1)
print(m)

S = np.cov(X)
print(S)

print(X @ X.T)