import numpy as np
import matplotlib.pylab as pl
import pandas as pd
from IPython.display import display

BIRTHDAY_MONTH = 1
BIRTHDAY_DAY = 1


def main():
    random_matrix = (np.random.rand(BIRTHDAY_MONTH+BIRTHDAY_DAY,
                     BIRTHDAY_MONTH+BIRTHDAY_DAY) * 10).astype(int)
    right_side_vector_T = (np.random.rand(
        BIRTHDAY_MONTH+BIRTHDAY_DAY) * 10).astype(int).T
    A, B = gaussian_elimination_no_pivoting(random_matrix, right_side_vector_T)

    A_pd = pd.DataFrame(A)
    B_pd = pd.DataFrame(np.array([B]))
    display(A_pd)
    display(B_pd)


# [a_11, a_12, a_13]   [x]       [b_11]
# [a_21, a_22, a_33]   [y]   =   [b_21]
# [a_31, a_32, a_33] . [z]       [b_31]
#
def gaussian_elimination_no_pivoting(A: np.array, B: np.array):
    A = A.copy()
    B = B.copy()
    assert len(A.shape) == 2
    assert len(B.shape) == 1

    for i in range(A.shape[1]):
        # always start from the diag element, so as a result it'll yeild low-echelon matrix
        for j in range(i, A.shape[0]):
            el = A[j][i]
            # 1's elemination part
            if i == j:
                if el != 1:
                    A, B = row_divide(A, B, j, i)
            else:
                if el != 0:
                    A, B = rows_eliminate(A, B, j, i)
            display(pd.DataFrame(A))
    return (A, B)


# would return the new matrix with row divided by the element A_iijj
def row_divide(A: np.array, B: np.array, ii, jj):
    n = A[ii][jj]
    for j in range(len(A[ii])):
        A[ii][j] /= n
    B[ii] /= n
    return (A, B)


def rows_eliminate_lu(A: np.array, B: np.array, ii, jj):
    A_copy = A.copy()
    B_copy = B.copy()
    support_row_index = ii + 1 if ii < A.shape[0] - 1 and A_copy[ii + 1][jj] != 0 else jj
    if A_copy[support_row_index][jj] == 0:
        raise ValueError('support value must be non-zero')
    support_row_multiplier = A_copy[ii][jj] / A_copy[support_row_index][jj]

    for j in range(len(A[support_row_index])):
        A_copy[support_row_index][j] *= support_row_multiplier
    B_copy[support_row_index] *= support_row_multiplier
    for j in range(len(A[support_row_index])):
        A[ii][j] -= A_copy[support_row_index][j]
    B[ii] -= B_copy[support_row_index]
    return (A, B, support_row_multiplier)

def swap(A, b, i):
    A_copy = A.copy()
    b_copy = b.copy()
    for j in range(A_copy.shape[0]):
        if j == i:
            continue
        if A_copy[j][i] != 0 and A_copy[i][j] != 0:
            swap_buff = A_copy[j]
            swap_buff_b = b_copy[j]
            A_copy[j] = A_copy[i]
            A_copy[i] = swap_buff
            b_copy[j] = b_copy[i]
            b_copy[i] = swap_buff_b
            return (A_copy, b_copy)  
    raise ValueError('row for pivoting is not found')


def gaussian_elimination_pivoting(A: np.array, b: np.array):
    assert len(A.shape) == 2
    assert len(b.shape) == 1
    for i in range(A.shape[0]):
        if A[i][i] == 0:
            A, b = swap(A, b, i)
    print('----------PIVOTING--------')
    display(pd.DataFrame(A))
    return gaussian_elimination_no_pivoting(A, b)

def LU_no_pivoting(A, b):
    A = A.copy()
    B = B.copy()
    C = np.identity(A.shape[0])
    assert len(A.shape) == 2
    assert len(B.shape) == 1

    for i in range(A.shape[1]):
        # always start from the diag element, so as a result it'll yeild low-echelon matrix
        for j in range(i, A.shape[0]):
            el = A[j][i]
            # 1's elemination part
            if i == j:
                if el != 1:
                    A, B = row_divide(A, B, j, i)
            else:
                if el != 0:
                    A, B, mult = rows_eliminate_lu(A, B, j, i)
                    C[j][i] = mult
            display(pd.DataFrame(A))
    return (A, B, C) # C is an L matrix

if __name__ == '__main__':
    main()