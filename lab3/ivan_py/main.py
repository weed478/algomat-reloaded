import numpy as np
from numpy.linalg import eig, inv


class Solver:
    @staticmethod
    def M1_norm(A: np.array):
        m = A.shape[0]
        n = A.shape[1]
        max_sum = -np.inf
        for j in range(n):
            sum = 0
            for i in range(m):
                sum += np.abs(A[j][i])
            if sum > max_sum:
                max_sum = sum
        return max_sum

    @staticmethod
    def M_inf_norm(A: np.array):
        m = A.shape[0]
        n = A.shape[1]
        max_sum = -np.inf
        for i in range(m):
            sum = 0
            for j in range(n):
                sum += np.abs(A.A[i][j])
            if sum > max_sum:
                max_sum = sum
        return max_sum

    @staticmethod
    def M_2_norm(A: np.matrix):
        H = A.H
        eval, evec = eig(A @ H)
        return np.sqrt(np.max(eval))

    @staticmethod
    def M_1_cond_number(A: np.matrix):
        return Solver.M1_norm(A) * Solver.M1_norm(inv(A))

    @staticmethod
    def M_2_cond_number(A: np.matrix):
        return Solver.M_2_norm(A) * Solver.M_2_norm(inv(A))

    @staticmethod
    def M_inf_cond_number(A: np.matrix):
        return Solver.M_inf_norm(A) * Solver.M_inf_norm(inv(A))


def main():
    X = np.matrix([
        [4, 9, 2],
        [3, 5, 7],
        [8, 1, 6],
    ])
    l2cond = Solver.M_2_cond_number(X)
    print(f'l2 cond number is {l2cond}')


if __name__ == '__main__':
    main()

