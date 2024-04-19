import numpy as np
from numpy.linalg import eig, inv


class Solver:
    @staticmethod
    def M1_norm(A: np.matrix):
        A = np.asarray(A)
        m = A.shape[0]
        n = A.shape[1]
        max_sum = -np.inf
        for j in range(n):
            sum = 0
            for i in range(m):
                sum += np.abs(A[i][j])
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

    @staticmethod
    def M_p_norm(A: np.array, p):
        evals, evecs = eig(A)
        return np.sum(evals.real ** p) ** (1/p)
    
    @staticmethod
    def M_p_cond_number(A: np.array, p):
        return Solver.M_p_norm(A, p) * Solver.M_p_norm(inv(A), p)

    @staticmethod
    def eigenvalue(A, v):
        if A.shape[1] != v.shape[0]:
                v = v.T
        val = np.dot(A,v) / v
        return val[0]
    
    # @staticmethod
    # def svd(A):
    #     threshold = 0.001
    #     prev_eig = 0
    #     cur_eig = np.random.rand(A.shape[1])
    #     prev_eigval = 0
    #     cur_eigval = 0
    #     while(True):
    #         if A.shape[1] != cur_eig.shape[0]:
    #             cur_eig = cur_eig.T
    #         it_eig = np.dot(A, cur_eig)
    #         norm = Solver.M_2_norm(it_eig)
    #         it_eig /= norm
    #         it_eigval = Solver.eigenvalue(A, it_eig)
    #         if np.abs(it_eigval - cur_eigval) < 0.001:
    #             break
    #         prev_eig = cur_eig
    #         prev_eigval = cur_eigval
    #         cur_eig = it_eig
    #         cur_eigval = it_eigval
    #     return cur_eig, cur_eigval
    @staticmethod
    def svd(A):
        return np.linalg.svd(A)

def main():
    X = np.array([
        [4, 9, 2],
        [3, 5, 7],
        [8, 1, 6],
    ], dtype=np.float32)
    # svd_res = Solver.svd(X)
    # print(f'M1 norm={Solver.M1_norm(X)}; M1 cond number={Solver.M_1_cond_number(X)}')
    # print(f'M2 norm={Solver.M_2_norm(X)}; M2 cond number={Solver.M_2_cond_number(X)}')
    # print(f'Minf norm={Solver.M_inf_norm(X)}; Minf cond number={Solver.M_inf_cond_number(X)}')
    print(f'Mp norm for p=15 = {Solver.M_p_norm(X, 2.0)}; Mp condition number={Solver.M_p_cond_number(X, 2.0)}')
    # print(f'SVD: U is {svd_res.U}\nSIGMA is {svd_res.S}\nV^t is {svd_res.Vh}')

    # print(np.linalg.norm(X, ord=15))


if __name__ == '__main__':
    main()
