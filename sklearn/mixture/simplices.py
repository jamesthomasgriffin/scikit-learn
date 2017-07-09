import numpy as np

class Simplices:
    def __init__(self, A):
        self.n_simplices = len(A)
        self.max_dim = max(len(S) for S in A) - 1
        self.simplices_by_dim = [set() for _ in range(self.max_dim + 1)]
        for S in A:
            self.simplices_by_dim[len(S)-1].add(S)

    def simplices(self, minD=0, maxD=None):
        for i in range(minD, max(maxD, self.max_dim)+1):
            for S in self.simplices_by_dim[i]:
                yield S

    def k_simplices_as_matrix(self, k, n_vertices):
        n_ksimplices = len(self.simplices_by_dim[k])
        n_verts_per_simplex = k + 1
        A = np.zeros(
            shape=(n_ksimplices, n_verts_per_simplex, n_vertices),
            dtype=float
            )
        for i, S in enumerate(self.simplices_by_dim[k]):
            for j, v in enumerate(S):
                A[i, j, v] = 1.0
        return A
