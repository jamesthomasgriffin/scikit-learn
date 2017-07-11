import numpy as np
from itertools import combinations, product

def _check_simplices(simplices, n_vertices):
    """This checks that the set of simplices is contained in the power set
    of [0, ..., n_vertices-1].
    """
    vertices = set()
    for S in simplices:
        vertices = vertices.union(S)
    if not vertices.issubset(set(range(n_vertices))):
        counter_example = vertices.difference(set(range(n_vertices))).pop()
        raise ValueError("A simplex contains a value: {}, which is not in "
                         "the vertex set {{0, ..., {}}}".format(
                         counter_example, n_vertices-1))
        return 0
    else:
        return 1

def complete_simplices(n_vertices, dimensions=None):
    """The complete (flag) complex.

    Parameters
    ----------
    n_vertices: integer
        The number of vertices.
    dimensions: list
        Specifies the dimensions of simplices to be given, defaults to
        all dimensions if None.

    Returns
    -------
    simplices: set of simplices
        The simplices.
    n_vertices: integer
        The number of vertices.
    """

    if dimensions is None:
        dimensions = range(n_vertices+1)
    simplices = set()
    for k in dimensions:
        simplices = simplices.union(combinations(range(n_vertices), k+1))
    return simplices, n_vertices

def grid_simplices(m, n=None):
    """An mxn grid consisting of points, edges and triangles.

    Parameters
    ----------
    m : integer
        The width of the grid, and height if n is not specified.
    n : integer
        The height of the grid.

    Returns
    -------
    simplices: set of simplices
        The simplices of the grid.
    n_vertices: integer
        The number of vertices.
    """
    n = n or m
    simplices = set()
    n_vertices = (n+1) * (m+1)
    simplices = simplices.union(
        [(v,) for v in range(n_vertices)])
    def v_bl():
        "An iterator giving vertices which are to the bottom left of a square."
        for a, b in product(range(m), range(n)):
            yield b*(m+1) + a
    def v_b():
        "An iterator giving vertices which are to the bottom of a line."
        for a, b in product(range(m), range(n+1)):
            yield b*(m+1) + a
    def v_l():
        "An iterator giving vertices which are to the left of a line."
        for a, b in product(range(m+1), range(n)):
            yield b*(m+1) + a
    simplices = simplices.union([(v, v+1) for v in v_b()])
    simplices = simplices.union([(v, v+(m+1)) for v in v_l()])
    simplices = simplices.union([(v, v+1+(m+1)) for v in v_bl()])
    simplices = simplices.union([(v, v+1, v+1+(m+1)) for v in v_bl()])
    simplices = simplices.union([(v, v+(m+1), v+1+(m+1)) for v in v_bl()])
    return simplices, n_vertices

class Simplices:
    def __init__(self, A, n_vertices):
        _check_simplices(A, n_vertices)
        self.n_vertices = n_vertices
        self.n_simplices = len(A)
        self.max_dim = max(len(S) for S in A) - 1
        self.simplices_by_dim = [[] for _ in range(self.max_dim + 1)]
        for S in A:
            self.simplices_by_dim[len(S)-1].append(frozenset(S))
        self.simplices_in_dim = [len(simps) for simps in self.simplices_by_dim]

    def simplices(self, minD=None, maxD=None):
        """Iterator for the simplices."""
        minD = minD or 0
        maxD = maxD or self.max_dim
        for i in range(minD, maxD+1):
            for S in self.simplices_by_dim[i]:
                yield S

    def k_simplices_as_matrix(self, k):
        """Returns one-hot matrices representing each k-simplex.

        Parameters
        ----------
        k : integer

        Returns
        -------
        M : array-like, shape (n_ksimplices, k+1, n_vertices)
            For each simplex, and each vertex of the simplex a basis vector
            representing that simplex.
        """
        n_ksimplices = len(self.simplices_by_dim[k])
        n_verts_per_simplex = k + 1
        M = np.zeros(
            shape=(n_ksimplices, n_verts_per_simplex, self.n_vertices))
        for i, S in enumerate(self.simplices_by_dim[k]):
            for j, v in enumerate(S):
               M[i, j, v] = 1.0
        return M

def _random_from_simplex(dim=2, samples=1, random_state=np.random.RandomState()):
    """Returns an array of random points from the simplex of dimension dim
    in R^{dim+1}, drawn using the uniform distribution.

    Parameters
    ----------
    dim : integer
        The dimension of the simplex.
    samples: integer
        The number of samples.
    random_state : RandomState
        A random number generator instance.


    Returns
    -------
    M : array-like, shape (samples, dim+1)
        The samples.
    """

    M = np.diff(np.pad(
            np.sort(random_state.random_sample((samples, dim)), axis=1), ((0,0),(1,1)),
            mode='constant', constant_values=((0,0),(0,1))
            ), axis=1)
    return M

class SimplicesRV(Simplices):
    """A class for representing a set of weighted simplices and generating
    a sample from its geometric realisation."""
    def __init__(self, A, n_vertices, weights=None):
        """The constructor takes a set of simplices, which can be any
        list-like of list-likes representing a subset of the power set of
        {1, ..., n_vertices-1}; along with a dictionary of weights.

        Parameters
        ----------
        A : list-like of list-likes
            The simplices.
        n_vertices: integer
            All simplices should be elements contained in {1, ..., n_vertices-1}
        weights: dictionary, with keys the set of simplices A
            A dictionary of floats describing a weighting on the set of
            simplices.  These are normalised to sum to 1.
        """

        Simplices.__init__(self, A, n_vertices)
        if weights is None:
            weights = {}
            c = 1. / self.n_simplices
            for S in self.simplices():
                weights[S] = c
        else:
            weight_factor = 1. / sum(weights)
            weights = {k: weight_factor * weights[k] for k in weights}
        self.weights_by_dim = []
        self.weight_in_dim = []
        for k in range(self.max_dim+1):
            self.weights_by_dim.append([weights[S] for S in self.simplices_by_dim[k]])
            self.weight_in_dim.append(sum(self.weights_by_dim[-1]))


    def sample(self, samples, guarantee=False, minD=None, maxD=None, random_state=np.random.RandomState()):
        """Returns a list of points sampled, each chosen uniformly from a
        simplex, which is in turn selected according to the weights.

        Parameters
        ----------
        samples: integer
            The number of samples to generate.
        guarantee: boolean
            If true, then include one sample from each simplex, then sample
            the remaining points according to the weights.
        minD: integer
            If given will sample only from dimensions minD and over.
        maxD: integer
            If given will sample only from dimensions maxD and under.

        Returns
        -------
        sample_points: array-like, shape (samples, n_vertices)
            The sampled points.
        sample_simplices: list, length: (samples,)
            The vertex from which each point was drawn.
        sample_weights: list, length: (samples,)
            The weight associated to each sample.  The total will only sum
            to 1 if all simplices are sampled.
        """
        sample_points = np.zeros((samples, self.n_vertices))
        sample_simplices = []
        sample_weights = []
        minD = minD or 0
        maxD = maxD or self.max_dim
        if guarantee is True:
            samples -= sum(self.simplices_in_dim[minD:maxD+1])
            if samples < 0:
                raise ValueError("If there is a guarantee that each simplex "
                                "must be sampled then the number of samples "
                                "must exceed the total number of simplices.")
        weight_factor = 1. / sum(self.weight_in_dim[minD:maxD+1])
        samples_by_dim = random_state.multinomial(samples,
            [weight_factor*w for w in self.weight_in_dim[minD:maxD+1]])
        total = 0
        for k, n in enumerate(samples_by_dim):
            total_k = 0
            d = k + minD
            samples_per_simplex = random_state.multinomial(n,
                [w / self.weight_in_dim[d] for w in self.weights_by_dim[d]])
            if guarantee is True:
                samples_per_simplex += 1
                n += self.simplices_in_dim[d]
            simplices_matrix = self.k_simplices_as_matrix(d)
            simplex_samples = _random_from_simplex(dim=d, samples=n)
            for i, S in enumerate(self.simplices_by_dim[d]):
                n_s = samples_per_simplex[i]
                sample_points[total:total+n_s,...] = \
                    np.tensordot(simplex_samples[total_k:total_k+n_s,...],
                                simplices_matrix[i],
                                axes=(1,0))
                sample_simplices += [S]*n_s
                sample_weights += [self.weights_by_dim[d][i]/n_s]*n_s
                total += n_s
                total_k += n_s
        return sample_points, sample_simplices, sample_weights
