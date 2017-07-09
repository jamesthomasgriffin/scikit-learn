"""
The :mod:`sklearn.mixture` module implements mixture modeling algorithms.
"""

from .gmm import sample_gaussian, log_multivariate_normal_density
from .gmm import GMM, distribute_covar_matrix_to_match_covariance_type
from .gmm import _validate_covars
from .dpgmm import DPGMM, VBGMM

from .gaussian_mixture import GaussianMixture
from .bayesian_mixture import BayesianGaussianMixture
from .simplicial_gaussian_mixture import GaussianSimplicialMixture
from .simplices import Simplices
from .simplicial_base import _random_from_simplex


__all__ = ['DPGMM',
           'GMM',
           'VBGMM',
           '_validate_covars',
           'distribute_covar_matrix_to_match_covariance_type',
           'log_multivariate_normal_density',
           'sample_gaussian',
           'GaussianMixture',
           'BayesianGaussianMixture',
           'GaussianSimplicialMixture',
           'Simplices']
