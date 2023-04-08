import numpy as np
from sklearn.utils.extmath import randomized_svd
from numpy.linalg import norm
import warnings

from video_maker import make_video

def _soft_thresh(X, threshold):
    "Apply soft thresholding to an array"

    sign = np.sign(X)
    return np.multiply(sign, np.maximum(np.abs(X) - threshold, 0))


def norm_1(X):
    return np.sum(np.abs(X))


def _sv_thresh(X, threshold, num_svalue):
    """
    Perform singular value thresholding.
    Parameters
    ---------
    X : array of shape [n_samples, n_features]
        The input array.
    threshold : float
        The threshold for the singualar values.
    num_svalue : int
        The number of singular values to compute.
    Returns
    -------
    X_thresh : array of shape [n_samples, n_features]
        The output after performing singular value thresholding.
    grater_sv : int
        The number of singular values of `X` which were greater than
        `threshold`
    (U, s, V): tuple
        The singular value decomposition
    """
    U, s, V = randomized_svd(X, num_svalue)
    greater_sv = np.count_nonzero(s > threshold)
    s = _soft_thresh(s, threshold)
    S = np.diag(s)
    X_thresh = np.dot(U, np.dot(S, V))
    return X_thresh, greater_sv, (U, s, V)


def rpca(M, lam=None, mu=None, max_iter=1000, eps_primal=1e-7, eps_dual=1e-5,
         rho=1.6, initial_sv=10, max_mu=1e6, verbose=False, save_interval=100):
    """Implements the Robust PCA algorithm via Principal Component Pursuit [1]_
    The Robust PCA algorithm minimizes
    .. math:: \\lVert L \\rVert_* + \\lambda \\lVert S \\rVert_1
    subject to
    .. math:: M = L + S
    where :math:`\\lVert X \\rVert_1` is the sum of absolute values of the
    matrix `X`.
    The algorithm used for optimization is the "Inexact ALM" method specified
    in [2]_
    Parameters
    ----------
    M : array-like, shape (n_samples, n_features)
        The input matrix.
    lam : float, optional
        The importance given to sparsity. Increasing this parameter will yeild
        a sparser `S`. If not given it is set to :math:`\\frac{1}{\\sqrt{n}}`
        where ``n = max(n_samples, n_features)``.
    mu : float, optional
        The initial value of the penalty parameter in the Augmented Lagrangian
        Multiplier (ALM) algorithm. This controls how much attention is given
        to the constraint in each iteration of the optimization problem.
    max_iter : int, optional
        The maximum number of iterations the optimization algortihm will run
        for.
    eps_primal : float, optional
        The threshold for the primal error in the convex optimization problem.
        If the primal and the dual error fall below ``eps_primal`` and
        ``eps_dual`` respectively, the algorithm converges.
    eps_dual :  float, optinal
         The theshold for the dual error in the convex optimzation problem.
    rho : float, optional
        The ratio of the paramter ``mu`` between two successive iterations.
        For each iteration ``mu`` is updated as ``mu = mu*rho``.
    initial_sv : int, optional
        The number of singular values to compute during the first iteration.
    max_mu : float, optional
        The maximum value that ``mu`` is allowed to take.
    verbose : bool, optional
        Whether to print convergence statistics during each iteration.
    save_interval : int, optional
        The number of iterations for which the result will be saved.
    Returns
    -------
    L : array, shape (n_samples, n_features)
        The low rank component.
    S : array, shape (n_samples, n_features)
        The sparse component.
    (U, s, Vt) : tuple of arrays
        The singular value decomposition of the ``L``
    n_iter : int
        The number of iterations taken to converge.
    References
    ----------
    .. [1] : Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). 
             Robust principal component analysis?. Journal of the ACM (JACM), 58(3), 1-37.
    .. [2] : Lin, Z., Chen, M., & Ma, Y. (2010). 
             The augmented lagrange multiplier method for exact recovery of corrupted low-rank matrices. 
             arXiv preprint arXiv:1009.5055.
    """

    # This implementation follows Algorithm 5 from the paper with minor
    # modifications

    if lam is None:
        lam = 1.0/np.sqrt(max(M.shape))

    d = min(M.shape)

    # See "Choosing Parameters" paragraph in section 4
    mu = 1.25/norm(M, 2)

    # The sparse matrix
    S = np.zeros_like(M)

    # The low rank matrix
    L = np.zeros_like(M)

    # See equation 10
    J = min(norm(M, 2), np.max(np.abs(M)))
    Y = M/J

    M_fro_norm = norm(M, 'fro')

    # This variable tried to predict how many singular values will be required.
    sv = initial_sv

    for iter_ in range(max_iter):
        # See Section 4, paragraph "Order of Updating A and E" to see why
        # `S` iterate is computed before `L` ierate.
        S_old = S
        S = _soft_thresh(M - L + (Y/mu), lam/mu)
        L, svp, (U, s, V) = _sv_thresh(M - S + (Y/mu), 1/mu, sv)
        Y = Y + mu*(M - L - S)

        mu_old = mu
        mu = rho*mu
        mu = min(mu, max_mu)

        # See Equation 18
        if svp < sv:
            sv = svp + 1
        else:
            sv = svp + int(round(0.05*d))

        sv = min(sv, M.shape[0], M.shape[1])

        primal_error = norm(M - L - S, 'fro')/M_fro_norm
        dual_error = mu_old*norm(S - S_old, 'fro')/M_fro_norm

        if verbose:
            print(f'Iteration {iter_:4d}: Primal Error = {primal_error:10f} \tDual Error = {dual_error:10f}')

        if (iter_+1) % save_interval == 0:
            make_video(L, filename=f'./output/background_{iter_}.mp4')
            make_video(S, filename=f'./output/foreground_{iter_}.mp4')

        if primal_error < eps_primal and dual_error < eps_dual:
            break

    if iter_ == max_iter-1:
        warnings.warn(f'Warning: Failed to converge within {max_iter} iterations')

    n_iter = iter_
    return L, S, np.count_nonzero(s)
