import numpy as np
from numpy.linalg import svd

def rpca(X, lmbda=1.0, rank=10, tol=1e-6, max_iter=1000):
    """
    Computes the Robust Principal Component Analysis (RPCA) of a data matrix X using the augmented Lagrange multiplier method.

    Args:
        X (numpy.ndarray): A 2D array of size (n_samples, n_features) representing the input data matrix.
        lmbda (float, optional): Regularization parameter for the sparse component. Default is 1.0.
        rank (int, optional): Rank of the low-rank component. If None, it is set to the minimum between the number of samples and the number of features. Default is 10.
        tol (float, optional): Tolerance for convergence of the optimization algorithm. Default is 1e-6.
        max_iter (int, optional): Maximum number of iterations for the optimization algorithm. Default is 1000.

    Returns:
        numpy.ndarray: A 2D array of size (n_samples, n_features) representing the low-rank component.
        numpy.ndarray: A 2D array of size (n_samples, n_features) representing the sparse component.
    """
    m, n = X.shape
    Y = X.copy()
    L = np.zeros((m, n))
    S = np.zeros((m, n))
    converged = False
    i = 0

    while not converged:
        L_prev = L.copy()
        S_prev = S.copy()

        # update L using the low-rank approximation
        U, sigma, Vt = svd(Y - S, full_matrices=False)
        L = np.dot(U[:, :rank], np.dot(np.diag(sigma[:rank]), Vt[:rank, :]))

        # update S using the soft thresholding operator
        S = np.sign(Y - L) * np.maximum(np.abs(Y - L) - lmbda, 0)

        # check for convergence
        delta_L = np.linalg.norm(L - L_prev)
        delta_S = np.linalg.norm(S - S_prev)
        converged = (delta_L < tol) and (delta_S < tol)

        # check for max iterations
        i += 1
        if i >= max_iter:
            converged = True

    return L, S
