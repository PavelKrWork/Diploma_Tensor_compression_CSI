import numpy as np
from scipy.linalg import inv
from scipy.linalg import khatri_rao

def reconstruct_tensor(lambdas, factors):
    """
    Reconstructs tensor from CPD for complex-valued data.
    
    Parameters:
    lambdas - weights (size R)
    factors - list of factor matrices [A_1, A_2, ..., A_N], each of size (I_i, R)
    
    Returns:
    Reconstructed tensor X_rec
    """
    N = len(factors)
    R = len(lambdas)
    X_rec = np.zeros([A.shape[0] for A in factors], dtype=np.complex128)
    
    for r in range(R):
        component = factors[0][:, r]
        for n in range(1, N):
            component = np.multiply.outer(component, factors[n][:, r])  # tensor outer product
        X_rec += lambdas[r] * component
    
    return X_rec

def cp_als_complex(X, R, max_iter=100, tol=1e-6):
    """
    CP-ALS for complex-valued tensors.
    
    Parameters:
    X - complex-valued input tensor
    R - rank of decomposition
    max_iter - maximum iterations
    tol - convergence tolerance
    
    Returns:
    Dictionary with:
        'lambdas' - component weights
        'factors' - factor matrices
        'errors' - reconstruction errors
    """
    dims = X.shape
    N = len(dims)
    errors = np.zeros(max_iter)

    eps = 1e-10
    
    # Initialize factor matrices
    factors = []
    for n in range(N):
        factors.append(np.random.randn(dims[n], R) + 1j*np.random.randn(dims[n], R))

    # Initialize Gram matrices
    grams = [A.conj().T @ A for A in factors]
    lambdas = np.ones(R, dtype=np.complex128)
    
    for iteration in range(max_iter):
        prev_factors = [A.copy() for A in factors]
        
        for n in range(N):
            # Compute S matrix
            S = np.ones((R, R), dtype=np.complex128)
            for m in range(N):
                if m != n:
                    S *= grams[m]
            
            # Compute Khatri-Rao product
            Z = None
            for m in range(N-1, -1, -1):
                if m != n:
                    if Z is None:
                        Z = factors[m]
                    else:
                        Z = khatri_rao(factors[m], Z)
            
            # Unfold tensor
            Xn = np.reshape(np.moveaxis(X, n, 0), (dims[n], -1))
            
            # Solve linear system
            M = Xn @ Z.conj()
            A = M @ inv(S.conj()  + eps * np.eye(R))
            
            # Normalize columns
            norms = np.linalg.norm(A, axis=0)
            lambdas = norms
            A = A / norms
            
            # Update factor and Gram matrix
            factors[n] = A
            grams[n] = A.conj().T @ A
        
        # Compute reconstruction error
        X_rec = reconstruct_tensor(lambdas, factors)
        rel_error = np.linalg.norm(X - X_rec) / np.linalg.norm(X)
        errors[iteration] = rel_error
        
        print(f'Iteration {iteration}, relative error: {rel_error:.4e}')
        
        if rel_error <= tol:
            print(f'Converged after {iteration} iterations')
            break
    
    return {'lambdas': lambdas, 'factors': factors, 'errors': errors}
