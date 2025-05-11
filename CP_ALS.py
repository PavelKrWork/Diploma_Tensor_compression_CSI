import numpy as np
from scipy.linalg import qr, pinv
from scipy.linalg import khatri_rao


def reconstruct_tensor(lambdas, factors):
    """
    Reconstructs tensor from CPD for complex-valued data.
    
    Parameters:
    lambdas - weights (size R)
    factors - list of factor matrices [A_1, A_2, ..., A_N], each of size (I_i, R)
    
    Returns:
    Reconstructed tensor X_hat
    """
    N = len(factors)
    R = len(lambdas)
    X_rec = np.zeros([A.shape[0] for A in factors], dtype=np.complex128)
    
    for r in range(R):
        component = factors[0][:, r]
        for n in range(1, N):
            component = np.multiply.outer(component, factors[n][:, r])
        X_rec += lambdas[r] * component
    
    return X_rec

def cp_als_complex(X, R, max_iter=100, tol=1e-6, init='random'):
    """
    CP-ALS for complex-valued tensors.
    
    Parameters:
    X - complex-valued input tensor
    R - rank of decomposition
    max_iter - maximum iterations
    tol - convergence tolerance
    init - initialization method ('random' or 'svd')
    
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
    if init == 'random':
        for n in range(N):
            factors.append(np.random.randn(dims[n], R) + 1j*np.random.randn(dims[n], R))
    elif init == 'svd':
        for n in range(N):
            Xn = np.reshape(np.moveaxis(X, n, 0), (dims[n], -1))
            U, S, V = np.linalg.svd(Xn, full_matrices=False)
            factors.append(U[:, :R].astype(np.complex128))
    else:
        raise ValueError("Unknown initialization method")
    
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
            A = M @ pinv(S.T + eps * np.eye(R))
            
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


def cp_als_qr_complex(X, R, max_iter=100, tol=1e-6, init='random'):
    """
    QR-accelerated CP-ALS for complex-valued tensors.
    
    Parameters:
    X - complex-valued input tensor
    R - rank of decomposition
    max_iter - maximum iterations
    tol - convergence tolerance
    init - initialization method ('random' or 'svd')
    
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
    
    # Initialize with QR decomposition
    factors = []
    Q_list = []
    R_list = []
    
    if init == 'random':
        for n in range(N):
            A = np.random.randn(dims[n], R) + 1j*np.random.randn(dims[n], R)
            Q, R_mat = qr(A, mode='economic')
            factors.append(A)
            Q_list.append(Q)
            R_list.append(R_mat)
  
    else:
        raise ValueError("Unknown initialization method")
    
    lambdas = np.ones(R, dtype=np.complex128)
    
    for iteration in range(max_iter):
        prev_factors = [A.copy() for A in factors]
        
        for n in range(N):
            # Compute Vn via Khatri-Rao product of R matrices
            Vn = None
            for m in range(N-1, -1, -1):
                if m != n:
                    if Vn is None:
                        Vn = R_list[m]
                    else:
                        Vn = khatri_rao(R_list[m], Vn)
            
            # QR decomposition of Vn
            Q0, R0 = qr(Vn, mode='economic')
            
            # Compute Y(n) = X(n) * Khatri-Rao product of Q matrices
            Qt = None
            for m in range(N-1, -1, -1):
                if m != n:
                    if Qt is None:
                        Qt = Q_list[m]
                    else:
                        Qt = np.kron(Q_list[m], Qt)
            
            Xn = np.reshape(np.moveaxis(X, n, 0), (dims[n], -1))
            Yn = Xn @ Qt.conj()
            Wn = Yn @ Q0.conj()
            
            # Solve linear system

            A = Wn @ pinv(R0.conj().T + eps*np.eye(R))
            
            # Normalize columns
            norms = np.linalg.norm(A, axis=0)
            lambdas = norms
            A = A / norms
            
            # Update QR decomposition
            Q, R_mat = qr(A, mode='economic')
            factors[n] = A
            Q_list[n] = Q
            R_list[n] = R_mat
        
        # Compute reconstruction error
        X_rec = reconstruct_tensor(lambdas, factors)
        rel_error = np.linalg.norm(X - X_rec) / np.linalg.norm(X)
        errors[iteration] = rel_error
        
        print(f'Iteration {iteration}, relative error: {rel_error:.4e}')
        
        if rel_error <= tol:
            print(f'Converged after {iteration} iterations')
            break
    
    return {'lambdas': lambdas, 'factors': factors, 'errors': errors}


# def Tacker():
#     pass

# def cpd_gradient_descent(X, rank, lr=0.01, max_iter=100, tol=1e-6):
#     """
#     Каноническое разложение тензора через градиентный спуск.
    
#     Параметры:
#         X : numpy.ndarray
#             Тензор размерности (I, J, K).
#         rank : int
#             Ранг разложения R.
#         lr : float
#             Скорость обучения.
#         max_iter : int
#             Максимальное число итераций.
#         tol : float
#             Допустимая ошибка для остановки.
            
#     Возвращает:
#         factors : tuple of numpy.ndarray
#             (A, B, C), где A (I×R), B (J×R), C (K×R).
#     """
#     I, J, K = X.shape
#     R = rank
    
#     # Инициализация случайными значениями
#     A = np.random.randn(I, R)
#     B = np.random.randn(J, R)
#     C = np.random.randn(K, R)
    
#     for step in range(max_iter):
#         # Восстановленный тензор
#         X_rec = np.einsum('ir,jr,kr->ijk', A, B, C)
        
#         # Ошибка
#         error = np.linalg.norm(X - X_rec)
#         if error < tol:
#             print(f"Сходимость достигнута на шаге {step}.")
#             break
        
#         # Градиенты
#         grad_A = -2 * np.einsum('ijk,jr,kr->ir', X - X_rec, B, C)
#         grad_B = -2 * np.einsum('ijk,ir,kr->jr', X - X_rec, A, C)
#         grad_C = -2 * np.einsum('ijk,ir,jr->kr', X - X_rec, A, B)
        
#         # Обновление параметров
#         A -= lr * grad_A
#         B -= lr * grad_B
#         C -= lr * grad_C
        
#         if step % 100 == 0:
#             print(f"Шаг {step}, ошибка: {error:.4f}")
    
#     return A, B, C

