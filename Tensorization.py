import numpy as np

def factorize(n):
    """Prime factorization of a number"""
    factors = []
    i = 2
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n = n // i
        i += 1
    if n > 1:
        factors.append(n)
    return factors


def reshape_with_prime_factors(arr):
    """Changes the shape of an array by breaking its dimensions into prime factors"""
    original_shape = arr.shape
    prime_factors = [factorize(d) for d in original_shape]
    
    new_shape = []
    for factors in prime_factors:
        new_shape.extend(factors)
    
    if np.prod(original_shape) != np.prod(new_shape):
        raise ValueError("Total number of elements changed during reshaping")
    
    return arr.reshape(new_shape)


def inverse_reshape_with_prime_factors(arr, original_shape):
    """
    Parameters:
        arr : numpy.ndarray
            Tensor with decomposed dimensions (e.g. (2,2,5,5,2,3) for the original (4,25,6))
        original_shape : tuple
           The original form of the tensor (eg (4,25,6))
            
    Returns:
        numpy.ndarray
            Tensor in original form
    """
    prime_factors = [factorize(d) for d in original_shape]
    
    expected_factors = []
    for factors in prime_factors:
        expected_factors.extend(factors)
    
    if tuple(expected_factors) != arr.shape:
        raise ValueError(f"Shape mismatch. Expected {tuple(expected_factors)}, got {arr.shape}")
    
    return arr.reshape(original_shape)
