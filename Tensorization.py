import numpy as np

def factorize(n):
    """Разложение числа на простые множители"""
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
    """Изменяет форму массива, разбивая размерности на простые множители"""
    original_shape = arr.shape
    prime_factors = [factorize(d) for d in original_shape]
    
    # Вычисляем новую форму
    new_shape = []
    for factors in prime_factors:
        new_shape.extend(factors)
    
    # Проверяем, что общее число элементов сохраняется
    if np.prod(original_shape) != np.prod(new_shape):
        raise ValueError("Total number of elements changed during reshaping")
    
    return arr.reshape(new_shape)


def inverse_reshape_with_prime_factors(arr, original_shape):
    """
    Восстанавливает исходную форму тензора после разбиения размерностей на простые множители.
    
    Parameters:
        arr : numpy.ndarray
            Тензор с разложенными размерностями (например, (2,2,5,5,2,3) для исходного (4,25,6))
        original_shape : tuple
            Исходная форма тензора (например, (4,25,6))
            
    Returns:
        numpy.ndarray
            Тензор в исходной форме
    """
    # Получаем простые множители для каждой исходной размерности
    prime_factors = [factorize(d) for d in original_shape]
    
    # Вычисляем ожидаемую разложенную форму
    expected_factors = []
    for factors in prime_factors:
        expected_factors.extend(factors)
    
    # Проверяем совместимость форм
    if tuple(expected_factors) != arr.shape:
        raise ValueError(f"Shape mismatch. Expected {tuple(expected_factors)}, got {arr.shape}")
    
    # Восстанавливаем исходную форму
    return arr.reshape(original_shape)


original_tensor = np.random.rand(4, 25, 6)

# Разбиваем на простые множители
factored_tensor = reshape_with_prime_factors(original_tensor)
print("Разложенная форма:", factored_tensor.shape)  # (2, 2, 5, 5, 2, 3)

# Восстанавливаем исходную форму
restored_tensor = inverse_reshape_with_prime_factors(factored_tensor, original_tensor.shape)
print("Восстановленная форма:", restored_tensor.shape)  # (4, 25, 6)

# Проверка корректности восстановления
print("Данные совпадают:", np.array_equal(original_tensor, restored_tensor))  # True