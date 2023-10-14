import numpy as np

def generate_matrix_a(size):
    a = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            a[i][j] = i + j
    return a

def generate_matrix_b(size):
    b = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            b[i][j] = i * j
    return b

def matrix_multiply(a, b):
    return np.dot(a, b)

if __name__ == "__main__":
    size = 3
    a = generate_matrix_a(size)
    b = generate_matrix_b(size)

    result = matrix_multiply(a, b)

    # If you want to print the result (it's going to be large!):
    print(result)