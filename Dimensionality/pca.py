import numpy as np
def PCA(data,n_components=3):
    cov_matrix = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    projection_matrix = eigenvectors[:, :n_components]
    return data.dot(projection_matrix)