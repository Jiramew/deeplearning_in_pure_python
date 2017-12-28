import numpy as np
from pca.pca import PCA


def sd_normalize(mat: np.ndarray):
    sd = np.sqrt(np.var(mat, 0))
    return mat / sd


def eigen_normalize(mat, eigen, ep):
    return mat / np.array(np.sqrt(eigen[0:2] + ep))


if __name__ == '__main__':
    data = np.array([[5, 0, 1],
                     [0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1],
                     [0, 0, 1],
                     [1, 2, 1],
                     [2, 3, 4],
                     [0, 0, 1]])

    pca = PCA(data)
    out, eigen, vector = pca.fit()
    pca_whitening_mat = sd_normalize(out)
    pca_whitening_mat2 = eigen_normalize(out, eigen, 0.01)
    print(pca_whitening_mat)
    a = 1
