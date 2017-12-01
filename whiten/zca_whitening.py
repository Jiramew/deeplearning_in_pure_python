import numpy as np

if __name__ == '__main__':
    data = np.array([[5, 0, 1],
                     [0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1],
                     [0, 0, 1],
                     [1, 2, 1],
                     [2, 3, 4],
                     [0, 0, 1]])

    data = data - np.mean(data, axis=0)

    sigma = np.dot(data, data.T) / (data.shape[0])
    (U, S, V) = np.linalg.svd(sigma)

    ZCA_white = np.dot(np.dot(U, np.diag(1 / np.sqrt(S + 0.1))), U.transpose())
    patch_ZCAwhite = np.dot(ZCA_white, data)
    print(patch_ZCAwhite)
