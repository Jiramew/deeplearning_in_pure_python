import numpy as np


class PCA(object):
    def __init__(self,
                 mat,
                 per=0.85,
                 nic=2):
        self.mat = mat
        self.input_dim = self.mat.shape[
            1]  # number of nodes in input unit(train data) not including bias node aka. the dimension of train data.
        self.input_num = self.mat.shape[0]  # number of instance

        self.percentage = per
        self.number_into_count = nic

    def svd(self):
        cov = self.cov()  # or directly from mat like sklearn
        U, sigma, VT = np.linalg.svd(cov)
        eigen_sorted = np.argsort(sigma)
        print(np.sqrt(sigma))
        if self.number_into_count is not None:
            eigen_vector = U[:, eigen_sorted[:-self.number_into_count - 1:-1]]
            return eigen_vector

    def normalize(self):
        return self.mat - np.mean(self.mat, 0)

    def cov(self):
        mat_pre = self.normalize()
        return np.dot(mat_pre.T, mat_pre) / (self.input_dim - 1)

    def fit(self):
        vector = self.svd()
        print(vector)
        return np.dot(self.normalize(), vector)


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
    print(pca.fit())
