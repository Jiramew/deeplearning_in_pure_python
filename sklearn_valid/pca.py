from sklearn import decomposition
import numpy as np

pca = decomposition.PCA(2)
data = np.array([[5, 0, 1],
                 [0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1],
                 [0, 0, 1],
                 [1, 2, 1],
                 [2, 3, 4],
                 [0, 0, 1]])

pca.fit(data)
print(pca.singular_values_)
print(pca.components_)
print(pca.transform(data))
